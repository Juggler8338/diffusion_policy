from typing import Union, Optional, Tuple
import logging
import torch
import torch.nn as nn
from diffusion_policy.model.diffusion.positional_embedding import SinusoidalPosEmb
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin

logger = logging.getLogger(__name__)

def modulate(x, shift, scale):
    """标准的 AdaLN 调制函数"""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class DiTBlock(nn.Module):
    """包含 AdaLN 的 DiT Block"""
    def __init__(self, n_emb, n_head, p_drop_attn):
        super().__init__()
        self.norm1 = nn.LayerNorm(n_emb, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(n_emb, n_head, dropout=p_drop_attn, batch_first=True)
        self.norm2 = nn.LayerNorm(n_emb, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(n_emb, 4 * n_emb),
            nn.GELU(),
            nn.Linear(4 * n_emb, n_emb)
        )
        # 用于调制参数的线性层：每个 block 需要为 norm1 和 norm2 分别生成 shift, scale
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(n_emb, 4 * n_emb) 
        )

    def forward(self, x, c):
        # 从条件向量 c 中预测 shift 和 scale
        # split 为 (shift_msa, scale_msa, shift_mlp, scale_mlp)
        shift_msa, scale_msa, shift_mlp, scale_mlp = self.adaLN_modulation(c).chunk(4, dim=1)
        
        # Attention 部分
        x = x + self.attn(modulate(self.norm1(x), shift_msa, scale_msa), 
                          modulate(self.norm1(x), shift_msa, scale_msa), 
                          modulate(self.norm1(x), shift_msa, scale_msa))[0]
        # MLP 部分
        x = x + self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

class TransformerForDiffusion(ModuleAttrMixin):
    def __init__(self,
            input_dim: int,
            output_dim: int,
            horizon: int,
            n_obs_steps: int = None,
            cond_dim: int = 0,
            n_layer: int = 12,
            n_head: int = 12,
            n_emb: int = 768,
            p_drop_emb: float = 0.1,
            p_drop_attn: float = 0.1,
            causal_attn: bool = False,
            time_as_cond: bool = True, # DiT 默认均为 True
            obs_as_cond: bool = True,
            **kwargs
        ) -> None:
        super().__init__()

        self.input_emb = nn.Linear(input_dim, n_emb)
        self.pos_emb = nn.Parameter(torch.zeros(1, horizon, n_emb))
        self.drop = nn.Dropout(p_drop_emb)

        # 条件编码器：合并时间步和观测信息
        self.time_emb = SinusoidalPosEmb(n_emb)
        if cond_dim > 0:
            # 将观测序列展平或池化处理，这里采用简单的线性映射以匹配维度
            self.cond_obs_emb = nn.Sequential(
                nn.Linear(cond_dim * n_obs_steps, n_emb),
                nn.SiLU(),
                nn.Linear(n_emb, n_emb)
            )
        else:
            self.cond_obs_emb = None

        # DiT 主干网络
        self.blocks = nn.ModuleList([
            DiTBlock(n_emb, n_head, p_drop_attn) for _ in range(n_layer)
        ])
        
        # 最后的输出层
        self.ln_f = nn.LayerNorm(n_emb, elementwise_affine=False, eps=1e-6)
        self.adaLN_final = nn.Sequential(
            nn.SiLU(),
            nn.Linear(n_emb, 2 * n_emb)
        )
        self.head = nn.Linear(n_emb, output_dim)

        # 这里的 mask 仅用于 causal 场景
        if causal_attn:
            mask = torch.triu(torch.ones(horizon, horizon), diagonal=1).bool()
            self.register_buffer("mask", mask)
        else:
            self.mask = None

        self.apply(self._init_weights)
        logger.info("DiT parameters: %e", sum(p.numel() for p in self.parameters()))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            if module.elementwise_affine:
                torch.nn.init.zeros_(module.bias)
                torch.nn.init.ones_(module.weight)

    def forward(self, sample, timestep, cond=None, **kwargs):
        """
        sample: (B, T, input_dim)
        timestep: (B,)
        cond: (B, To, cond_dim)
        """
        # 1. 准备条件向量 c
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        timesteps = timesteps.expand(sample.shape[0])
        
        c = self.time_emb(timesteps) # (B, n_emb)
        
        if cond is not None and self.cond_obs_emb is not None:
            # 将历史观测信息合并进条件向量
            obs_feat = self.cond_obs_emb(cond.view(cond.shape[0], -1))
            c = c + obs_feat

        # 2. 准备输入 Token
        x = self.input_emb(sample) + self.pos_emb
        x = self.drop(x)

        # 3. 通过 DiT Blocks (AdaLN 注入)
        for block in self.blocks:
            x = block(x, c)

        # 4. 最后的调制与输出
        shift, scale = self.adaLN_final(c).chunk(2, dim=1)
        x = modulate(self.ln_f(x), shift, scale)
        x = self.head(x)
        return x

    def get_optim_groups(self, weight_decay: float=1e-3):
        # 保持原有的优化器分组逻辑
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.MultiheadAttention)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn
                if pn.endswith("bias"):
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)
        
        no_decay.add("pos_emb")
        param_dict = {pn: p for pn, p in self.named_parameters()}
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        return optim_groups

    def configure_optimizers(self, learning_rate: float=1e-4, weight_decay: float=1e-3, betas: Tuple[float, float]=(0.9,0.95)):
        optim_groups = self.get_optim_groups(weight_decay=weight_decay)
        return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)