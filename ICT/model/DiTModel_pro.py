# DiTModel.py
from typing import Union, Optional, Tuple
import logging
import torch
import torch.nn as nn
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin

logger = logging.getLogger(__name__)

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class GaussianFourierProjection(nn.Module):
    def __init__(self, embed_dim: int, scale: float = 2.0 * torch.pi):
        super().__init__()
        self.W = nn.Parameter(torch.randn(1, embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x):
        if x.ndim == 0:
            x = x.unsqueeze(0)
        x_proj = x[:, None] * self.W
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
    
class DiTBlock(nn.Module):
    """完整 AdaLN-Zero Block（DiT 论文标准实现）"""
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
        # 输出 6 个调制参数：shift/scale/gate × 2
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(n_emb, 6 * n_emb)
        )

    def forward(self, x, c):
        mod = self.adaLN_modulation(c)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=1)
        
        # Attn
        attn_out = self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa),
            modulate(self.norm1(x), shift_msa, scale_msa),
            modulate(self.norm1(x), shift_msa, scale_msa)
        )[0]
        x = x + gate_msa.unsqueeze(1) * attn_out
        
        # MLP
        mlp_out = self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        x = x + gate_mlp.unsqueeze(1) * mlp_out
        
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
            time_as_cond: bool = True,
            obs_as_cond: bool = True,
            **kwargs
        ) -> None:
        super().__init__()

        self.input_emb = nn.Linear(input_dim, n_emb)
        self.pos_emb = nn.Parameter(torch.zeros(1, horizon, n_emb))
        self.drop = nn.Dropout(p_drop_emb)

        self.time_emb = GaussianFourierProjection(n_emb)

        # 更稳定的 obs conditioning：mean pool over time
        if cond_dim > 0:
            self.cond_obs_emb = nn.Linear(cond_dim, n_emb)
        else:
            self.cond_obs_emb = None

        self.blocks = nn.ModuleList([
            DiTBlock(n_emb, n_head, p_drop_attn) for _ in range(n_layer)
        ])
        
        self.ln_f = nn.LayerNorm(n_emb, elementwise_affine=False, eps=1e-6)
        self.adaLN_final = nn.Sequential(
            nn.SiLU(),
            nn.Linear(n_emb, 2 * n_emb)  # 只输出 shift/scale（无 gate）
        )
        self.head = nn.Linear(n_emb, output_dim)

        self.apply(self._init_weights)
        # 关键：zero-init 所有 gate 参数
        for block in self.blocks:
            nn.init.zeros_(block.adaLN_modulation[-1].bias)

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
        # timestep 为 float [0,1]
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], dtype=torch.float32, device=sample.device)
        elif timestep.ndim == 0:
            timestep = timestep.unsqueeze(0)
        timestep = timestep.to(sample.device).float().expand(sample.shape[0])

        c = self.time_emb(timestep)

        if cond is not None and self.cond_obs_emb is not None:
            obs_feat = cond.mean(dim=1)  # (B, To, cond_dim) -> (B, cond_dim)
            obs_feat = self.cond_obs_emb(obs_feat)
            c = c + obs_feat

        x = self.input_emb(sample) + self.pos_emb
        x = self.drop(x)

        for block in self.blocks:
            x = block(x, c)

        shift, scale = self.adaLN_final(c).chunk(2, dim=1)
        x = modulate(self.ln_f(x), shift, scale)
        x = self.head(x)
        return x

    def get_optim_groups(self, weight_decay: float=1e-3):
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