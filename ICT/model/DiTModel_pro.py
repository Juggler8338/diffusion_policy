# DiTModel.py
# - 采用更细粒度的 AdaLN 调制（attn 和 mlp 分别有独立的 shift/scale/gate）
# - 时间嵌入改为 learnable sinusoidal（_TimeNetwork）
# - Final layer 采用 LeRobot 的 AdaLN 风格

from typing import Union, Optional, Tuple
import logging
import torch
import torch.nn as nn
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin
import numpy as np
logger = logging.getLogger(__name__)

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class _TimeNetwork(nn.Module):
    def __init__(self, frequency_embedding_dim, hidden_dim, learnable_w=False, max_period=1000):
        assert frequency_embedding_dim % 2 == 0
        half_dim = frequency_embedding_dim // 2
        super().__init__()
        w = np.log(max_period) / (half_dim - 1)
        w = torch.exp(torch.arange(half_dim) * -w).float()
        self.register_parameter("w", nn.Parameter(w, requires_grad=learnable_w))
        self.out_net = nn.Sequential(
            nn.Linear(frequency_embedding_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, t):
        assert len(t.shape) == 1
        t = t[:, None] * self.w[None]
        t = torch.cat((torch.cos(t), torch.sin(t)), dim=1)
        return self.out_net(t)
    
# 在 ICT/model/DiTModel_pro.py 中修复所有 modulation 层的广播维度
# 因为我们使用 batch_first=True，序列维度在 dim=1，需要 unsqueeze(1) 而非 unsqueeze(0) 或 unsqueeze(-1)
# 修改 _ShiftScaleMod
class _ShiftScaleMod(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.act = nn.SiLU()
        self.scale = nn.Linear(dim, dim)
        self.shift = nn.Linear(dim, dim)

    def forward(self, x, c):
        c = self.act(c)  # (B, dim)
        scale = self.scale(c).unsqueeze(1)  # (B, 1, dim)
        shift = self.shift(c).unsqueeze(1)  # (B, 1, dim)
        return x * (1 + scale) + shift

    def reset_parameters(self):
        nn.init.zeros_(self.scale.weight)
        nn.init.zeros_(self.shift.weight)
        nn.init.zeros_(self.scale.bias)
        nn.init.zeros_(self.shift.bias)

# 修改 _ZeroScaleMod（只有 scale）
class _ZeroScaleMod(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.act = nn.SiLU()
        self.scale = nn.Linear(dim, dim)

    def forward(self, x, c):
        c = self.act(c)  # (B, dim)
        scale = self.scale(c).unsqueeze(1)  # (B, 1, dim)
        return x * scale

    def reset_parameters(self):
        nn.init.zeros_(self.scale.weight)
        nn.init.zeros_(self.scale.bias)

class DiTBlock(nn.Module):
    """LeRobot 风格的细粒度 AdaLN-Zero Block"""
    def __init__(self, n_emb, n_head, p_drop_attn=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(n_emb, eps=1e-6, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(n_emb, eps=1e-6, elementwise_affine=False)
        
        self.attn = nn.MultiheadAttention(n_emb, n_head, dropout=p_drop_attn, batch_first=True)
        
        self.linear1 = nn.Linear(n_emb, 4 * n_emb)
        self.linear2 = nn.Linear(4 * n_emb, n_emb)
        self.act = nn.GELU()
        
        # 独立的调制层
        self.attn_modulate = _ShiftScaleMod(n_emb)
        self.attn_gate = _ZeroScaleMod(n_emb)
        self.mlp_modulate = _ShiftScaleMod(n_emb)
        self.mlp_gate = _ZeroScaleMod(n_emb)

    def forward(self, x, c):  # c = time_emb + cond_emb
        # Attn branch
        x_res = self.attn_modulate(self.norm1(x), c)
        attn_out, _ = self.attn(x_res, x_res, x_res)
        x = x + self.attn_gate(attn_out, c)
        
        # MLP branch
        x_res = self.mlp_modulate(self.norm2(x), c)
        mlp_out = self.linear2(self.act(self.linear1(x_res)))
        x = x + self.mlp_gate(mlp_out, c)
        
        return x

class FinalLayer(nn.Module):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.linear = nn.Linear(hidden_size, out_size)
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))

    def forward(self, x, c):
        shift, scale = self.adaLN(c).chunk(2, dim=-1)
        x = modulate(self.norm(x), shift, scale)
        return self.linear(x)

class TransformerForDiffusion(ModuleAttrMixin):
    def __init__(self,
            input_dim: int,
            output_dim: int,
            horizon: int,
            cond_dim: int = 0,  # 现在是 n_obs_steps * per_step_dim
            n_layer: int = 12,
            n_head: int = 12,
            n_emb: int = 768,
            p_drop_emb: float = 0.1,
            p_drop_attn: float = 0.1,
            time_dim: int = 256,
            **kwargs
        ) -> None:
        super().__init__()

        self.n_emb = n_emb
        self.horizon = horizon
        
        self.input_emb = nn.Linear(input_dim, n_emb)
        self.pos_emb = nn.Parameter(torch.zeros(1, horizon, n_emb))
        nn.init.xavier_uniform_(self.pos_emb)
        
        self.time_emb = _TimeNetwork(time_dim, n_emb)
        
        if cond_dim > 0:
            self.cond_proj = nn.Linear(cond_dim, n_emb)
        else:
            self.cond_proj = None
            
        self.drop = nn.Dropout(p_drop_emb)

        self.blocks = nn.ModuleList([
            DiTBlock(n_emb, n_head, p_drop_attn) for _ in range(n_layer)
        ])
        
        self.final_layer = FinalLayer(n_emb, output_dim)

        self.apply(self._init_weights)
        # zero-init gates
        for block in self.blocks:
            block.attn_gate.reset_parameters()
            block.mlp_gate.reset_parameters()

        logger.info("DiT parameters: %e", sum(p.numel() for p in self.parameters()))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            if module.elementwise_affine:
                nn.init.zeros_(module.bias)
                nn.init.ones_(module.weight)

    def forward(self, sample, timestep, cond=None):
        # timestep: (B,) float in [0,1]
        time_emb = self.time_emb(timestep)
        
        if cond is not None and self.cond_proj is not None:
            cond_emb = self.cond_proj(cond)  # (B, n_emb)
            c = time_emb + cond_emb
        else:
            c = time_emb

        x = self.input_emb(sample) + self.pos_emb
        x = self.drop(x)
        
        for block in self.blocks:
            x = block(x, c)
            
        x = self.final_layer(x, c)
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