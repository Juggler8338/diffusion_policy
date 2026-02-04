from typing import Union, Optional, Tuple
import logging
import torch
import torch.nn as nn
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin

logger = logging.getLogger(__name__)

def modulate(x, shift, scale):
    """标准的 AdaLN 调制函数"""
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
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(n_emb, 4 * n_emb) 
        )

    def forward(self, x, c, attn_mask=None):
        # x: (B, T, D) or (B, S+T, D) after ICT injection
        # c: (B, D) global condition (usually Time)
        shift_msa, scale_msa, shift_mlp, scale_mlp = self.adaLN_modulation(c).chunk(4, dim=1)
        
        # Attention
        # Note: attn_mask logic needs to be handled carefully when sequence length changes
        attn_out, _ = self.attn(
            query=modulate(self.norm1(x), shift_msa, scale_msa),
            key=modulate(self.norm1(x), shift_msa, scale_msa),
            value=modulate(self.norm1(x), shift_msa, scale_msa),
            attn_mask=attn_mask
        )
        x = x + attn_out
        
        # MLP
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
            time_as_cond: bool = True,
            obs_as_cond: bool = True,
            ict_start_layer: int = 4, # 默认从第4层开始注入观测 Token
            **kwargs
        ) -> None:
        super().__init__()

        # Input Embedding (Action)
        self.input_emb = nn.Linear(input_dim, n_emb)
        self.pos_emb = nn.Parameter(torch.zeros(1, horizon, n_emb))
        self.drop = nn.Dropout(p_drop_emb)

        # Time Embedding (作为 Global AdaLN condition)
        self.time_emb = GaussianFourierProjection(n_emb)
        
        # In-Context Token (ICT) Configuration
        self.cond_dim = cond_dim
        self.n_obs_steps = n_obs_steps
        self.ict_start_layer = ict_start_layer
        
        # 定义 ICT 投影层：将 (cond_dim) 映射为 (n_emb)
        # 注意：这里我们保留时序结构，不做 Flatten
        if cond_dim > 0:
            self.ict_projector = nn.Sequential(
                nn.Linear(cond_dim, n_emb),
                nn.Mish(),
                nn.Linear(n_emb, n_emb)
            )
            # 为观测序列准备的位置编码
            self.ict_pos_emb = nn.Parameter(torch.zeros(1, n_obs_steps, n_emb))
            torch.nn.init.normal_(self.ict_pos_emb, std=0.02)
        else:
            self.ict_projector = None

        # Blocks
        self.blocks = nn.ModuleList([
            DiTBlock(n_emb, n_head, p_drop_attn) for _ in range(n_layer)
        ])
        
        # Final Layer
        self.ln_f = nn.LayerNorm(n_emb, elementwise_affine=False, eps=1e-6)
        self.adaLN_final = nn.Sequential(
            nn.SiLU(),
            nn.Linear(n_emb, 2 * n_emb)
        )
        self.head = nn.Linear(n_emb, output_dim)

        # Causal Mask caching
        self.causal_attn = causal_attn
        self.horizon = horizon
        self.register_buffer("causal_mask", self._generate_causal_mask(horizon))
        self.ict_mask = None # run-time generated/cached

        # Init
        self.apply(self._init_weights)
        logger.info("DiT-ICT parameters: %e", sum(p.numel() for p in self.parameters()))

    def _generate_causal_mask(self, size):
        # 1 means mask (ignore), 0 means attend
        # PyTorch MHA expects boolean mask where True means IGNORE or float mask with -inf
        # Here we use float mask for stability with custom implementations usually
        mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
        return mask

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
        sample: (B, T, input_dim) -> Actions (Noisy)
        timestep: (B,)
        cond: (B, T_obs, cond_dim) -> Observations
        """
        B, T, _ = sample.shape
        
        # 1. Global Condition c (Time Embedding)
        # 在 JiT 中，class label 被加到了 global c 中。
        # 这里为了简单和稳定，global c 只包含 Time。
        # 观测信息 (Obs) 将作为 Token 在中间层注入。
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        timesteps = timesteps.expand(B)
        
        c = self.time_emb(timesteps) # (B, n_emb)

        # 2. Action Input Embedding
        x = self.input_emb(sample) + self.pos_emb
        x = self.drop(x)

        # 3. 准备 ICT (In-Context Tokens)
        ict_tokens = None
        if cond is not None and self.ict_projector is not None:
            # cond: (B, T_obs, cond_dim)
            # project to (B, T_obs, n_emb)
            ict_tokens = self.ict_projector(cond)
            ict_tokens = ict_tokens + self.ict_pos_emb
        
        # 4. Forward Pass with Injection
        current_mask = self.causal_attn and self.causal_mask # Initial mask (T, T)

        for i, block in enumerate(self.blocks):
            
            # --- ICT Injection Logic ---
            if ict_tokens is not None and i == self.ict_start_layer:
                # 拼接: [Obs Tokens, Action Tokens]
                x = torch.cat([ict_tokens, x], dim=1)
                
                # 更新 Mask
                if self.causal_attn:
                    # 我们需要构造一个新的 Mask (S+T, S+T)
                    # S = T_obs, T = Horizon
                    S = ict_tokens.shape[1]
                    total_len = S + T
                    
                    # 构造 Mask 矩阵
                    # Region 1 (SxS): Obs 关注 Obs -> 全 0 (允许关注)
                    # Region 2 (SxT): Obs 关注 Action -> 全 1 (Mask掉，Obs不该看未来的Action)
                    # Region 3 (TxS): Action 关注 Obs -> 全 0 (允许关注)
                    # Region 4 (TxT): Action 关注 Action -> Causal Mask
                    
                    new_mask = torch.zeros((total_len, total_len), device=x.device, dtype=torch.bool)
                    
                    # Masking upper right (Obs seeing Action)
                    new_mask[:S, S:] = True 
                    
                    # Masking bottom right (Action seeing future Action)
                    new_mask[S:, S:] = self.causal_mask
                    
                    current_mask = new_mask
                else:
                    current_mask = None # 非 Causal 模式通常允许全关注

            # --- Block Forward ---
            x = block(x, c, attn_mask=current_mask)

        # 5. Post-process
        # 如果注入了 ICT，Action Token 在后半部分
        if ict_tokens is not None and len(self.blocks) > self.ict_start_layer:
            S = ict_tokens.shape[1]
            x = x[:, S:] # Slice off the observation tokens

        # 6. Final Head
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
    


