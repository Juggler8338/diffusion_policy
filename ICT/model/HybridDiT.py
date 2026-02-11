import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ==========================================
# 1. 基础组件 (Time Embedding, Modulation, Final Layer)
# ==========================================

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class _TimeNetwork(nn.Module):
    def __init__(self, frequency_embedding_dim, hidden_dim, learnable_w=False, max_period=1000):
        super().__init__()
        assert frequency_embedding_dim % 2 == 0
        half_dim = frequency_embedding_dim // 2
        w = np.log(max_period) / (half_dim - 1)
        w = torch.exp(torch.arange(half_dim) * -w).float()
        self.register_parameter("w", nn.Parameter(w, requires_grad=learnable_w))
        self.out_net = nn.Sequential(
            nn.Linear(frequency_embedding_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, t):
        # t: (B,)
        if len(t.shape) == 1:
            t = t[:, None] * self.w[None]
        else:
            t = t * self.w[None]
        t = torch.cat((torch.cos(t), torch.sin(t)), dim=-1)
        return self.out_net(t)

class _ShiftScaleMod(nn.Module):
    """用于 Norm 后的 Shift/Scale 调制"""
    def __init__(self, dim):
        super().__init__()
        self.act = nn.SiLU()
        self.scale = nn.Linear(dim, dim)
        self.shift = nn.Linear(dim, dim)

    def forward(self, x, c):
        c = self.act(c)
        scale = self.scale(c).unsqueeze(1)
        shift = self.shift(c).unsqueeze(1)
        return x * (1 + scale) + shift

    def reset_parameters(self):
        nn.init.zeros_(self.scale.weight)
        nn.init.zeros_(self.shift.weight)
        nn.init.zeros_(self.scale.bias)
        nn.init.zeros_(self.shift.bias)

class _ZeroScaleMod(nn.Module):
    """用于残差连接前的 Gating (初始化为0)"""
    def __init__(self, dim):
        super().__init__()
        self.act = nn.SiLU()
        self.scale = nn.Linear(dim, dim)

    def forward(self, x, c):
        c = self.act(c)
        scale = self.scale(c).unsqueeze(1)
        return x * scale

    def reset_parameters(self):
        nn.init.zeros_(self.scale.weight)
        nn.init.zeros_(self.scale.bias)

class FinalLayer(nn.Module):
    """输出层，同样使用 AdaLN"""
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.linear = nn.Linear(hidden_size, out_size)
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))

    def forward(self, x, c):
        shift, scale = self.adaLN(c).chunk(2, dim=-1)
        x = modulate(self.norm(x), shift, scale)
        return self.linear(x)

# ==========================================
# 2. 新增组件: Condition Encoder
# ==========================================

class ConditionEncoder(nn.Module):
    """
    负责处理序列 Condition (如历史观测)
    Output: memory (B, T_cond, n_emb)
    """
    def __init__(self, cond_dim, n_emb, n_layer, n_head, max_seq_len=100):
        super().__init__()
        self.input_proj = nn.Linear(cond_dim, n_emb)
        # 固定正弦位置编码或可学习位置编码，这里使用可学习以匹配 DiT 风格
        self.pos_emb = nn.Parameter(torch.zeros(1, max_seq_len, n_emb))
        nn.init.xavier_uniform_(self.pos_emb)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=n_emb,
            nhead=n_head,
            dim_feedforward=4*n_emb,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)

    def forward(self, cond):
        # cond: (B, T_cond, cond_dim)
        x = self.input_proj(cond)
        # 截断位置编码以匹配输入长度
        seq_len = x.shape[1]
        x = x + self.pos_emb[:, :seq_len, :]
        memory = self.encoder(x)
        return memory

# ==========================================
# 3. 核心修改: DiT Decoder Block (Hybrid)
# ==========================================

class DiTDecoderBlock(nn.Module):
    """
    结构: 
    1. Self-Attention (AdaLN modulated)
    2. Cross-Attention (AdaLN modulated, Queries Condition Memory)
    3. MLP (AdaLN modulated)
    """
    def __init__(self, n_emb, n_head, p_drop_attn=0.0):
        super().__init__()
        
        # --- Self Attention Branch ---
        self.norm1 = nn.LayerNorm(n_emb, eps=1e-6, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(n_emb, n_head, dropout=p_drop_attn, batch_first=True)
        self.attn_modulate = _ShiftScaleMod(n_emb)
        self.attn_gate = _ZeroScaleMod(n_emb)

        # --- Cross Attention Branch (New!) ---
        self.norm_cross = nn.LayerNorm(n_emb, eps=1e-6, elementwise_affine=False)
        self.cross_attn = nn.MultiheadAttention(n_emb, n_head, dropout=p_drop_attn, batch_first=True)
        self.cross_modulate = _ShiftScaleMod(n_emb) # Time 注入 Cross Attention
        self.cross_gate = _ZeroScaleMod(n_emb)

        # --- MLP Branch ---
        self.norm2 = nn.LayerNorm(n_emb, eps=1e-6, elementwise_affine=False)
        self.linear1 = nn.Linear(n_emb, 4 * n_emb)
        self.act = nn.GELU()
        self.linear2 = nn.Linear(4 * n_emb, n_emb)
        self.mlp_modulate = _ShiftScaleMod(n_emb)
        self.mlp_gate = _ZeroScaleMod(n_emb)

    def forward(self, x, c, memory, memory_mask=None):
        # x: (B, T, D) -> Noisy Actions
        # c: (B, D)    -> Time Embedding
        # memory: (B, T_cond, D) -> Condition Memory

        # 1. Self-Attention (处理 x 内部关系)
        x_res = self.attn_modulate(self.norm1(x), c)
        # 可以在这里加 causal mask，如果是自回归生成
        attn_out, _ = self.attn(x_res, x_res, x_res)
        x = x + self.attn_gate(attn_out, c)

        # 2. Cross-Attention (处理 x 对 condition 的关注)
        if memory is not None:
            x_res = self.cross_modulate(self.norm_cross(x), c)
            # Query = x_res (Time-Modulated), Key/Value = memory
            cross_out, _ = self.cross_attn(
                query=x_res, 
                key=memory, 
                value=memory, 
                key_padding_mask=memory_mask
            )
            x = x + self.cross_gate(cross_out, c)

        # 3. MLP
        x_res = self.mlp_modulate(self.norm2(x), c)
        mlp_out = self.linear2(self.act(self.linear1(x_res)))
        x = x + self.mlp_gate(mlp_out, c)
        
        return x

    def reset_parameters(self):
        # 初始化所有 Gate 为 0
        self.attn_gate.reset_parameters()
        self.cross_gate.reset_parameters()
        self.mlp_gate.reset_parameters()
        # Modulate 初始化
        self.attn_modulate.reset_parameters()
        self.cross_modulate.reset_parameters()
        self.mlp_modulate.reset_parameters()


# ==========================================
# 4. 主模型: Hybrid DiT
# ==========================================

class HybridDiT(nn.Module):
    def __init__(self,
            input_dim: int,
            output_dim: int,
            horizon: int,
            cond_dim: int,           # 序列条件的每一帧维度
            cond_seq_len: int = 10,  # 序列条件长度 (例如 n_obs_steps)
            n_layer: int = 12,
            n_head: int = 12,
            n_emb: int = 768,
            time_dim: int = 256,
            p_drop_emb: float = 0.1,
            p_drop_attn: float = 0.1,
            n_cond_layers: int = 2   # Condition Encoder 的层数
        ):
        super().__init__()

        self.n_emb = n_emb
        self.horizon = horizon
        
        # 1. 主干输入 Embedder
        self.input_emb = nn.Linear(input_dim, n_emb)
        self.pos_emb = nn.Parameter(torch.zeros(1, horizon, n_emb))
        nn.init.xavier_uniform_(self.pos_emb)
        self.drop = nn.Dropout(p_drop_emb)

        # 2. Time Embedder (生成 c)
        self.time_emb = _TimeNetwork(time_dim, n_emb)

        # 3. Condition Encoder (生成 memory)
        # 如果 n_cond_layers > 0，则使用 Transformer Encoder
        if n_cond_layers > 0:
            self.cond_encoder = ConditionEncoder(
                cond_dim=cond_dim, 
                n_emb=n_emb, 
                n_layer=n_cond_layers, 
                n_head=n_head,
                max_seq_len=cond_seq_len
            )
        else:
            # 简单的线性投影，如果不想要深层 Encoder
            self.cond_encoder = nn.Sequential(
                nn.Linear(cond_dim, n_emb),
                nn.SiLU(),
                nn.Linear(n_emb, n_emb)
            )

        # 4. Decoder Blocks (Main Trunk)
        self.blocks = nn.ModuleList([
            DiTDecoderBlock(n_emb, n_head, p_drop_attn) for _ in range(n_layer)
        ])
        
        # 5. Output Layer
        self.final_layer = FinalLayer(n_emb, output_dim)

        self.apply(self._init_weights)
        
        # Zero-init gates specifically
        for block in self.blocks:
            block.reset_parameters()

        logger.info(f"HybridDiT Params: {sum(p.numel() for p in self.parameters())/1e6:.2f}M")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            if module.elementwise_affine:
                nn.init.zeros_(module.bias)
                nn.init.ones_(module.weight)

    def forward(self, 
        sample: torch.Tensor, 
        timestep: torch.Tensor, 
        cond: torch.Tensor, 
        cond_mask: Optional[torch.Tensor] = None
    ):
        """
        sample: (B, T, input_dim) -> Noisy Action Sequence
        timestep: (B,)
        cond: (B, T_cond, cond_dim) -> History Observations
        """
        # 1. Process Time -> Global Context 'c'
        c = self.time_emb(timestep) # (B, n_emb)

        # 2. Process Condition -> Sequence Memory 'memory'
        # cond_encoder 自动处理投影和位置编码
        memory = self.cond_encoder(cond) # (B, T_cond, n_emb)

        # 3. Process Input
        x = self.input_emb(sample) + self.pos_emb
        x = self.drop(x)
        
        # 4. Pass through Hybrid Blocks
        for block in self.blocks:
            x = block(x, c, memory, memory_mask=cond_mask)
            
        # 5. Final Output
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
        # Ensure cond_encoder position embeddings are not decayed
        if hasattr(self.cond_encoder, 'pos_emb'):
             no_decay.add("cond_encoder.pos_emb")

        param_dict = {pn: p for pn, p in self.named_parameters()}
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        return optim_groups