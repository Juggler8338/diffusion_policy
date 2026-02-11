# HybridDiT_policy.py
from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from ICT.model.HybridDiT import HybridDiT 
from diffusion_policy.common.pytorch_util import dict_apply, replace_submodules
import diffusion_policy.model.vision.crop_randomizer as dmvc
import robomimic.utils.obs_utils as ObsUtils
import robomimic.models.base_nets as rmbn
from robomimic.algo import algo_factory
from robomimic.algo.algo import PolicyAlgo
from diffusion_policy.common.robomimic_config_util import get_robomimic_config

class HybridDiTPolicy(BaseImagePolicy):
    def __init__(self, 
            shape_meta: dict,
            horizon, 
            n_action_steps, 
            n_obs_steps,
            num_inference_steps=None,
            crop_shape=(76, 76),
            obs_encoder_group_norm=False,
            eval_fixed_crop=False,
            # DiT 参数
            n_layer=12,
            n_head=12,
            n_emb=768,
            p_drop_emb=0.1,
            p_drop_attn=0.1,
            time_dim=256,
            n_cond_layers=2, # 新增：Condition Encoder 的层数
            # 推理/训练参数
            clip_sample=False,
            clip_sample_range=1.0,
            training_noise_sampling="uniform", 
            **kwargs):
        super().__init__()
        
        # 1. 解析 shape_meta 和配置 Robomimic Encoder (保持不变)
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        obs_shape_meta = shape_meta['obs']
        obs_config = {
            'low_dim': [],
            'rgb': [],
            'depth': [],
            'scan': []
        }
        obs_key_shapes = dict()
        for key, attr in obs_shape_meta.items():
            shape = attr['shape']
            obs_key_shapes[key] = list(shape)
            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                obs_config['rgb'].append(key)
            elif type == 'low_dim':
                obs_config['low_dim'].append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")

        config = get_robomimic_config(
            algo_name='bc_rnn',
            hdf5_type='image',
            task_name='square',
            dataset_type='ph')
        
        with config.unlocked():
            config.observation.modalities.obs = obs_config
            if crop_shape is None:
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality['obs_randomizer_class'] = None
            else:
                ch, cw = crop_shape
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality.obs_randomizer_kwargs.crop_height = ch
                        modality.obs_randomizer_kwargs.crop_width = cw

        ObsUtils.initialize_obs_utils_with_config(config)

        policy: PolicyAlgo = algo_factory(
                algo_name=config.algo_name,
                config=config,
                obs_key_shapes=obs_key_shapes,
                ac_dim=action_dim,
                device='cpu',
            )

        obs_encoder = policy.nets['policy'].nets['encoder'].nets['obs']
        
        if obs_encoder_group_norm:
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(num_groups=x.num_features//16, num_channels=x.num_features)
            )

        if eval_fixed_crop:
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, rmbn.CropRandomizer),
                func=lambda x: dmvc.CropRandomizer(
                    input_shape=x.input_shape,
                    crop_height=x.crop_height,
                    crop_width=x.crop_width,
                    num_crops=x.num_crops,
                    pos_enc=x.pos_enc
                )
            )

        obs_feature_dim = obs_encoder.output_shape()[0]
        
        # =========================================================
        # 2. 核心修改区域：参数定义
        # =========================================================
        
        # [修改点 1] cond_dim 不再乘以 n_obs_steps
        # HybridDiT 接收序列 (B, T, D)，所以这里只传 D
        input_cond_dim = obs_feature_dim 

        self.model = HybridDiT(
            input_dim=action_dim,
            output_dim=action_dim,
            horizon=horizon,
            cond_dim=input_cond_dim,  # 单帧特征维度
            cond_seq_len=n_obs_steps, # 序列长度
            n_layer=n_layer,
            n_head=n_head,
            n_emb=n_emb,
            time_dim=time_dim,
            p_drop_emb=p_drop_emb,
            p_drop_attn=p_drop_attn,
            n_cond_layers=n_cond_layers # 新增参数
        )

        self.obs_encoder = obs_encoder
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_feature_dim = obs_feature_dim
        
        self.clip_sample = clip_sample
        self.clip_sample_range = clip_sample_range
        
        self.training_noise_sampling = training_noise_sampling
        if training_noise_sampling == "beta":
            import torch.distributions as D
            s = 0.999
            beta_dist = D.Beta(1.5, 1.0)
            affine = D.transforms.AffineTransform(loc=s, scale=-s)
            self.noise_dist = D.TransformedDistribution(beta_dist, [affine])
        
        if num_inference_steps is None:
            num_inference_steps = 100
        self.num_inference_steps = num_inference_steps

    # ========= inference ============
    @torch.no_grad()
    def conditional_sample(self, cond, **kwargs):
        # cond shape: (B, n_obs_steps, obs_feature_dim)  <-- 注意这里是 3D
        B, device, dtype = cond.shape[0], cond.device, cond.dtype
        
        # 初始化噪声 (t=0, noise) -> (t=1, data) 的 Flow Matching 定义
        # 这里的 trajectory 是 x0 (noise)
        trajectory = torch.randn(B, self.horizon, self.action_dim, device=device, dtype=dtype)
        
        dt = 1.0 / self.num_inference_steps
        
        for k in range(self.num_inference_steps):
            # t 从 0 到 1
            t_val = k / self.num_inference_steps
            t = torch.full((B,), t_val, device=device, dtype=dtype)
            
            # [修改点 2] 调用 forward，传入序列 cond
            v = self.model(sample=trajectory, timestep=t, cond=cond)
            
            # Euler Integration: x_{t+1} = x_t + v * dt
            trajectory = trajectory + v * dt
            
            if self.clip_sample:
                trajectory = torch.clamp(trajectory, -self.clip_sample_range, self.clip_sample_range)
                
        return trajectory

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        nobs = self.normalizer.normalize(obs_dict)
        # 获取 Batch Size
        B = next(iter(nobs.values())).shape[0]
        
        # 提取 n_obs_steps 的观测并展平 batch 维度以通过 encoder
        # Input: (B, n_obs_steps, C, H, W) -> (B * n_obs_steps, C, H, W)
        this_nobs = dict_apply(nobs, lambda x: x[:, :self.n_obs_steps].reshape(-1, *x.shape[2:]))
        
        obs_features = self.obs_encoder(this_nobs) # (B * n_obs_steps, feat_dim)
        
        # [修改点 3] 保持序列结构，不要 Flatten 成 2D
        cond = obs_features.reshape(B, self.n_obs_steps, -1) 
        # cond shape: (B, n_obs_steps, obs_feature_dim)
        
        naction_pred = self.conditional_sample(cond)
        
        action_pred = self.normalizer['action'].unnormalize(naction_pred)
        
        start = self.n_obs_steps - 1
        end = start + self.n_action_steps
        action = action_pred[:, start:end]
        
        return {
            'action': action,
            'action_pred': action_pred
        }

    # ========= training ============
    def compute_loss(self, batch):
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        B = nactions.shape[0]
        
        # 1. 处理观测特征
        this_nobs = dict_apply(nobs, lambda x: x[:, :self.n_obs_steps].reshape(-1, *x.shape[2:]))
        obs_features = self.obs_encoder(this_nobs)
        
        # [修改点 4] 训练时同样保持序列结构
        cond = obs_features.reshape(B, self.n_obs_steps, -1) 
        # (B, n_obs_steps, obs_feature_dim)
        
        # 2. Flow Matching Loss Construction
        x1 = nactions  # Target Data (Horizon, Action_dim)
        x0 = torch.randn_like(x1) # Noise
        
        # 采样 t
        if self.training_noise_sampling == "beta":
            t = self.noise_dist.sample((B,)).to(x1.device)
        else:
            t = torch.rand(B, device=x1.device)
            
        t_expand = t[:, None, None]
        
        # Rectified Flow 插值: x_t = t * x1 + (1 - t) * x0
        # 当 t=0, x_t = x0 (noise)
        # 当 t=1, x_t = x1 (data)
        x_t = t_expand * x1 + (1 - t_expand) * x0
        
        # Vector Field Target: v = d/dt (x_t) = x1 - x0
        target_v = x1 - x0
        
        # 3. 模型前向传播
        # 注意：这里传入的 cond 是 3D 张量
        pred_v = self.model(sample=x_t, timestep=t, cond=cond)
        
        loss = F.mse_loss(pred_v, target_v, reduction='mean')
        return loss

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def get_optimizer(self, 
                      transformer_weight_decay: float, 
                      obs_encoder_weight_decay: float,
                      learning_rate: float, 
                      betas: Tuple[float, float]):
        # 注意：HybridDiT 也有 get_optim_groups 方法
        optim_groups = self.model.get_optim_groups(weight_decay=transformer_weight_decay)
        optim_groups.append({
            "params": self.obs_encoder.parameters(),
            "weight_decay": obs_encoder_weight_decay
        })
        return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)