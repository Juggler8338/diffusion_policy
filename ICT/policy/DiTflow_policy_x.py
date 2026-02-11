# DiT_ode_policy.py
# 修改后版本：条件处理向 LeRobot 版对齐
# - global_cond = all n_obs_steps 的 obs features 拼接后压平 (B, n_obs_steps * feature_dim)
# - 移除 mean pool 和拼接 obs 到 action token 的模式
# - 训练时支持 beta 分布采样 t（来自 Pi0）
# - 推理时支持 clip_sample
# - 保留 robomimic 视觉 encoder 和 diffusion_policy 框架结构
# - 移除 conditional mask（LeRobot 版无此功能）

from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from ICT.model.DiTModel_pro import TransformerForDiffusion 
from diffusion_policy.common.pytorch_util import dict_apply, replace_submodules
import diffusion_policy.model.vision.crop_randomizer as dmvc
import robomimic.utils.obs_utils as ObsUtils
import robomimic.models.base_nets as rmbn
from robomimic.algo import algo_factory
from robomimic.algo.algo import PolicyAlgo
from diffusion_policy.common.robomimic_config_util import get_robomimic_config

class DiffusionTransformerODEPolicy(BaseImagePolicy):
    def __init__(self, 
            shape_meta: dict,
            horizon, 
            n_action_steps, 
            n_obs_steps,
            num_inference_steps=None,
            crop_shape=(76, 76),
            obs_encoder_group_norm=False,
            eval_fixed_crop=False,
            n_layer=8,
            n_head=4,
            n_emb=256,
            p_drop_emb=0.0,
            p_drop_attn=0.3,
            time_dim=256,
            clip_sample=False,
            clip_sample_range=1.0,
            training_noise_sampling="uniform",  # "uniform" or "beta"
            **kwargs):
        super().__init__()
        
        # parse shape_meta (同原版)
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
        
        # LeRobot 风格：global_cond 是 n_obs_steps * obs_feature_dim
        cond_dim = obs_feature_dim * n_obs_steps

        model = TransformerForDiffusion(
            input_dim=action_dim,
            output_dim=action_dim,
            horizon=horizon,
            cond_dim=cond_dim,
            n_layer=n_layer,
            n_head=n_head,
            n_emb=n_emb,
            p_drop_emb=p_drop_emb,
            p_drop_attn=p_drop_attn,
            time_dim=time_dim,
        )

        self.obs_encoder = obs_encoder
        self.model = model
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
        # uniform 在 compute_loss 中直接采样
        
        if num_inference_steps is None:
            num_inference_steps = 100  # LeRobot 默认 100
        self.num_inference_steps = num_inference_steps

    # ========= inference ============
    @torch.no_grad()
    def conditional_sample(self, cond, **kwargs):
        B, device, dtype = cond.shape[0], cond.device, cond.dtype
        trajectory = torch.randn(B, self.horizon, self.action_dim, device=device, dtype=dtype)
        dt = 1.0 / self.num_inference_steps
        
        t = 0.0
    
        for k in range(self.num_inference_steps):
            t = torch.full((B,), k / self.num_inference_steps, device=device, dtype=dtype)
            pred_target = self.model(trajectory, t, cond=cond)
            denom = (1 - t)[:, None, None]
            v = (pred_target - trajectory) / denom
            trajectory = trajectory + v * dt
            if self.clip_sample:
                trajectory = torch.clamp(trajectory, -self.clip_sample_range, self.clip_sample_range)
                
        return trajectory

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        nobs = self.normalizer.normalize(obs_dict)
        B = next(iter(nobs.values())).shape[0]
        
        # 提取 n_obs_steps 的观测
        this_nobs = dict_apply(nobs, lambda x: x[:, :self.n_obs_steps].reshape(-1, *x.shape[2:]))
        obs_features = self.obs_encoder(this_nobs)  # (B*To, feat_dim)
        obs_features = obs_features.reshape(B, self.n_obs_steps, -1)
        
        # LeRobot 风格：压平所有时间步的特征
        cond = obs_features.reshape(B, -1)  # (B, n_obs_steps * feat_dim)
        
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
        
        # 提取 n_obs_steps 的观测
        this_nobs = dict_apply(nobs, lambda x: x[:, :self.n_obs_steps].reshape(-1, *x.shape[2:]))
        obs_features = self.obs_encoder(this_nobs)
        obs_features = obs_features.reshape(B, self.n_obs_steps, -1)
        cond = obs_features.reshape(B, -1)  # (B, n_obs_steps * feat_dim)
        
        trajectory = nactions  # (B, horizon, Da)
        
        # 采样噪声和 t
        x0 = torch.randn_like(trajectory)
        if self.training_noise_sampling == "beta":
            t = self.noise_dist.sample((B,)).to(trajectory.device)
        else:
            t = torch.rand(B, device=trajectory.device)
            
        t_expand = t[:, None, None]
        x_t = t_expand * trajectory + (1 - t_expand) * x0
        
        pred_target = self.model(x_t, t, cond)
        loss = F.mse_loss(pred_target, trajectory, reduction='mean')
        return loss

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def get_optimizer(self, 
                      transformer_weight_decay: float, 
                      obs_encoder_weight_decay: float,
                      learning_rate: float, 
                      betas: Tuple[float, float]):
        optim_groups = self.model.get_optim_groups(weight_decay=transformer_weight_decay)
        optim_groups.append({
            "params": self.obs_encoder.parameters(),
            "weight_decay": obs_encoder_weight_decay
        })
        return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)