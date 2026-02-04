# DiT_ode_policy_pro.py
from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from ICT.model.DiTModel_x import TransformerForDiffusion
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.common.robomimic_config_util import get_robomimic_config
from robomimic.algo import algo_factory
from robomimic.algo.algo import PolicyAlgo
import robomimic.utils.obs_utils as ObsUtils
import robomimic.models.base_nets as rmbn
import diffusion_policy.model.vision.crop_randomizer as dmvc
from diffusion_policy.common.pytorch_util import dict_apply, replace_submodules


class DiffusionTransformerODEPolicy(BaseImagePolicy):
    def __init__(self, 
            shape_meta: dict,
            # noise_scheduler 已移除（Flow Matching 不需要）
            horizon, 
            n_action_steps, 
            n_obs_steps,
            num_inference_steps=None,
            crop_shape=(76, 76),
            obs_encoder_group_norm=False,
            eval_fixed_crop=False,
            n_layer=8,
            n_cond_layers=0,
            n_head=4,
            n_emb=256,
            p_drop_emb=0.0,
            p_drop_attn=0.3,
            causal_attn=True,
            time_as_cond=True,
            obs_as_cond=True,
            pred_action_steps_only=False,
            **kwargs):
        super().__init__()

        # parse shape_meta
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
                func=lambda x: nn.GroupNorm(
                    num_groups=x.num_features//16, 
                    num_channels=x.num_features)
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
        input_dim = action_dim if obs_as_cond else (obs_feature_dim + action_dim)
        output_dim = input_dim
        cond_dim = obs_feature_dim if obs_as_cond else 0

        model = TransformerForDiffusion(
            input_dim=input_dim,
            output_dim=output_dim,
            horizon=horizon,
            n_obs_steps=n_obs_steps,
            cond_dim=cond_dim,
            n_layer=n_layer,
            n_head=n_head,
            n_emb=n_emb,
            p_drop_emb=p_drop_emb,
            p_drop_attn=p_drop_attn,
            causal_attn=causal_attn,
            time_as_cond=time_as_cond,
            obs_as_cond=obs_as_cond,
            n_cond_layers=n_cond_layers
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if (obs_as_cond) else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_cond = obs_as_cond
        self.pred_action_steps_only = pred_action_steps_only
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = 30  # Heun 下 30 步足够高质量（直线路径下可进一步减少）
        self.num_inference_steps = num_inference_steps
    # Euler 一阶采样
    def conditional_sample(self, 
            condition_data, condition_mask,
            cond=None,
            **kwargs):
        model = self.model
        B = condition_data.shape[0]
        device = condition_data.device
        dtype = condition_data.dtype

        # 初始噪声（t=0 时 x_t = x_0）
        x_0 = torch.randn(size=condition_data.shape, dtype=dtype, device=device)
        trajectory = x_0.clone()
        dt = 1.0 / self.num_inference_steps
        t = 0.0
        eps = 1e-5  # 防止 t→1 时除零

        for _ in range(self.num_inference_steps):
            t_tensor = torch.full((B,), t, device=device, dtype=dtype)

            # 强制已知部分跟随理论直线路径
            if condition_mask.any():
                x_t_known = t * condition_data + (1 - t) * x_0
                trajectory[condition_mask] = x_t_known[condition_mask]

            # 模型直接预测目标 x_1
            pred_target = model(trajectory, t_tensor, cond)

            # 【推荐】Target anchoring：强制 conditioned 部分直接输出真实目标值
            if condition_mask.any():
                pred_target = pred_target * (~condition_mask) + condition_data * condition_mask

            # 计算 velocity：v = (pred_x_1 - x_t) / (1 - t)
            v = (pred_target - trajectory) / (1 - t + eps)

            # Euler 一阶更新
            trajectory = trajectory + v * dt
            t = t + dt

        # 最终强制 conditioned 部分为真实值
        if condition_mask.any():
            trajectory[condition_mask] = condition_data[condition_mask]

        return trajectory

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        assert 'past_action' not in obs_dict
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        device = self.device
        dtype = self.dtype

        cond = None
        cond_data = None
        cond_mask = None
        if self.obs_as_cond:
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            cond = nobs_features.reshape(B, To, -1)
            shape = (B, self.horizon, Da)
            if self.pred_action_steps_only:
                shape = (B, self.n_action_steps, Da)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            nobs_features = nobs_features.reshape(B, To, -1)
            shape = (B, self.horizon, Da+Do)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs_features
            cond_mask[:,:To,Da:] = True

        nsample = self.conditional_sample(cond_data, cond_mask, cond=cond, **self.kwargs)
        
        naction_pred = nsample[...,:Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        if self.pred_action_steps_only:
            action = action_pred
        else:
            start = To - 1
            end = start + self.n_action_steps
            action = action_pred[:,start:end]
        
        return {
            'action': action,
            'action_pred': action_pred
        }

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def get_optimizer(self, transformer_weight_decay: float, obs_encoder_weight_decay: float,
                      learning_rate: float, betas: Tuple[float, float]):
        optim_groups = self.model.get_optim_groups(weight_decay=transformer_weight_decay)
        optim_groups.append({
            "params": self.obs_encoder.parameters(),
            "weight_decay": obs_encoder_weight_decay
        })
        return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)

    def compute_loss(self, batch):
        assert 'valid_mask' not in batch
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]
        To = self.n_obs_steps

        cond = None
        trajectory = nactions  # x_1（目标）
        if self.obs_as_cond:
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            cond = nobs_features.reshape(batch_size, To, -1)
            if self.pred_action_steps_only:
                start = To - 1
                end = start + self.n_action_steps
                trajectory = nactions[:,start:end]
        else:
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            trajectory = torch.cat([nactions, nobs_features], dim=-1).detach()

        if self.pred_action_steps_only:
            condition_mask = torch.zeros_like(trajectory, dtype=torch.bool)
        else:
            condition_mask = self.mask_generator(trajectory.shape)

        x_1 = trajectory
        x_0 = torch.randn(x_1.shape, device=x_1.device, dtype=x_1.dtype)
        t = torch.rand((batch_size,), device=x_1.device, dtype=x_1.dtype)
        t_expand = t.reshape(-1, 1, 1)
        x_t = t_expand * x_1 + (1 - t_expand) * x_0

        # 训练时强制 conditioned 部分为真实目标值（与原代码一致）
        x_t[condition_mask] = x_1[condition_mask]

        # 模型直接预测目标 x_1
        pred_target = self.model(x_t, t, cond)

        loss_mask = ~condition_mask
        loss = F.mse_loss(pred_target, x_1, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        return loss