from typing import Dict, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.transformer_for_diffusion import TransformerForDiffusion
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
            noise_scheduler: DDPMScheduler,
            # task params
            horizon, 
            n_action_steps, 
            n_obs_steps,
            num_inference_steps=None,
            # image
            crop_shape=(76, 76),
            obs_encoder_group_norm=False,
            eval_fixed_crop=False,
            # arch
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
            # parameters passed to step
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

        # get raw robomimic config
        config = get_robomimic_config(
            algo_name='bc_rnn',
            hdf5_type='image',
            task_name='square',
            dataset_type='ph')
        
        with config.unlocked():
            # set config with shape_meta
            config.observation.modalities.obs = obs_config

            if crop_shape is None:
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality['obs_randomizer_class'] = None
            else:
                # set random crop parameter
                ch, cw = crop_shape
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality.obs_randomizer_kwargs.crop_height = ch
                        modality.obs_randomizer_kwargs.crop_width = cw

        # init global state
        ObsUtils.initialize_obs_utils_with_config(config)

        # load model
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

        # create diffusion model
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
        # Keep noise_scheduler for config compatibility, though we use custom ODE logic
        self.noise_scheduler = noise_scheduler
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

        # Default to 10 steps for ODE if not specified (ODE is usually faster)
        if num_inference_steps is None:
            num_inference_steps = 10 
        self.num_inference_steps = num_inference_steps

    # ========= inference (ODE Euler) ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            cond=None, generator=None,
            **kwargs
            ):
        """
        Implements Euler Integration for Rectified Flow.
        Path: X_t = t * X_1 + (1 - t) * X_0
        ODE: dX_t = (X_1 - X_0) dt = v(X_t, t) dt
        """
        model = self.model
        B = condition_data.shape[0]
        device = condition_data.device
        dtype = condition_data.dtype

        # 1. Sample Initial Noise (X_0)
        # In Rectified Flow, we start from noise at t=0
        x_0 = torch.randn(
            size=condition_data.shape, 
            dtype=dtype,
            device=device,
            generator=generator)
        
        trajectory = x_0.clone()

        # 2. ODE Solver (Euler Method)
        # Integrate from t=0 to t=1
        steps = self.num_inference_steps
        dt = 1.0 / steps
        
        # Grid for time: [0, dt, 2*dt, ..., 1.0]
        # Note: Depending on implementation, one might go 0 -> 1 or 1 -> 0.
        # Rectified Flow (Liu et al.) typically defines flow from Noise(0) to Data(1).
        
        for i in range(steps):
            # Current time t
            t_val = i / steps
            # Create time tensor for batch
            t = torch.full((B,), t_val, device=device, dtype=dtype)
            
            # --- Inpainting / Conditioning Logic ---
            # If we have fixed conditions (like observations in the sequence),
            # we must enforce them to follow the straight line trajectory explicitly.
            # Clean data condition is `condition_data` (X_1_known).
            # Initial noise condition is `x_0` (X_0_known).
            # Expected value at t: X_t_known = t * X_1_known + (1-t) * X_0_known
            if condition_mask.any():
                x_t_known = t_val * condition_data + (1 - t_val) * x_0
                trajectory[condition_mask] = x_t_known[condition_mask]

            # --- Model Prediction ---
            # Predict velocity v = X_1 - X_0
            # Note: TransformerForDiffusion expects t to be handled appropriately.
            # If it uses sinusoidal embedding, feeding float 0-1 works but scale matters.
            # Often it's safer to pass t directly.
            velocity_pred = model(trajectory, t, cond)

            # --- Euler Step ---
            # X_{t+1} = X_t + v * dt
            trajectory = trajectory + velocity_pred * dt

        # Final step enforcement (at t=1, trajectory should exactly match condition_data where masked)
        trajectory[condition_mask] = condition_data[condition_mask]        

        return trajectory

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        assert 'past_action' not in obs_dict # not implemented yet
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        cond = None
        cond_data = None
        cond_mask = None
        if self.obs_as_cond:
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, To, Do
            cond = nobs_features.reshape(B, To, -1)
            shape = (B, T, Da)
            if self.pred_action_steps_only:
                shape = (B, self.n_action_steps, Da)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, To, Do
            nobs_features = nobs_features.reshape(B, To, -1)
            shape = (B, T, Da+Do)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs_features
            cond_mask[:,:To,Da:] = True

        # run sampling
        nsample = self.conditional_sample(
            cond_data, 
            cond_mask,
            cond=cond,
            **self.kwargs)
        
        # unnormalize prediction
        naction_pred = nsample[...,:Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        if self.pred_action_steps_only:
            action = action_pred
        else:
            start = To - 1
            end = start + self.n_action_steps
            action = action_pred[:,start:end]
        
        result = {
            'action': action,
            'action_pred': action_pred
        }
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def get_optimizer(
            self, 
            transformer_weight_decay: float, 
            obs_encoder_weight_decay: float,
            learning_rate: float, 
            betas: Tuple[float, float]
        ) -> torch.optim.Optimizer:
        optim_groups = self.model.get_optim_groups(
            weight_decay=transformer_weight_decay)
        optim_groups.append({
            "params": self.obs_encoder.parameters(),
            "weight_decay": obs_encoder_weight_decay
        })
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas
        )
        return optimizer

    def compute_loss(self, batch):
        # normalize input
        assert 'valid_mask' not in batch
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]
        To = self.n_obs_steps

        # handle different ways of passing observation
        cond = None
        trajectory = nactions
        if self.obs_as_cond:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, 
                lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            cond = nobs_features.reshape(batch_size, To, -1)
            if self.pred_action_steps_only:
                start = To - 1
                end = start + self.n_action_steps
                trajectory = nactions[:,start:end]
        else:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            trajectory = torch.cat([nactions, nobs_features], dim=-1).detach()

        # generate impainting mask
        if self.pred_action_steps_only:
            condition_mask = torch.zeros_like(trajectory, dtype=torch.bool)
        else:
            condition_mask = self.mask_generator(trajectory.shape)

        # --- Rectified Flow / Velocity Matching Loss ---
        
        # 1. Prepare Data (X_1)
        # trajectory is X_1 (Data)
        x_1 = trajectory

        # 2. Sample Noise (X_0)
        x_0 = torch.randn(x_1.shape, device=x_1.device, dtype=x_1.dtype)
        
        # 3. Sample Time t ~ Uniform[0, 1]
        t = torch.rand((batch_size,), device=x_1.device, dtype=x_1.dtype)
        
        # 4. Interpolate X_t = t * X_1 + (1 - t) * X_0
        # Broadcast t to match x_1 shape [B, T, D]
        t_expand = t.reshape(-1, 1, 1)
        x_t = t_expand * x_1 + (1 - t_expand) * x_0
        
        # 5. Calculate Velocity Target v = X_1 - X_0
        # The flow ODE is dX_t/dt = X_1 - X_0
        target_v = x_1 - x_0

        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning to input
        # Note: In Flow Matching, conditioning on X_1 part is implicit if we want to fix it.
        # But for training the vector field, we usually just train on the full trajectory.
        # However, if we want to support inpainting during inference, we should probably
        # mask the loss or input.
        # Here we follow the original logic: overwrite the conditioned part?
        # For training a global field, we usually don't overwrite X_t with clean data 
        # unless we are doing conditional training. 
        # But `condition_mask` usually implies we KNOW these values.
        # In original DDPM code: `noisy_trajectory[condition_mask] = trajectory[condition_mask]`
        # This makes the model see CLEAN data at the conditioned spots.
        # We can do the same for Flow Matching:
        x_t[condition_mask] = x_1[condition_mask]
        
        # Predict the velocity
        # Pass t directly (model should handle float 0-1)
        pred_v = self.model(x_t, t, cond)

        # MSE Loss on Velocity
        loss = F.mse_loss(pred_v, target_v, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        return loss