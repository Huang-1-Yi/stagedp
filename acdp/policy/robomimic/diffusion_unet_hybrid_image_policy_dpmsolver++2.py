from typing import Dict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.common.robomimic_config_util import get_robomimic_config
from robomimic.algo import algo_factory
from robomimic.algo.algo import PolicyAlgo
import robomimic.utils.obs_utils as ObsUtils
try:
    import robomimic.models.base_nets as rmbn
    if not hasattr(rmbn, 'CropRandomizer'):
        raise ImportError("CropRandomizer is not in robomimic.models.base_nets")
except ImportError:
    import robomimic.models.obs_core as rmbn
import diffusion_policy.model.vision.crop_randomizer as dmvc
from diffusion_policy.common.pytorch_util import dict_apply, replace_submodules
from diffusion_policy.model.vision.rot_randomizer import RotRandomizer


class DiffusionUnetHybridImagePolicy(BaseImagePolicy):
    def __init__(self, 
            shape_meta: dict,
            noise_scheduler: DPMSolverMultistepScheduler,
            horizon, 
            n_action_steps, 
            n_obs_steps,
            num_inference_steps=None,
            obs_as_global_cond=True,
            crop_shape=(76, 76),
            diffusion_step_embed_dim=256,
            down_dims=(256,512,1024),
            kernel_size=5,
            n_groups=8,
            cond_predict_scale=True,
            obs_encoder_group_norm=False,
            eval_fixed_crop=False,
            rot_aug=False,
            # parameters passed to step
            **kwargs):
        super().__init__()

        # 移除全局种子设置，改为在需要时局部设置

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
            # replace batch norm with group norm
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
        input_dim = action_dim + obs_feature_dim 
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = action_dim
            global_cond_dim = obs_feature_dim * n_obs_steps

        model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale
        )

        self.obs_encoder = obs_encoder
        self.model = model

        # 添加梯度裁剪和数值稳定层
        self.grad_clip_value = 1.0
        self.feature_stabilizer = nn.Sequential(
            nn.LayerNorm(obs_feature_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        self.normalizer = LinearNormalizer()
        self.rot_randomizer = RotRandomizer()

        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.rot_aug = rot_aug
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

        print("Diffusion params: %e" % sum(p.numel() for p in self.model.parameters()))
        print("Vision params: %e" % sum(p.numel() for p in self.obs_encoder.parameters()))
    
    def _stabilize_gradients(self):
        """梯度裁剪和数值稳定"""
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_value)
        torch.nn.utils.clip_grad_value_(self.model.parameters(), 1.0)
    
    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            local_cond=None, global_cond=None,
            generator=None,
            **kwargs
            ):
        model = self.model
        scheduler = self.noise_scheduler

        # 初始化轨迹
        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)
    
        # 设置时间步
        scheduler.set_timesteps(self.num_inference_steps)

        # DPM-Solver 需要管理历史输出
        model_outputs = []
        
        for i, t in enumerate(scheduler.timesteps):
            # 1. 应用条件
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. 预测模型输出
            model_output = model(trajectory, t, 
                local_cond=local_cond, global_cond=global_cond)

            # 3. 数值稳定性检查
            if not torch.isfinite(model_output).all():
                # 使用更稳定的数值处理方法
                model_output = torch.clamp(model_output, -1e4, 1e4)
                model_output = torch.nan_to_num(model_output, nan=0.0, posinf=1e4, neginf=-1e4)

            # 4. 管理历史输出 (DPM-Solver 特性)
            model_outputs.append(model_output)
            if len(model_outputs) > getattr(scheduler.config, 'solver_order', 2):
                model_outputs.pop(0)

            # 5. 计算前一步样本
            step_kwargs = {
                'generator': generator,
                'return_dict': True
            }
            step_kwargs.update(kwargs)
            
            # 对于 DPM-Solver，需要传递历史输出
            if hasattr(scheduler, 'algorithm_type') and len(model_outputs) > 1:
                step_kwargs['model_outputs'] = model_outputs[:-1]  # 传递之前的历史输出
                
            step_result = scheduler.step(
                model_output, t, trajectory, **step_kwargs)
            trajectory = step_result.prev_sample

            # 6. 轨迹数值稳定性检查
            if not torch.isfinite(trajectory).all():
                trajectory = torch.clamp(trajectory, -1e4, 1e4)
                trajectory = torch.nan_to_num(trajectory, nan=0.0, posinf=1e4, neginf=-1e4)

        # 最后确保条件被正确应用
        trajectory[condition_mask] = condition_data[condition_mask]        
        
        return trajectory

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        assert 'past_action' not in obs_dict
        # 归一化输入
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # 构建输入
        device = self.device
        dtype = self.dtype
        
        # 只在需要时创建生成器，而不是每次都固定种子
        if hasattr(self, 'use_deterministic_sampling') and self.use_deterministic_sampling:
            generator = torch.Generator(device=device).manual_seed(42)
        else:
            generator = None

        # 处理观察数据
        local_cond = None
        global_cond = None

        # 提取观察特征
        this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
        nobs_features = self.obs_encoder(this_nobs)

        # 特征稳定性处理
        nobs_features = self.feature_stabilizer(nobs_features)
        
        if self.obs_as_global_cond:
            # 重塑为 B, Do
            global_cond = nobs_features.reshape(B, -1)
            # 动作的空数据
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # 重塑为 B, To, Do
            nobs_features = nobs_features.reshape(B, To, -1)
            cond_data = torch.zeros(size=(B, T, Da+Do), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs_features
            cond_mask[:,:To,Da:] = True

        # 运行采样
        nsample = self.conditional_sample(
            cond_data, 
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            generator=generator,
            **self.kwargs)
        
        # 反归一化预测
        naction_pred = nsample[...,:Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # 获取动作
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:,start:end]
        
        # 最终检查动作是否有效
        if not torch.isfinite(action).all():
            # 如果动作包含 NaN/Inf，使用零动作并记录警告
            print("Warning: Generated action contains NaN/Inf, using zero action as fallback")
            action = torch.zeros_like(action)
        
        result = {
            'action': action,
            'action_pred': action_pred
        }
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        # 归一化输入
        assert 'valid_mask' not in batch
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        if self.rot_aug:
            nobs, nactions = self.rot_randomizer(nobs, nactions)
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        # 处理观察数据
        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory
        if self.obs_as_global_cond:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, 
                lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # 特征稳定性处理
            nobs_features = self.feature_stabilizer(nobs_features)
            # reshape back to B, Do
            global_cond = nobs_features.reshape(batch_size, -1)
        else:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # 特征稳定性处理
            nobs_features = self.feature_stabilizer(nobs_features)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()

        # 生成掩码
        condition_mask = self.mask_generator(trajectory.shape)

        # 采样噪声
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        # 为每个图像采样随机时间步
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (bsz,), device=trajectory.device
        ).long()
        # 添加噪声
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)
        
        # 计算损失掩码
        loss_mask = ~condition_mask

        # 应用条件
        noisy_trajectory[condition_mask] = cond_data[condition_mask]
        
        # 预测噪声残差
        pred = self.model(noisy_trajectory, timesteps, 
            local_cond=local_cond, global_cond=global_cond)

        # 数值稳定性检查
        if not torch.isfinite(pred).all():
            pred = torch.clamp(pred, -1e4, 1e4)
            pred = torch.nan_to_num(pred, nan=0.0, posinf=1e4, neginf=-1e4)

        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        
        # 如果损失为 NaN，使用零损失并记录警告
        if not torch.isfinite(loss):
            print("Warning: Loss is NaN/Inf, using zero loss as fallback")
            loss = torch.zeros_like(loss)
            
        return loss