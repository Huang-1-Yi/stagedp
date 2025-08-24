from typing import Dict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

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
            noise_scheduler: DDPMScheduler,
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
        self.sigma_min = 0.002  # 最小噪声水平
        self.sigma_max = 80  # 最大噪声水平

    # ========= 新的conditional_sample 方法 ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            local_cond=None, global_cond=None,
            generator=None,
            **kwargs
            ):
        # 初始化噪声轨迹
        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator
        )
        
        # 应用初始条件
        trajectory[condition_mask] = condition_data[condition_mask]
        
        # 自定义 DPM-Solver 采样
        trajectory = self.dpm_sampler(
            net=self.model,
            latents=trajectory,
            class_labels=None,
            condition=global_cond,
            unconditional_condition=None,
            num_steps=self.num_inference_steps,     # 准备采样参数：推理步
            sigma_min=self.sigma_min, 
            sigma_max=self.sigma_max, 
            schedule_type='time_uniform',
            schedule_rho=7,
            afs=False,
            denoise_to_zero=True,
            inner_steps=3,
            r=0.5,
            condition_mask=condition_mask,
            condition_data=condition_data,
            local_cond=local_cond,
            global_cond=global_cond
        )
        
        # 确保条件被应用
        trajectory[condition_mask] = condition_data[condition_mask]
        return trajectory

    # ========= 添加自定义 DPM-Sampler ============
    def dpm_sampler(
        self,
        net, 
        latents, 
        class_labels=None, 
        condition=None, 
        unconditional_condition=None,
        num_steps=None, 
        sigma_min=0.002, 
        sigma_max=80, 
        schedule_type='time_uniform', 
        schedule_rho=7, 
        afs=False, 
        denoise_to_zero=False, 
        return_inters=False,
        inner_steps=3,
        r=0.5,
        local_cond=None,  # 添加原始模型需要的参数
        global_cond=None,  # 添加原始模型需要的参数
        **kwargs
    ):
        # 时间步调度
        t_steps = self.get_schedule(num_steps, sigma_min, sigma_max, 
                            device=latents.device, 
                            schedule_type=schedule_type, 
                            schedule_rho=schedule_rho)
        
        # 主循环
        x_next = latents * t_steps[0]
        model_prev_list = []
        t_prev_list = []
        
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            x_cur = x_next
            
            # 获取当前模型输出
            # ==== 修改这里：使用原始模型的调用方式 ====
            # 将sigma值转换为整数时间步
            timestep = self.sigma_to_timestep(t_cur)
            
            # 直接调用模型，使用原始参数格式
            model_prev = net(
                x_cur, 
                timestep, 
                local_cond=local_cond, 
                global_cond=global_cond
            )
            
            # 更新历史记录
            model_prev_list.append(model_prev)
            t_prev_list.append(t_cur)
            
            # 根据阶数选择更新方法
            if len(model_prev_list) == 1:
                x_next = self.dpm_solver_first_update(
                    x_cur, t_cur, t_next, model_prev, 
                    predict_x0=True, scale=1)
            else:
                x_next = self.multistep_dpm_solver_second_update(
                    x_cur, model_prev_list, t_prev_list, t_next,
                    predict_x0=True, scale=1)
            
            # 维护历史缓冲区大小
            if len(model_prev_list) > 2:
                model_prev_list.pop(0)
                t_prev_list.pop(0)
        
        if denoise_to_zero:
            # ==== 修改这里：使用原始模型的调用方式 ====
            # 将最后一个sigma值转换为整数时间步
            timestep = self.sigma_to_timestep(t_next)
            
            # 直接调用模型，使用原始参数格式
            x_next = net(
                x_next, 
                timestep, 
                local_cond=local_cond, 
                global_cond=global_cond
            )
            
        return x_next

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        assert 'past_action' not in obs_dict
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
        local_cond = None
        global_cond = None
        if self.obs_as_global_cond:
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            global_cond = nobs_features.reshape(B, -1)
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            nobs_features = nobs_features.reshape(B, To, -1)
            cond_data = torch.zeros(size=(B, T, Da+Do), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs_features
            cond_mask[:,:To,Da:] = True

        # run sampling
        nsample = self.conditional_sample(
            cond_data, 
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            **self.kwargs)
        
        # unnormalize prediction
        naction_pred = nsample[...,:Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
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

    def compute_loss(self, batch):
        assert 'valid_mask' not in batch
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        if self.rot_aug:
            nobs, nactions = self.rot_randomizer(nobs, nactions)
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory
        if self.obs_as_global_cond:
            this_nobs = dict_apply(nobs, 
                lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            global_cond = nobs_features.reshape(batch_size, -1)
        else:
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()

        # generate impainting mask
        condition_mask = self.mask_generator(trajectory.shape)

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (bsz,), device=trajectory.device
        ).long()

        # 加噪声方式不变
        # Add noise to the clean images according to the noise magnitude at each timestep
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)
        
        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = cond_data[condition_mask]
        
        # Predict the noise residual
        pred = self.model(noisy_trajectory, timesteps, 
            local_cond=local_cond, global_cond=global_cond)

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
        return loss

    def sigma_to_timestep(self, sigma):
        """
        将sigma值转换为整数时间步
        """
        # 获取调度器配置
        num_train_timesteps = self.noise_scheduler.config.num_train_timesteps
        
        # 计算相对位置 (sigma_max -> 0, sigma_min -> num_train_timesteps-1)
        ratio = (self.sigma_max - sigma) / (self.sigma_max - self.sigma_min)
        timestep = torch.clamp((ratio * num_train_timesteps).long(), 0, num_train_timesteps-1)
        
        return timestep

    # ========= 添加辅助函数 ============
    def get_schedule(self, num_steps, sigma_min, sigma_max, device=None, 
                    schedule_type='polynomial', schedule_rho=7, net=None):
        if schedule_type == 'polynomial':
            step_indices = torch.arange(num_steps, device=device)
            t_steps = (sigma_max**(1/schedule_rho) + step_indices/(num_steps-1) * 
                    (sigma_min**(1/schedule_rho) - sigma_max**(1/schedule_rho)))**schedule_rho
        elif schedule_type == 'logsnr':
            logsnr_max = -1 * torch.log(torch.tensor(sigma_min))
            logsnr_min = -1 * torch.log(torch.tensor(sigma_max))
            t_steps = torch.linspace(logsnr_min.item(), logsnr_max.item(), 
                                    steps=num_steps, device=device)
            t_steps = (-t_steps).exp()
        elif schedule_type == 'time_uniform':
            return torch.linspace(sigma_max, sigma_min, num_steps, device=device)
        elif schedule_type == 'discrete':
            assert net is not None
            t_steps_min = net.sigma_inv(torch.tensor(sigma_min, device=device))
            t_steps_max = net.sigma_inv(torch.tensor(sigma_max, device=device))
            step_indices = torch.arange(num_steps, device=device)
            t_steps_temp = (t_steps_max + step_indices/(num_steps-1) * 
                        (t_steps_min**(1/schedule_rho) - t_steps_max))**schedule_rho
            t_steps = net.sigma(t_steps_temp)
        else:
            raise ValueError(f"Unknown schedule type {schedule_type}")
        return t_steps

    def expand_dims(self, v, dims):
        """将1D张量扩展到指定维度"""
        return v[(...,) + (None,)*(dims - 1)]

    def get_denoised(self, net, x, t, class_labels=None, condition=None, 
                    unconditional_condition=None, local_cond=None, global_cond=None):
        """
        统一去噪接口，适配不同类型的扩散模型
        """
        return net(
            x, 
            timestep=t, 
            local_cond=local_cond, 
            global_cond=global_cond
        )

    def dpm_solver_first_update(self, x, s, t, model_s=None, predict_x0=True, scale=1):
        s, t = s.reshape(-1, 1), t.reshape(-1, 1)  # 调整为1D
        lambda_s, lambda_t = -1 * s.log(), -1 * t.log()
        h = lambda_t - lambda_s
        
        phi_1 = torch.expm1(-h) if predict_x0 else torch.expm1(h)
        if predict_x0:
            x_t = (t/s) * x - scale * phi_1 * model_s
        else:
            x_t = x - scale * t * phi_1 * model_s
        return x_t

    def multistep_dpm_solver_second_update(self, x, model_prev_list, t_prev_list, t, predict_x0=True, scale=1):
        t = t.reshape(-1, 1)
        model_prev_1, model_prev_0 = model_prev_list[-2], model_prev_list[-1]
        t_prev_1, t_prev_0 = t_prev_list[-2].reshape(-1, 1), t_prev_list[-1].reshape(-1, 1)
        lambda_prev_1, lambda_prev_0, lambda_t = -1*t_prev_1.log(), -1*t_prev_0.log(), -1*t.log()

        h_0 = lambda_prev_0 - lambda_prev_1
        h = lambda_t - lambda_prev_0
        r0 = h_0 / h
        D1_0 = (1./r0) * (model_prev_0 - model_prev_1)
        phi_1 = torch.expm1(-h) if predict_x0 else torch.expm1(h)
        
        if predict_x0:
            x_t = (t/t_prev_0) * x - scale * (phi_1 * model_prev_0 + 0.5 * phi_1 * D1_0)
        else:
            x_t = x - scale * (t * phi_1 * model_prev_0 + 0.5 * t * phi_1 * D1_0)
        return x_t