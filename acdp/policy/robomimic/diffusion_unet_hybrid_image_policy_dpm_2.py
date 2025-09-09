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
            diffusion_step_embed_dim=256,# 设置扩散步骤的嵌入维度，并传递给模型的初始化
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
            # obs_encoder.obs_nets['agentview_image'].nets[0].nets
        
        # obs_encoder.obs_randomizers['agentview_image']
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
        # 简单的条件输入处理 输入包括动作和观察的联合特征，总是将动作和观察数据联合处理，因此输入维度为动作维度加上观察特征维度
        input_dim = action_dim + obs_feature_dim 
        global_cond_dim = None
        if obs_as_global_cond:
            # obs_as_global_cond如果为 True，仅使用观察数据的特征作为条件输入；
            # obs_as_global_cond如果为 False，将观察特征和动作特征结合作为输入
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

    # 3. dpm-2s采样器（2阶）
    def dpm_2_sampler(self, 
            condition_data, condition_mask,
            local_cond=None, global_cond=None,
            generator=None,
            **kwargs
        ):
        """
        DPM-Solver-2 sampler: https://arxiv.org/abs/2206.00927.
        """
        model = self.model
        num_steps = self.num_inference_steps
        device=condition_data.device

        # 0.自定义欧拉采样器,设置核心参数
        sigma_min=0.002                 # 最小噪声水平
        sigma_max=80.0                  # 最大噪声水平
        schedule_type='polynomial'      # 时间表类型
        schedule_rho=7                  # 调度指数
        # 功能开关
        afs=False                        # 启用解析第一步
        denoise_to_zero=False            # 最终去噪（新增）
        return_inters=False             # 返回中间结果（新增）
        r=0.5                           # DPM-2的比例系数
        prediction_type = self.noise_scheduler.config.prediction_type
        return_inters = False

        # 1. 时间步生成（替代Diffusers的set_timesteps）sampler_type='euler' （连续sigma值）
        if schedule_type == 'polynomial':
            step_indices = torch.arange(num_steps, device=device)
            t_steps = (sigma_max ** (1 / schedule_rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / schedule_rho) - sigma_max ** (1 / schedule_rho))) ** schedule_rho
        elif schedule_type == 'logsnr':
            logsnr_max = -1 * torch.log(torch.tensor(sigma_min))
            logsnr_min = -1 * torch.log(torch.tensor(sigma_max))
            t_steps = torch.linspace(logsnr_min.item(), logsnr_max.item(), steps=num_steps, device=device)
            t_steps = (-t_steps).exp()
        elif schedule_type == 'time_uniform':
            epsilon_s = 1e-3
            vp_sigma = lambda beta_d, beta_min: lambda t: (np.e ** (0.5 * beta_d * (t ** 2) + beta_min * t) - 1) ** 0.5
            vp_sigma_inv = lambda beta_d, beta_min: lambda sigma: ((beta_min ** 2 + 2 * beta_d * (sigma ** 2 + 1).log()).sqrt() - beta_min) / beta_d
            step_indices = torch.arange(num_steps, device=device)
            vp_beta_d = 2 * (np.log(torch.tensor(sigma_min).cpu() ** 2 + 1) / epsilon_s - np.log(torch.tensor(sigma_max).cpu() ** 2 + 1)) / (epsilon_s - 1)
            vp_beta_min = np.log(torch.tensor(sigma_max).cpu() ** 2 + 1) - 0.5 * vp_beta_d
            t_steps_temp = (1 + step_indices / (num_steps - 1) * (epsilon_s ** (1 / schedule_rho) - 1)) ** schedule_rho
            t_steps = vp_sigma(vp_beta_d.clone().detach().cpu(), vp_beta_min.clone().detach().cpu())(t_steps_temp.clone().detach().cpu())
        elif schedule_type == 'discrete':
            assert model is not None
            t_steps_min = model.sigma_inv(torch.tensor(sigma_min, device=device))
            t_steps_max = model.sigma_inv(torch.tensor(sigma_max, device=device))
            step_indices = torch.arange(num_steps, device=device)
            t_steps_temp = (t_steps_max + step_indices / (num_steps - 1) * (t_steps_min ** (1 / schedule_rho) - t_steps_max)) ** schedule_rho
            t_steps = model.sigma(t_steps_temp)
        else:
            raise ValueError("Got wrong schedule type {}".format(schedule_type))

        # 2. 初始化噪声轨迹
        a = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator
        )

        # 3. 应用条件约束
        a_next = a * t_steps[0]
        a_next[condition_mask] = condition_data[condition_mask]
        if return_inters:
            inters_at = [a_next.unsqueeze(0)]
            inters_denoised, inters_eps = [], []

        # 3. 主采样循环
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):   # 0, ..., N-1
            a_cur = a_next
            if afs and i == 0:      # afs技巧：首次步使用分析解（可选），提高采样质量
                d_cur = a_cur / ((1 + t_cur**2).sqrt())
            else:                   # 其它时候直接调用模型获取去噪结果
                # 调用模型预测去噪结果
                model_output = model(a_cur, t_cur, local_cond=local_cond, global_cond=global_cond)

                # 根据prediction_type转换模型输出为去噪数据
                if prediction_type == 'epsilon':  # 模型预测噪声
                    a_denoised = a_cur - t_cur * model_output
                elif prediction_type == 'sample':  # 模型预测去噪后的数据
                    a_denoised = model_output
                elif prediction_type == 'v_prediction':  # 模型预测速度
                    # 速度与噪声的关系: v = -σ·ε
                    a_denoised = a_cur + t_cur * model_output
                else:
                    raise ValueError(f"Unsupported prediction type {prediction_type}")
                
                # 确保条件部分不变
                a_denoised[condition_mask] = condition_data[condition_mask]  
                # 计算当前梯度
                d_cur = (a_cur - a_denoised) / t_cur
            
            # dpm-2法更新轨迹
            t_mid = (t_next ** r) * (t_cur ** (1 - r))
            a_next = a_cur + (t_mid - t_cur) * d_cur
        
            # 调用模型预测去噪结果
            model_output = model(a_next, t_next, local_cond=local_cond, global_cond=global_cond)
            # 根据prediction_type转换模型输出为去噪数据
            if prediction_type == 'epsilon':  # 模型预测噪声
                a_denoised = a_cur - t_cur * model_output
            elif prediction_type == 'sample':  # 模型预测去噪后的数据
                a_denoised = model_output
            elif prediction_type == 'v_prediction':  # 模型预测速度
                # 速度与噪声的关系: v = -σ·ε
                a_denoised = a_cur + t_cur * model_output
            else:
                raise ValueError(f"Unsupported prediction type {prediction_type}")
            
            # 确保条件部分不变
            a_denoised[condition_mask] = condition_data[condition_mask]  
            # 计算当前梯度
            d_prime = (a_next - a_denoised) / t_mid
            # dpm-2法更新轨迹
            a_next = a_cur + (t_next - t_cur) * ((1 / (2*r)) * d_prime + (1 - 1 / (2*r)) * d_cur)

            # 收集去噪结果和梯度
            if return_inters:
                if prediction_type == 'epsilon':
                    inters_denoised.append(a_denoised.unsqueeze(0))
                elif prediction_type == 'sample':
                    inters_eps.append(d_cur.unsqueeze(0))
                elif prediction_type == 'full':
                    inters_at.append(a_next.unsqueeze(0))
        
        # 追求速度而非最高质量时，跳过最终去噪步骤
        if denoise_to_zero: # 在完成所有采样步骤后，执行一个额外的去噪步骤，从 sigma_min（最小噪声水平）到 0（完全去噪）
            a_denoised = model(a_next, t_steps[-1], local_cond=local_cond, global_cond=global_cond)
            
            # 根据prediction_type转换模型输出为去噪数据
            if prediction_type == 'epsilon':  # 模型预测噪声
                a_denoised = a_cur - t_cur * model_output
            elif prediction_type == 'sample':  # 模型预测去噪后的数据
                a_denoised = model_output
            elif prediction_type == 'v_prediction':  # 模型预测速度
                # 速度与噪声的关系: v = -σ·ε
                a_denoised = a_cur + t_cur * model_output
            else:
                raise ValueError(f"Unsupported prediction type {prediction_type}")
            
            # 使用最后一个时间步，确保应用最终条件约束
            a_denoised[condition_mask] = condition_data[condition_mask]  
            d_cur = (a_next - a_denoised) / t_next
            if return_inters:
                if prediction_type == 'epsilon':
                    inters_denoised.append(a_denoised.unsqueeze(0))
                elif prediction_type == 'sample':
                    inters_eps.append(d_cur.unsqueeze(0))
                elif prediction_type == 'full':
                    inters_at.append(a_next.unsqueeze(0))
        
        if return_inters:
            return torch.cat(inters_at, dim=0).to(a.device), torch.cat(inters_denoised, dim=0).to(a.device), torch.cat(inters_eps, dim=0).to(a.device)
        return a_denoised


    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        功能：根据当前观察数据预测动作。
        参数：
            obs_dict: 包含观察数据的字典，必须包括键 "obs"。
        关键步骤：
            1.对观察数据进行归一化处理。
            2.构建输入数据（观察和动作数据）。
            3.根据是否使用全局条件 (obs_as_global_cond)，分别处理本地条件和全局条件。
            4.调用 conditional_sample 方法进行条件采样，生成轨迹。
            5.生成预测的动作，并将其反归一化（恢复到原始尺度）。
            6.返回预测的动作。
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
        local_cond = None
        global_cond = None
        if self.obs_as_global_cond:
            # condition through global feature
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, Do
            global_cond = nobs_features.reshape(B, -1)
            # empty data for action
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, To, Do
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
        """
        功能：设置归一化器（LinearNormalizer）。
        参数：
            normalizer: 传入的归一化器。
        关键步骤：
            加载归一化器的状态字典。
        """
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        """
        功能：计算模型的损失函数。
        参数：
            batch: 输入的批次数据，包含观察数据和动作数据。
        关键步骤：
            1.对输入数据进行归一化处理。
            2.根据是否使用旋转增强 (rot_aug)，对数据进行旋转增强。
            3.处理观察数据，通过全局条件或其他方式构建条件数据。
            4.使用 mask_generator 生成损失掩码。
            5.计算噪声并添加到图像中，模拟扩散过程。
            6.根据预测类型（epsilon 或 sample）计算目标（噪声或轨迹）。
            7.使用均方误差（MSE）损失函数计算损失。
            8.计算和返回最终的平均损失。
        """
        # normalize input
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
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, 
                lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, Do
            global_cond = nobs_features.reshape(batch_size, -1)
        else:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
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
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
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
