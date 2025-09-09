from typing import Dict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
# from diffusers.schedulers.scheduling_ddpm import DDPMScheduler


# 新增：导入自定义采样器
from acdp.sampler.solvers_amed import (
    amed_sampler, 
    euler_sampler,
    ipndm_sampler,
    dpm_2_sampler,
    dpm_pp_sampler,
    heun_sampler,
    get_schedule  # 时间调度函数
)

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
            # noise_scheduler: DDPMScheduler,# 移除
            horizon, 
            n_action_steps, 
            n_obs_steps,
            sampler_type: str = 'amed',  # 新增：采样器类型选择
            sampler_kwargs: dict = None,  # 新增：采样器参数
            # num_inference_steps: int = 20,  # 修改：固定推理步数
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
        """
        修改：
          1. 移除了noise_scheduler参数
          2. 新增sampler_type和sampler_kwargs参数
          3. 固定num_inference_steps
        功能：初始化 DiffusionUnetHybridImagePolicy 类，设置模型的所有超参数，加载配置，初始化各类组件。
        参数：
            shape_meta: 包含动作和观察数据的形状信息。
            noise_scheduler: 用于控制扩散过程的调度器(DDPM调度器)。
            horizon: 轨迹的长度。
            n_action_steps: 每次动作的时间步长。
            n_obs_steps: 观察数据的时间步长。
            其他相关的配置参数，如 crop_shape、diffusion_step_embed_dim 等。
        关键步骤：
            解析形状元数据 (shape_meta)。
            配置观察数据的类型和模态（RGB、低维、深度等)。
            加载 Robomimic 配置，设定观察数据的随机化方式（如裁剪）。
            初始化 obs_encoder（用于处理观察数据）和 model（扩散模型）。
            初始化 noise_scheduler 和 mask_generator 等其他相关组件。
        """
        super().__init__()

        # 解析形状元数据 (保持不变)
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

        # 获取robomimic配置 (保持不变)
        config = get_robomimic_config(
            algo_name='bc_rnn',
            hdf5_type='image',
            task_name='square',
            dataset_type='ph')
        
        with config.unlocked():
            # 设置配置 (保持不变)
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


        # 初始化观察工具 (保持不变)
        ObsUtils.initialize_obs_utils_with_config(config)

        # 加载模型 (保持不变)
        policy: PolicyAlgo = algo_factory(
                algo_name=config.algo_name,
                config=config,
                obs_key_shapes=obs_key_shapes,
                ac_dim=action_dim,
                device='cpu',
            )

        obs_encoder = policy.nets['policy'].nets['encoder'].nets['obs']
        
        if obs_encoder_group_norm:
            # 替换批归一化为组归一化 (保持不变)
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

        # 创建扩散模型 (保持不变)
        obs_feature_dim = obs_encoder.output_shape()[0]
        # 简单的条件输入处理 输入包括动作和观察的联合特征
        # 总是将动作和观察数据联合处理，因此输入维度为动作维度加上观察特征维度
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
        # 移除: self.noise_scheduler
        # self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        self.normalizer = LinearNormalizer()
        self.rot_randomizer = RotRandomizer()

        ##########################################################################
        # 设置采样器
        self.sampler_type = sampler_type
        self.sampler_kwargs = sampler_kwargs or {}
        self.sampler_kwargs.setdefault('sigma_min', 0.002)
        self.sampler_kwargs.setdefault('sigma_max', 80.0)
        self.sampler_kwargs.setdefault('schedule_type', 'polynomial')
        self.sampler_kwargs.setdefault('schedule_rho', 7)
        self.sampler_kwargs.setdefault('afs', True)  # 启用解析第一步
        self.sampler_kwargs.setdefault('prediction_type', 'epsilon')
        self.sampler_kwargs.setdefault('AMED_predictor', False)
        print(f"Using {self.sampler_type} sampler with args {self.sampler_kwargs}")
        ############################################################################

        # 选择采样器函数
        self.sampler_fn = {
            'amed': amed_sampler,
            'euler': euler_sampler,
            'ipndm': ipndm_sampler,
            'dpm_2': dpm_2_sampler,
            'dpm_pp': dpm_pp_sampler,
            'heun': heun_sampler
        }.get(sampler_type, amed_sampler)
        ##########################################################################

        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.rot_aug = rot_aug
        self.kwargs = kwargs

        self.num_inference_steps = num_inference_steps

        print("Diffusion params: %e" % sum(p.numel() for p in self.model.parameters()))
        print("Vision params: %e" % sum(p.numel() for p in self.obs_encoder.parameters()))
    
    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            local_cond=None, global_cond=None,
            generator=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        """
        功能：基于给定的条件数据和条件掩码，通过扩散模型进行采样。
        参数：
            condition_data、condition_mask: 条件数据和掩码。
            local_cond、global_cond: 本地条件和全局条件（根据 obs_as_global_cond 来决定）。
            generator: 随机数生成器，用于采样。
        关键步骤：
            初始化扩散轨迹。
            设置调度器的时间步。
            在每个时间步上，应用条件数据和掩码进行采样。
            调用扩散模型的输出，并通过调度器一步步反向扩散生成目标数据。
        """
        """
        添加调试打印语句 
        """
        print(f"[conditional_sample] 开始条件采样")
        print(f"[conditional_sample] 输入形状: condition_data={condition_data.shape}, condition_mask={condition_mask.shape}")
        
        # 获取设备信息
        print(f"Sampling with {self.sampler_type} sampler")
        print(f"device: {condition_data.device}, dtype: {condition_data.dtype}")
        print(f"self.device: {self.device}, self.dtype: {self.dtype}")
        
        # 生成初始噪声轨迹
        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,   # 使用输入数据的设备
            generator=generator)
        
        # 生成时间步调度
        t_steps = get_schedule(
            self.num_inference_steps,
            self.sampler_kwargs['sigma_min'],
            self.sampler_kwargs['sigma_max'],
            device=self.device,
            schedule_type=self.sampler_kwargs['schedule_type'],
            schedule_rho=self.sampler_kwargs['schedule_rho']
        )
        print(f"[conditional_sample] 时间步调度: t_steps={t_steps.shape}, 值范围=[{t_steps.min().item()}, {t_steps.max().item()}]")
        
        # 创建条件函数（处理条件数据）
        def denoise_fn(x, t):
            # 应用条件约束
            x = torch.where(condition_mask, condition_data, x)
            # 处理时间步形状 (B,) -> (B,1,1,1)
            t = t.reshape(-1, 1, 1, 1)
            # 调用模型
            return self.model(x, t, local_cond=local_cond, global_cond=global_cond)

        # 使用自定义采样器
        trajectory = self.sampler_fn(
            net=denoise_fn,  # 包装模型为条件函数
            latents=trajectory,
            class_labels=None,
            condition=global_cond,  # 全局条件
            num_steps=self.num_inference_steps,
            sigma_min=self.sampler_kwargs['sigma_min'],
            sigma_max=self.sampler_kwargs['sigma_max'],
            schedule_type=self.sampler_kwargs['schedule_type'],
            schedule_rho=self.sampler_kwargs['schedule_rho'],
            afs=self.sampler_kwargs['afs'],
            AMED_predictor=self.sampler_kwargs['AMED_predictor'],
            **self.kwargs
        )
        print(f"[conditional_sample] 采样后轨迹形状: {trajectory.shape}")

        # 确保条件数据被正确应用
        trajectory = torch.where(condition_mask, condition_data, trajectory)
        print(f"[conditional_sample] 最终轨迹形状: {trajectory.shape}")

        return trajectory


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
        print(f"[predict_action] 开始预测动作")
        print(f"[predict_action] 输入观察数据: {list(obs_dict.keys())}")

        assert 'past_action' not in obs_dict # not implemented yet
        # 归一化输入
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps
        print(f"[predict_action] 归一化后数据形状: B={B}, To={To}, T={T}, Da={Da}, Do={Do}")

        # 设备信息
        device = self.device
        dtype = self.dtype

        # 处理观察数据
        local_cond = None
        global_cond = None
        if self.obs_as_global_cond:
            print(f"[predict_action] 使用观察作为全局条件")
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
            print(f"[predict_action] 条件数据形状: {cond_data.shape}, 条件掩码形状: {cond_mask.shape}")

        # 运行采样 (接口不变)
        nsample = self.conditional_sample(
            cond_data, 
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            **self.kwargs
        )
        print(f"[predict_action] 采样结果形状: {nsample.shape}")

        # 反归一化预测
        naction_pred = nsample[...,:Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)
        print(f"[predict_action] 动作预测形状: {action_pred.shape}")

        # 获取动作
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:,start:end]
        print(f"[predict_action] 最终动作形状: {action.shape}")
        
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
        修改：移除Diffusers依赖，手动实现加噪过程

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
        # 归一化输入
        assert 'valid_mask' not in batch
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])

        if self.rot_aug and self.rot_randomizer is not None:
            nobs, nactions = self.rot_randomizer(nobs, nactions)
        
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        # 处理观察数据 (保持不变)
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

        # 生成掩码 (保持不变)
        condition_mask = self.mask_generator(trajectory.shape)

        # === 手动实现加噪过程 ===
        # 采样随机时间步
        timesteps = torch.randint(
            0, self.num_inference_steps, 
            (batch_size,), device=trajectory.device
        ).long()

        # 计算alpha和sigma
        sigma_min = self.sampler_kwargs['sigma_min']
        sigma_max = self.sampler_kwargs['sigma_max']
        # 将时间步映射到[0,1]区间
        t_ratio = timesteps.float() / (self.num_inference_steps - 1)
        
        # 计算噪声水平 (对数均匀采样)
        sigma = torch.exp(t_ratio * math.log(sigma_max) + 
                         (1 - t_ratio) * math.log(sigma_min))
        # 计算alpha和缩放因子
        alpha = 1.0 / torch.sqrt(1.0 + sigma**2)
        scale = alpha * sigma

        # 确保alpha和scale有正确的维度用于广播
        alpha = alpha.view(-1, 1, 1)
        scale = scale.view(-1, 1, 1)

        # 采样噪声
        noise = torch.randn_like(trajectory)
        
        # 加噪 (扩散过程)
        noisy_trajectory = alpha * trajectory + scale * noise

        # 计算损失掩码
        loss_mask = ~condition_mask
        # 应用条件约束
        noisy_trajectory[condition_mask] = cond_data[condition_mask]
        # 预测噪声残差
        pred = self.model(noisy_trajectory, timesteps, 
            local_cond=local_cond, global_cond=global_cond)
        pred_type = self.sampler_kwargs['prediction_type']
        if pred_type == 'epsilon':
            target = noise  # 默认使用epsilon预测
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        # 计算损失
        loss = F.mse_loss(pred, target, reduction='none')
        
        loss = loss * loss_mask.type(loss.dtype)
        
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        
        loss = loss.mean()
        return loss
