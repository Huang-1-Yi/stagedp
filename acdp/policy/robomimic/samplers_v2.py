import torch
import numpy as np
# 最后输出的x_next对应t_steps[-1]，即sigma_min时的去噪结果

    # # ===== 替换原有固定值 =====
    # # 获取调度器的累积alpha乘积
    # alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(device)
    
    # # 计算噪声水平σ (sigma = sqrt((1 - α_bar)/α_bar))
    # sigmas = torch.sqrt((1 - alphas_cumprod) / alphas_cumprod)
    
    # # 设置sigma范围
    # sigma_min = sigmas[-1].item()  # 最大噪声对应训练结束时的σ
    # sigma_max = sigmas[0].item()   # 最小噪声对应训练开始时的σ
    
    # # 避免数值问题
    # sigma_min = max(sigma_min, 1e-3)
    # sigma_max = max(sigma_max, sigma_min + 1e-3)

    # # 0.自定义欧拉采样器,设置核心参数，时间表类型schedule_type='polynomial'
    # # sigma_min=0.0001          # 最小噪声水平
    # # sigma_max=2.6             # 最大噪声水平
    # schedule_rho=3              # 调度指数=3 适配 squaredcos_cap_v2调度器的特性

# 1. 欧拉采样器
def euler_sampler(self, 
        condition_data, condition_mask,
        local_cond=None, global_cond=None,
        generator=None,
        **kwargs
    ):
    """
    Euler sampler (equivalent to the DDIM sampler: https://arxiv.org/abs/2010.02502).


    参数:
        net: 封装的扩散模型
        latents: PyTorch张量，在时间`sigma_max`的输入样本
        class_labels: PyTorch张量，条件采样的条件或引导采样
        condition: PyTorch张量，用于LDM和Stable Diffusion模型的调节条件
        unconditional_condition: PyTorch张量，用于LDM和Stable Diffusion模型的无条件调节
        num_steps: `int`，带`num_steps-1`间隔的总时间步数
        sigma_min: `float`，采样结束时的sigma值
        sigma_max: `float`，采样开始时的sigma值
        schedule_type: `str`，时间调度类型。支持三种类型:
            - 'polynomial': 多项式时间调度（EDM推荐）
            - 'logsnr': 均匀logSNR时间调度（小分辨率数据集DPM-Solver推荐）
            - 'time_uniform': 均匀时间调度（高分辨率数据集DPM-Solver推荐）
            - 'discrete': LDM使用的时间调度（使用LDM和Stable Diffusion代码库的预训练扩散模型时推荐）
        schedule_rho: `float`，时间步指数。当schedule_type为['polynomial', 'time_uniform']时需要指定
        afs: `bool`，是否在采样开始时使用解析第一步（AFS）
        denoise_to_zero: `bool`，是否在采样结束时从`sigma_min`去噪到`0`
        return_inters: `bool`，是否保存中间结果（整个采样轨迹），从初始噪声状态到最终生成结果的所有中间采样状态
        return_denoised: `bool`，是否保存中间去噪结果
        return_eps: `bool`，是否保存中间分数函数（梯度）
    返回:
        PyTorch张量。如果return_inters=True则返回生成样本或采样轨迹的批次
    """
    model = self.model
    num_steps = self.num_inference_steps
    device=condition_data.device

    # 0.自定义欧拉采样器,设置核心参数
    sigma_min=0.002             # 最小噪声水平
    sigma_max=80.0              # 最大噪声水平
    schedule_type='polynomial'   # 时间表类型
    schedule_rho=7              # 调度指数
    # 功能开关
    afs=False                    # 启用解析第一步
    denoise_to_zero=False         # 最终去噪（新增）
    return_inters=False         # 返回中间结果（新增）
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
    t_steps.to(device)
    a = a.to(device)
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
            # 使用去噪数据计算当前梯度
            d_cur = (a_cur - a_denoised) / t_cur
        # 通过欧拉法更新
        a_next = a_cur + (t_next - t_cur) * d_cur
        
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
        model_output = model(a_next, t_steps[-1], local_cond=local_cond, global_cond=global_cond)
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

# 2. Heun采样器（2阶）
def heun_sampler(self, 
        condition_data, condition_mask,
        local_cond=None, global_cond=None,
        generator=None,
        **kwargs
    ):
    """
    Heun's second sampler. Introduced in EDM: https://arxiv.org/abs/2206.00364.

    参数:
        net: 封装的扩散模型
        latents: PyTorch张量，在时间`sigma_max`的输入样本
        class_labels: PyTorch张量，条件采样的条件或引导采样
        condition: PyTorch张量，用于LDM和Stable Diffusion模型的调节条件
        unconditional_condition: PyTorch张量，用于LDM和Stable Diffusion模型的无条件调节
        num_steps: `int`，带`num_steps-1`间隔的总时间步数
        sigma_min: `float`，采样结束时的sigma值
        sigma_max: `float`，采样开始时的sigma值
        schedule_type: `str`，时间调度类型。支持三种类型:
            - 'polynomial': 多项式时间调度（EDM推荐）
            - 'logsnr': 均匀logSNR时间调度（小分辨率数据集DPM-Solver推荐）
            - 'time_uniform': 均匀时间调度（高分辨率数据集DPM-Solver推荐）
            - 'discrete': LDM使用的时间调度（使用LDM和Stable Diffusion代码库的预训练扩散模型时推荐）
        schedule_rho: `float`，时间步指数。当schedule_type为['polynomial', 'time_uniform']时需要指定
        afs: `bool`，是否在采样开始时使用解析第一步（AFS）
        denoise_to_zero: `bool`，是否在采样结束时从`sigma_min`去噪到`0`
        return_inters: `bool`，是否保存中间结果（整个采样轨迹），从初始噪声状态到最终生成结果的所有中间采样状态
        return_denoised: `bool`，是否保存中间去噪结果，Whether to save intermediate results, i.e. the whole sampling trajectory.
        return_eps: `bool`，是否保存中间分数函数（梯度），Whether to save intermediate d_cur, i.e. the gradient.
    返回:
        PyTorch张量。如果return_inters=True则返回生成样本或采样轨迹的批次
    """
    model = self.model
    num_steps = self.num_inference_steps
    device=condition_data.device

    # 0.自定义欧拉采样器,设置核心参数
    sigma_min=0.002             # 最小噪声水平
    sigma_max=80.0              # 最大噪声水平
    schedule_type='polynomial'   # 时间表类型
    schedule_rho=7              # 调度指数
    # 功能开关
    afs=False                    # 启用解析第一步
    denoise_to_zero=False         # 最终去噪（新增）
    return_inters=False         # 返回中间结果（新增）
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
            a_denoised = model(a_cur, t_cur, local_cond=local_cond, global_cond=global_cond)
            # 确保条件部分不变
            a_denoised[condition_mask] = condition_data[condition_mask]  
            # 计算当前梯度
            d_cur = (a_cur - a_denoised) / t_cur
        # 欧拉法更新轨迹
        a_next = a_cur + (t_next - t_cur) * d_cur
    
        # 调用模型预测去噪结果
        a_denoised = model(a_next, t_next, local_cond=local_cond, global_cond=global_cond)
        # 确保条件部分不变
        a_denoised[condition_mask] = condition_data[condition_mask]  
        # 计算当前梯度
        d_prime = (a_next - a_denoised) / t_next
        # heun法更新轨迹
        a_next = a_cur + (t_next - t_cur) * (0.5 * d_cur + 0.5 * d_prime)

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

# 4. dpm-pp采样器（1-3阶）
def dpm_pp_sampler(self, 
        condition_data, condition_mask,
        local_cond=None, global_cond=None,
        generator=None,
        **kwargs
    ):
    """
    Multistep DPM-Solver++ sampler: https://arxiv.org/abs/2211.01095. 
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
    max_order=3                     # 最大阶数
    predict_x0=True                 # 预测x0
    lower_order_final=True          # 最后一步使用低阶方法
    prediction_type = self.noise_scheduler.config.prediction_type
    return_inters = False

    assert max_order >= 1 and max_order <= 3, "currently only support max_order=1,2,3"
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
    buffer_model = []
    buffer_t = []

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

        buffer_model.append(dynamic_thresholding_fn(a_denoised)) if predict_x0 else buffer_model.append(d_cur)
        buffer_t.append(t_cur)
        if lower_order_final:
            order = i + 1 if i + 1 < max_order else min(max_order, num_steps - (i + 1))
        else:
            order = min(max_order, i + 1)
        a_next = dpm_pp_update(a_cur, buffer_model, buffer_t, t_next, order, predict_x0=predict_x0)
        
        # 收集去噪结果和梯度
        if return_inters:
            if prediction_type == 'epsilon':
                inters_denoised.append(a_denoised.unsqueeze(0))
            elif prediction_type == 'sample':
                inters_eps.append(d_cur.unsqueeze(0))
            elif prediction_type == 'full':
                inters_at.append(a_next.unsqueeze(0))

        if len(buffer_model) >= 3:
            buffer_model = [a.detach() for a in buffer_model[-3:]]
            buffer_t = [a.detach() for a in buffer_t[-3:]]
        else:
            buffer_model = [a.detach() for a in buffer_model]
            buffer_t = [a.detach() for a in buffer_t]            
        

    
    # 追求速度而非最高质量时，跳过最终去噪步骤
    if denoise_to_zero: # 在完成所有采样步骤后，执行一个额外的去噪步骤，从 sigma_min（最小噪声水平）到 0（完全去噪）
        a_denoised = model(a_next, t_steps[-1], local_cond=local_cond, global_cond=global_cond)
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

# 5. ipndm采样器（1-4阶）
def ipndm_sampler(self, 
        condition_data, condition_mask,
        local_cond=None, global_cond=None,
        generator=None,
        **kwargs
    ):
    """
    Improved PNDM sampler: https://arxiv.org/abs/2204.13902.
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
    max_order=4                     # PNDM的最大阶数
    prediction_type = self.noise_scheduler.config.prediction_type
    return_inters = False

    assert max_order >= 1 and max_order <= 4, "PNDM only supports order 1 to 4."
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
    buffer_model = []

    # 3. 主采样循环
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):   # 0, ..., N-1
        a_cur = a_next
        if afs and i == 0:      # afs技巧：首次步使用分析解（可选），提高采样质量
            d_cur = a_cur / ((1 + t_cur**2).sqrt())
        else:                   # 其它时候直接调用模型获取去噪结果
            # 调用模型预测去噪结果
            a_denoised = model(a_cur, t_cur, local_cond=local_cond, global_cond=global_cond)
            # 确保条件部分不变
            a_denoised[condition_mask] = condition_data[condition_mask]  
            # 计算当前梯度
            d_cur = (a_cur - a_denoised) / t_cur
        
        order = min(max_order, i+1)
        if order == 1:      # First Euler step.
            x_next = a_cur + (t_next - t_cur) * d_cur
        elif order == 2:    # Use one history point.
            x_next = a_cur + (t_next - t_cur) * (3 * d_cur - buffer_model[-1]) / 2
        elif order == 3:    # Use two history points.
            x_next = a_cur + (t_next - t_cur) * (23 * d_cur - 16 * buffer_model[-1] + 5 * buffer_model[-2]) / 12
        elif order == 4:    # Use three history points.
            x_next = a_cur + (t_next - t_cur) * (55 * d_cur - 59 * buffer_model[-1] + 37 * buffer_model[-2] - 9 * buffer_model[-3]) / 24
        if return_inters:
            if prediction_type == 'epsilon':
                inters_denoised.append(a_denoised.unsqueeze(0))
            elif prediction_type == 'sample':
                inters_eps.append(d_cur.unsqueeze(0))
            elif prediction_type == 'full':
                inters_at.append(a_next.unsqueeze(0))
        
        if len(buffer_model) == max_order - 1:
            for k in range(max_order - 2):
                buffer_model[k] = buffer_model[k+1]
            buffer_model[-1] = d_cur
        else:
            buffer_model.append(d_cur)

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

# 6. 变步长的IPNDM采样器
def ipndm_v_sampler(self, 
        condition_data, condition_mask,
        local_cond=None, global_cond=None,
        generator=None,
        **kwargs
    ):
    pass

# 扩散指数积分采样器
def deis_samplerdef(self, 
        condition_data, condition_mask,
        local_cond=None, global_cond=None,
        generator=None,
        **kwargs
    ):
    """
    A pytorch implementation of DEIS: https://arxiv.org/abs/2204.13902.
    """
    pass


def optimal_sampler(self, 
        condition_data, condition_mask,
        local_cond=None, global_cond=None,
        generator=None,
        **kwargs
    ):
    """
    Optimal Euler sampler (generate images from the dataset).
    """
    pass
    for i in range(x.shape[0]):
        l2_norms = torch.norm(cifar10_dataset - x[i].unsqueeze(0), p=2, dim=(1, 2, 3))  # (50000,)
        weights = torch.nn.functional.softmax((-1 * l2_norms ** 2) / (2 * t ** 2)).reshape(-1, 1, 1, 1)
        if i == 0:
            denoised = torch.sum(torch.mul(cifar10_dataset, weights), dim=0).unsqueeze(0)
        else:
            denoised = torch.cat((denoised, torch.sum(torch.mul(cifar10_dataset, weights), dim=0).unsqueeze(0)), dim=0)














def expand_dims(v, dims):
    """
    Expand the tensor `v` to the dim `dims`.

    Args:
        v: a PyTorch tensor with shape [N].
        dim: a `int`.
    Returns:
        a PyTorch tensor with shape [N, 1, 1, ..., 1] and the total dimension is `dims`.
    """
    return v[(...,) + (None,)*(dims - 1)]
    
#----------------------------------------------------------------------------

def dynamic_thresholding_fn(x0):
    """
    The dynamic thresholding method
    """
    dims = x0.dim()
    p = 0.995
    s = torch.quantile(torch.abs(x0).reshape((x0.shape[0], -1)), p, dim=1)
    s = expand_dims(torch.maximum(s, 1. * torch.ones_like(s).to(s.device)), dims)
    x0 = torch.clamp(x0, -s, s) / s
    return x0

#----------------------------------------------------------------------------

def dpm_pp_update(x, model_prev_list, t_prev_list, t, order, predict_x0=True):
    if order == 1:
        return dpm_solver_first_update(x, t_prev_list[-1], t, model_s=model_prev_list[-1], predict_x0=predict_x0)
    elif order == 2:
        return multistep_dpm_solver_second_update(x, model_prev_list, t_prev_list, t, predict_x0=predict_x0)
    elif order == 3:
        return multistep_dpm_solver_third_update(x, model_prev_list, t_prev_list, t, predict_x0=predict_x0)
    else:
        raise ValueError("Solver order must be 1 or 2 or 3, got {}".format(order))

#----------------------------------------------------------------------------

def dpm_solver_first_update(x, s, t, model_s=None, predict_x0=True):
    s, t = s.reshape(-1, 1, 1, 1), t.reshape(-1, 1, 1, 1)
    lambda_s, lambda_t = -1 * s.log(), -1 * t.log()
    h = lambda_t - lambda_s

    phi_1 = torch.expm1(-h) if predict_x0 else torch.expm1(h)
    # VE-SDE formulation
    if predict_x0:
        x_t = (t / s) * x - phi_1 * model_s
    else:
        x_t = x - t * phi_1 * model_s
    return x_t

#----------------------------------------------------------------------------

def multistep_dpm_solver_second_update(x, model_prev_list, t_prev_list, t, predict_x0=True):
    t = t.reshape(-1, 1, 1, 1)
    model_prev_1, model_prev_0 = model_prev_list[-2], model_prev_list[-1]
    t_prev_1, t_prev_0 = t_prev_list[-2].reshape(-1, 1, 1, 1), t_prev_list[-1].reshape(-1, 1, 1, 1)
    lambda_prev_1, lambda_prev_0, lambda_t = -1 * t_prev_1.log(), -1 * t_prev_0.log(), -1 * t.log()

    h_0 = lambda_prev_0 - lambda_prev_1
    h = lambda_t - lambda_prev_0
    r0 = h_0 / h
    D1_0 = (1. / r0) * (model_prev_0 - model_prev_1)
    phi_1 = torch.expm1(-h) if predict_x0 else torch.expm1(h)
    # VE-SDE formulation
    if predict_x0:
        x_t = (t / t_prev_0) * x - phi_1 * model_prev_0 - 0.5 * phi_1 * D1_0
    else:
        x_t = x - t * phi_1 * model_prev_0 - 0.5 * t * phi_1 * D1_0
    return x_t

#----------------------------------------------------------------------------

def multistep_dpm_solver_third_update(x, model_prev_list, t_prev_list, t, predict_x0=True):
    
    t = t.reshape(-1, 1, 1, 1)
    model_prev_2, model_prev_1, model_prev_0 = model_prev_list[-3], model_prev_list[-2], model_prev_list[-1]
    
    t_prev_2, t_prev_1, t_prev_0 = t_prev_list[-3], t_prev_list[-2], t_prev_list[-1]
    t_prev_2, t_prev_1, t_prev_0 = t_prev_2.reshape(-1, 1, 1, 1), t_prev_1.reshape(-1, 1, 1, 1), t_prev_0.reshape(-1, 1, 1, 1)
    lambda_prev_2, lambda_prev_1, lambda_prev_0, lambda_t = -1 * t_prev_2.log(), -1 * t_prev_1.log(), -1 * t_prev_0.log(), -1 * t.log()

    h_1 = lambda_prev_1 - lambda_prev_2
    h_0 = lambda_prev_0 - lambda_prev_1
    h = lambda_t - lambda_prev_0
    r0, r1 = h_0 / h, h_1 / h
    D1_0 = (1. / r0) * (model_prev_0 - model_prev_1)
    D1_1 = (1. / r1) * (model_prev_1 - model_prev_2)
    D1 = D1_0 + (r0 / (r0 + r1)) * (D1_0 - D1_1)
    D2 = (1. / (r0 + r1)) * (D1_0 - D1_1)
    
    phi_1 = torch.expm1(-h) if predict_x0 else torch.expm1(h)
    phi_2 = phi_1 / h + 1. if predict_x0 else phi_1 / h - 1.
    phi_3 = phi_2 / h - 0.5
    # VE-SDE formulation
    if predict_x0:
        x_t = (t / t_prev_0) * x - phi_1 * model_prev_0 + phi_2 * D1 - phi_3 * D2
    else:
        x_t =  x - t * phi_1 * model_prev_0 - t * phi_2 * D1 - t * phi_3 * D2
    return x_t

