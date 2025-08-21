"""
获取采样的时间调度：根据不同的时间调度类型，计算时间步的值。
get_schedule
    polynomial——EDM模型——多项式调度通过指数公式计算     调度通过指数公式计算
    logsnr——小分辨率图像——均匀logSNR分布               调度通过对数公式计算
    time_uniform——高分辨率图像——均匀时间分布           调度通过 beta 参数计算
    discrete——LDM/Stable Diffusion——适配离散模型      调度依赖于预训练模型的 sigma 函数


通过计算张量的分位数，动态调整阈值范围，并对张量进行裁剪和归一化
dynamic_thresholding_fn

DPM-Solver++ 的更新方法：根据阶数选择不同的更新方法：一阶、二阶或三阶更新。每种方法使用不同的公式计算更新后的状态
dpm_pp_update

DPM-Solver 的一阶更新方法：通过计算时间步之间的差值和指数函数，更新状态。根据 predict_x0 的值选择不同的公式
dpm_solver_first_update

DPM-Solver 的二阶多步更新方法：通过计算时间步之间的差值和模型之间的差值，结合二阶公式更新状态
multistep_dpm_solver_second_update

DPM-Solver 的三阶多步更新方法：通过计算时间步和模型之间的多阶差值，结合三阶公式更新状态。使用 phi 函数计算高阶项
multistep_dpm_solver_third_update
"""
import torch
import numpy as np

#----------------------------------------------------------------------------

def get_schedule(num_steps, sigma_min, sigma_max, device=None, schedule_type='polynomial', schedule_rho=7, net=None):
    """
    获取采样的时间调度。

    参数：
        num_steps: 一个 `int` 类型。时间步数的总数，间隔为 `num_steps-1`。
        sigma_min: 一个 `float` 类型。采样结束时的 sigma 值。
        sigma_max: 一个 `float` 类型。采样开始时的 sigma 值。
        device: 一个 torch 设备。
        schedule_type: 一个 `str` 类型。时间调度的类型。支持以下几种类型：
            - 'polynomial': 多项式时间调度。（推荐用于 EDM）
            - 'logsnr': 均匀 logSNR 时间调度。（推荐用于小分辨率数据集的 DPM-Solver）
            - 'time_uniform': 均匀时间调度。（推荐用于高分辨率数据集的 DPM-Solver）
            - 'discrete': LDM 使用的时间调度。（推荐用于 LDM 和 Stable Diffusion 代码库中的预训练扩散模型）
        schedule_rho: 一个 `float` 类型。时间步指数。
        net: 一个预训练的扩散模型。当 schedule_type == 'discrete' 时需要。
    返回值：
        一个形状为 [num_steps] 的 PyTorch 张量。

    实现逻辑分析：
    根据不同的时间调度类型，计算时间步的值。多项式调度通过指数公式计算，logsnr 调度通过对数公式计算，time_uniform 调度通过 beta 参数计算，discrete 调度依赖于预训练模型的 sigma 函数。
    """
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
        assert net is not None
        t_steps_min = net.sigma_inv(torch.tensor(sigma_min, device=device))
        t_steps_max = net.sigma_inv(torch.tensor(sigma_max, device=device))
        step_indices = torch.arange(num_steps, device=device)
        t_steps_temp = (t_steps_max + step_indices / (num_steps - 1) * (t_steps_min ** (1 / schedule_rho) - t_steps_max)) ** schedule_rho
        t_steps = net.sigma(t_steps_temp)
    else:
        raise ValueError("Got wrong schedule type {}".format(schedule_type))
    return t_steps.to(device)


# Copied from the DPM-Solver codebase (https://github.com/LuChengTHU/dpm-solver).
# Different from the original codebase, we use the VE-SDE formulation for simplicity
# while the official implementation uses the equivalent VP-SDE formulation. 
##############################
### Utils for DPM-Solver++ ###
##############################
#----------------------------------------------------------------------------

def expand_dims(v, dims):
    """
    将张量 `v` 扩展到维度 `dims`。

    参数：
        v: 一个形状为 [N] 的 PyTorch 张量。
        dims: 一个 `int` 类型。
    返回值：
        一个形状为 [N, 1, 1, ..., 1] 的 PyTorch 张量，总维度为 `dims`。

    实现逻辑分析：
    通过在张量的末尾添加多个 None 维度，扩展张量的维度。
    """
    return v[(...,) + (None,)*(dims - 1)]
    
#----------------------------------------------------------------------------

def dynamic_thresholding_fn(x0):
    """
    动态阈值方法。

    参数：
        x0: 输入张量。
    返回值：
        经过动态阈值处理后的张量。

    实现逻辑分析：
    通过计算张量的分位数，动态调整阈值范围，并对张量进行裁剪和归一化。
    """
    dims = x0.dim()
    p = 0.995
    s = torch.quantile(torch.abs(x0).reshape((x0.shape[0], -1)), p, dim=1)
    s = expand_dims(torch.maximum(s, 1. * torch.ones_like(s).to(s.device)), dims)
    x0 = torch.clamp(x0, -s, s) / s
    return x0

#----------------------------------------------------------------------------

def dpm_pp_update(x, model_prev_list, t_prev_list, t, order, predict_x0=True, scale=1):
    """
    DPM-Solver++ 的更新方法。

    参数：
        x: 当前状态。
        model_prev_list: 之前的模型列表。
        t_prev_list: 之前的时间步列表。
        t: 当前时间步。
        order: 更新的阶数。
        predict_x0: 是否预测 x0。
        scale: 缩放因子。
    返回值：
        更新后的状态。

    实现逻辑分析：
    根据阶数选择不同的更新方法：一阶、二阶或三阶更新。每种方法使用不同的公式计算更新后的状态。
    """
    if order == 1:
        return dpm_solver_first_update(x, t_prev_list[-1], t, model_s=model_prev_list[-1], predict_x0=predict_x0, scale=scale)
    elif order == 2:
        return multistep_dpm_solver_second_update(x, model_prev_list, t_prev_list, t, predict_x0=predict_x0, scale=scale)
    elif order == 3:
        return multistep_dpm_solver_third_update(x, model_prev_list, t_prev_list, t, predict_x0=predict_x0, scale=scale)
    else:
        raise ValueError("Solver order must be 1 or 2 or 3, got {}".format(order))

#----------------------------------------------------------------------------

def dpm_solver_first_update(x, s, t, model_s=None, predict_x0=True, scale=1):
    """
    DPM-Solver 的一阶更新方法。

    参数：
        x: 当前状态。
        s: 起始时间步。
        t: 目标时间步。
        model_s: 当前模型。
        predict_x0: 是否预测 x0。
        scale: 缩放因子。
    返回值：
        更新后的状态。

    实现逻辑分析：
    通过计算时间步之间的差值和指数函数，更新状态。根据 predict_x0 的值选择不同的公式。
    """
    s, t = s.reshape(-1, 1, 1, 1), t.reshape(-1, 1, 1, 1)
    lambda_s, lambda_t = -1 * s.log(), -1 * t.log()
    h = lambda_t - lambda_s

    phi_1 = torch.expm1(-h) if predict_x0 else torch.expm1(h)
    if predict_x0:
        x_t = (t / s) * x - scale * phi_1 * model_s
    else:
        x_t = x - scale * t * phi_1 * model_s
    return x_t

#----------------------------------------------------------------------------

def multistep_dpm_solver_second_update(x, model_prev_list, t_prev_list, t, predict_x0=True, scale=1):
    """
    DPM-Solver 的二阶多步更新方法。

    参数：
        x: 当前状态。
        model_prev_list: 之前的模型列表。
        t_prev_list: 之前的时间步列表。
        t: 当前时间步。
        predict_x0: 是否预测 x0。
        scale: 缩放因子。
    返回值：
        更新后的状态。

    实现逻辑分析：
    通过计算时间步之间的差值和模型之间的差值，结合二阶公式更新状态。
    """
    t = t.reshape(-1, 1, 1, 1)
    model_prev_1, model_prev_0 = model_prev_list[-2], model_prev_list[-1]
    t_prev_1, t_prev_0 = t_prev_list[-2].reshape(-1, 1, 1, 1), t_prev_list[-1].reshape(-1, 1, 1, 1)
    lambda_prev_1, lambda_prev_0, lambda_t = -1 * t_prev_1.log(), -1 * t_prev_0.log(), -1 * t.log()

    h_0 = lambda_prev_0 - lambda_prev_1
    h = lambda_t - lambda_prev_0
    r0 = h_0 / h
    D1_0 = (1. / r0) * (model_prev_0 - model_prev_1)
    phi_1 = torch.expm1(-h) if predict_x0 else torch.expm1(h)
    if predict_x0:
        x_t = (t / t_prev_0) * x - scale * (phi_1 * model_prev_0 + 0.5 * phi_1 * D1_0)
    else:
        x_t = x - scale * (t * phi_1 * model_prev_0 + 0.5 * t * phi_1 * D1_0)
    return x_t

#----------------------------------------------------------------------------

def multistep_dpm_solver_third_update(x, model_prev_list, t_prev_list, t, predict_x0=True, scale=1):
    """
    DPM-Solver 的三阶多步更新方法。

    参数：
        x: 当前状态。
        model_prev_list: 之前的模型列表。
        t_prev_list: 之前的时间步列表。
        t: 当前时间步。
        predict_x0: 是否预测 x0。
        scale: 缩放因子。
    返回值：
        更新后的状态。

    实现逻辑分析：
    通过计算时间步和模型之间的多阶差值，结合三阶公式更新状态。使用 phi 函数计算高阶项。
    """
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
    if predict_x0:
        x_t = (t / t_prev_0) * x - scale * (phi_1 * model_prev_0 - phi_2 * D1 + phi_3 * D2)
    else:
        x_t =  x - scale * (t * phi_1 * model_prev_0 + t * phi_2 * D1 + t * phi_3 * D2)
    return x_t


