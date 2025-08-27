import torch
import torch
from solver_utils import *
from torch.nn.parallel import DataParallel
from concurrent.futures import ThreadPoolExecutor
from torch_utils import distributed as dist

# ----------------------------------------------------------------------------
# 1) 单步 + 需要 inner_steps
# (1) DPM-Solver++ (2S)：数据预测（与题主示例同风格，支持 r、AFS、几何阶段点）  1
# ----------------------------------------------------------------------------
def dpmpp_2s_with_inner(
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
    **kwargs
):
    # Time step discretization.
    t_steps = get_schedule(num_steps, sigma_min, sigma_max, device=latents.device,
                           schedule_type=schedule_type, schedule_rho=schedule_rho)

    # Main sampling loop.
    x_next = latents * t_steps[0]
    inters = [x_next.unsqueeze(0)]
    x_list = [x_next]

    for i_out, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):  # 0, ..., N-1
        x_cur = x_next
        # Inner schedule (geometric/polynomial)
        t_s = get_schedule(inner_steps, t_next, t_cur, device=latents.device,
                           schedule_type='polynomial', schedule_rho=7)

        for j_in, (t_c, t_n) in enumerate(zip(t_s[:-1], t_s[1:])):
            # 1) slope at t_c (AFS on the first inner step)
            use_afs = (afs and j_in == 0)
            if use_afs:
                d_cur = x_cur / ((1 + t_c**2).sqrt())
            else:
                denoised_c = get_denoised(net, x_cur, t_c,
                                          class_labels=class_labels,
                                          condition=condition,
                                          unconditional_condition=unconditional_condition)
                d_cur = (x_cur - denoised_c) / t_c

            # 2) stage point (geometric, RK2-r)
            t_mid = (t_n ** r) * (t_c ** (1 - r))
            x_pred = x_cur + (t_mid - t_c) * d_cur

            # 3) slope at t_mid
            denoised_mid = get_denoised(net, x_pred, t_mid,
                                        class_labels=class_labels,
                                        condition=condition,
                                        unconditional_condition=unconditional_condition)
            d_mid = (x_pred - denoised_mid) / t_mid

            # 4) RK2-r composition to t_n
            x_cur = x_cur + (t_n - t_c) * ((1 / (2 * r)) * d_mid + (1 - 1 / (2 * r)) * d_cur)

        x_next = x_cur
        x_list.append(x_next)
        if return_inters:
            inters.append(x_cur.unsqueeze(0))

    if denoise_to_zero:
        x_next = get_denoised(net, x_next, t_next,
                              class_labels=class_labels,
                              condition=condition,
                              unconditional_condition=unconditional_condition)
        if return_inters:
            inters.append(x_next.unsqueeze(0))

    if return_inters:
        return torch.cat(inters, dim=0).to(latents.device)
    return x_next, x_list


# ----------------------------------------------------------------------------
#  (2) DPM-Solver 二阶（“噪声”视角）：单步 + inner_steps                        
# 说明：保持题主代码风格；用 denoised=x0̂ 先换算 epŝ=(x - x0̂)/t，再走与上面相同的 RK2-r 合成。
# ----------------------------------------------------------------------------
def dpm_solver2_with_inner(
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
    **kwargs
):
    t_steps = get_schedule(num_steps, sigma_min, sigma_max, device=latents.device,
                           schedule_type=schedule_type, schedule_rho=schedule_rho)
    x_next = latents * t_steps[0]
    inters = [x_next.unsqueeze(0)]
    x_list = [x_next]

    for i_out, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x_cur = x_next
        t_s = get_schedule(inner_steps, t_next, t_cur, device=latents.device,
                           schedule_type='polynomial', schedule_rho=7)

        for j_in, (t_c, t_n) in enumerate(zip(t_s[:-1], t_s[1:])):
            # epŝ at t_c (AFS optional: still reuse your analytic drift)
            use_afs = (afs and j_in == 0)
            if use_afs:
                # approximate d = (x - x0̂)/t ≈ x / sqrt(1 + t^2)
                d_cur = x_cur / ((1 + t_c**2).sqrt())
            else:
                x0_hat_c = get_denoised(net, x_cur, t_c,
                                        class_labels=class_labels,
                                        condition=condition,
                                        unconditional_condition=unconditional_condition)
                # epŝ = (x - x0̂) / t ; here d_cur equals epŝ (EDM)
                d_cur = (x_cur - x0_hat_c) / t_c

            # stage point & predict
            t_mid = (t_n ** r) * (t_c ** (1 - r))
            x_pred = x_cur + (t_mid - t_c) * d_cur

            # slope at stage (via epŝ from x0̂)
            x0_hat_mid = get_denoised(net, x_pred, t_mid,
                                      class_labels=class_labels,
                                      condition=condition,
                                      unconditional_condition=unconditional_condition)
            d_mid = (x_pred - x0_hat_mid) / t_mid

            # RK2-r compose
            x_cur = x_cur + (t_n - t_c) * ((1 / (2 * r)) * d_mid + (1 - 1 / (2 * r)) * d_cur)

        x_next = x_cur
        x_list.append(x_next)
        if return_inters:
            inters.append(x_cur.unsqueeze(0))

    if denoise_to_zero:
        x_next = get_denoised(net, x_next, t_next,
                              class_labels=class_labels,
                              condition=condition,
                              unconditional_condition=unconditional_condition)
        if return_inters:
            inters.append(x_next.unsqueeze(0))

    if return_inters:
        return torch.cat(inters, dim=0).to(latents.device)
    return x_next, x_list


# ----------------------------------------------------------------------------
# 2) 单步 + 不需要 inner_steps（每个外层步只做一次二阶 2S）
# (1) DPM-Solver++ (2S)                                              3
# ----------------------------------------------------------------------------
def dpmpp_2s_singlestep(
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
    r=0.5,
    denoise_to_zero=False,
    return_inters=False,
    **kwargs
):
    t_steps = get_schedule(num_steps, sigma_min, sigma_max, device=latents.device,
                           schedule_type=schedule_type, schedule_rho=schedule_rho)
    x_next = latents * t_steps[0]
    inters = [x_next.unsqueeze(0)]
    x_list = [x_next]

    for i_out, (t_c, t_n) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        # slope at t_c
        x0_hat_c = get_denoised(net, x_next, t_c,
                                class_labels=class_labels,
                                condition=condition,
                                unconditional_condition=unconditional_condition)
        d_cur = (x_next - x0_hat_c) / t_c
        # stage & predict
        t_mid = (t_n ** r) * (t_c ** (1 - r))
        x_pred = x_next + (t_mid - t_c) * d_cur
        # slope at stage
        x0_hat_mid = get_denoised(net, x_pred, t_mid,
                                  class_labels=class_labels,
                                  condition=condition,
                                  unconditional_condition=unconditional_condition)
        d_mid = (x_pred - x0_hat_mid) / t_mid
        # compose
        x_next = x_next + (t_n - t_c) * ((1 / (2 * r)) * d_mid + (1 - 1 / (2 * r)) * d_cur)

        x_list.append(x_next)
        if return_inters:
            inters.append(x_next.unsqueeze(0))

    if denoise_to_zero:
        x_next = get_denoised(net, x_next, t_n,
                              class_labels=class_labels,
                              condition=condition,
                              unconditional_condition=unconditional_condition)
        if return_inters:
            inters.append(x_next.unsqueeze(0))

    if return_inters:
        return torch.cat(inters, dim=0).to(latents.device)
    return x_next, x_list


# ----------------------------------------------------------------------------
# (2) DPM-Solver 二阶：单步 + 无 inner_steps（同样通过 x0̂→epŝ 的等价转换）
# ----------------------------------------------------------------------------
def dpm_solver2_singlestep(
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
    r=0.5,
    denoise_to_zero=False,
    return_inters=False,
    **kwargs
):
    t_steps = get_schedule(num_steps, sigma_min, sigma_max, device=latents.device,
                           schedule_type=schedule_type, schedule_rho=schedule_rho)
    x_next = latents * t_steps[0]
    inters = [x_next.unsqueeze(0)]
    x_list = [x_next]

    for i_out, (t_c, t_n) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x0_hat_c = get_denoised(net, x_next, t_c,
                                class_labels=class_labels,
                                condition=condition,
                                unconditional_condition=unconditional_condition)
        d_cur = (x_next - x0_hat_c) / t_c
        t_mid = (t_n ** r) * (t_c ** (1 - r))
        x_pred = x_next + (t_mid - t_c) * d_cur
        x0_hat_mid = get_denoised(net, x_pred, t_mid,
                                  class_labels=class_labels,
                                  condition=condition,
                                  unconditional_condition=unconditional_condition)
        d_mid = (x_pred - x0_hat_mid) / t_mid
        x_next = x_next + (t_n - t_c) * ((1 / (2 * r)) * d_mid + (1 - 1 / (2 * r)) * d_cur)

        x_list.append(x_next)
        if return_inters:
            inters.append(x_next.unsqueeze(0))

    if denoise_to_zero:
        x_next = get_denoised(net, x_next, t_n,
                              class_labels=class_labels,
                              condition=condition,
                              unconditional_condition=unconditional_condition)
        if return_inters:
            inters.append(x_next.unsqueeze(0))

    if return_inters:
        return torch.cat(inters, dim=0).to(latents.device)
    return x_next, x_list


# ----------------------------------------------------------------------------
# 3) 多步 + 需要 inner_steps
# (1) DPM-Solver++ (2M)：在内层子步上做 AB2（二阶多步），首子步用欧拉起步         2
# ----------------------------------------------------------------------------
def dpmpp_2m_with_inner(
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
    denoise_to_zero=False,
    return_inters=False,
    inner_steps=3,
    **kwargs
):
    t_steps = get_schedule(num_steps, sigma_min, sigma_max, device=latents.device,
                           schedule_type=schedule_type, schedule_rho=schedule_rho)
    x_next = latents * t_steps[0]
    inters = [x_next.unsqueeze(0)]
    x_list = [x_next]

    d_prev = None  # 上一子步的 d = (x - x0̂)/t

    for i_out, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x_cur = x_next
        t_s = get_schedule(inner_steps, t_next, t_cur, device=latents.device,
                           schedule_type='polynomial', schedule_rho=7)

        for j_in, (t_c, t_n) in enumerate(zip(t_s[:-1], t_s[1:])):
            x0_hat_c = get_denoised(net, x_cur, t_c,
                                    class_labels=class_labels,
                                    condition=condition,
                                    unconditional_condition=unconditional_condition)
            d_cur = (x_cur - x0_hat_c) / t_c
            h = (t_n - t_c)
            if d_prev is None:
                # Euler start
                x_cur = x_cur + h * d_cur
            else:
                # Adams–Bashforth 2: x_{n+1} = x_n + h * (1.5 d_n - 0.5 d_{n-1})
                x_cur = x_cur + h * (1.5 * d_cur - 0.5 * d_prev)
            d_prev = d_cur

        x_next = x_cur
        x_list.append(x_next)
        if return_inters:
            inters.append(x_cur.unsqueeze(0))

    if denoise_to_zero:
        x_next = get_denoised(net, x_next, t_next,
                              class_labels=class_labels,
                              condition=condition,
                              unconditional_condition=unconditional_condition)
        if return_inters:
            inters.append(x_next.unsqueeze(0))

    if return_inters:
        return torch.cat(inters, dim=0).to(latents.device)
    return x_next, x_list


# ----------------------------------------------------------------------------
# (2) DPM-Solver 二阶（“噪声”视角）：多步 + inner_steps（用 x0̂→epŝ 等价）
# ----------------------------------------------------------------------------
def dpm_solver2_with_inner_2m(
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
    denoise_to_zero=False,
    return_inters=False,
    inner_steps=3,
    **kwargs
):
    t_steps = get_schedule(num_steps, sigma_min, sigma_max, device=latents.device,
                           schedule_type=schedule_type, schedule_rho=schedule_rho)
    x_next = latents * t_steps[0]
    inters = [x_next.unsqueeze(0)]
    x_list = [x_next]

    d_prev = None

    for i_out, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x_cur = x_next
        t_s = get_schedule(inner_steps, t_next, t_cur, device=latents.device,
                           schedule_type='polynomial', schedule_rho=7)

        for j_in, (t_c, t_n) in enumerate(zip(t_s[:-1], t_s[1:])):
            x0_hat_c = get_denoised(net, x_cur, t_c,
                                    class_labels=class_labels,
                                    condition=condition,
                                    unconditional_condition=unconditional_condition)
            d_cur = (x_cur - x0_hat_c) / t_c  # ≡ epŝ (EDM)
            h = (t_n - t_c)
            if d_prev is None:
                x_cur = x_cur + h * d_cur
            else:
                x_cur = x_cur + h * (1.5 * d_cur - 0.5 * d_prev)
            d_prev = d_cur

        x_next = x_cur
        x_list.append(x_next)
        if return_inters:
            inters.append(x_cur.unsqueeze(0))

    if denoise_to_zero:
        x_next = get_denoised(net, x_next, t_next,
                              class_labels=class_labels,
                              condition=condition,
                              unconditional_condition=unconditional_condition)
        if return_inters:
            inters.append(x_next.unsqueeze(0))

    if return_inters:
        return torch.cat(inters, dim=0).to(latents.device)
    return x_next, x_list


# ----------------------------------------------------------------------------
# 4) 多步 + 不需要 inner_steps（直接在外层步上 2M）
# (1) DPM-Solver++ (2M)
# ----------------------------------------------------------------------------
def dpmpp_2m_singlestep(
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
    denoise_to_zero=False,
    return_inters=False,
    **kwargs
):
    t_steps = get_schedule(num_steps, sigma_min, sigma_max, device=latents.device,
                           schedule_type=schedule_type, schedule_rho=schedule_rho)
    x_next = latents * t_steps[0]
    inters = [x_next.unsqueeze(0)]
    x_list = [x_next]

    d_prev = None

    for i_out, (t_c, t_n) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x0_hat_c = get_denoised(net, x_next, t_c,
                                class_labels=class_labels,
                                condition=condition,
                                unconditional_condition=unconditional_condition)
        d_cur = (x_next - x0_hat_c) / t_c
        h = (t_n - t_c)
        if d_prev is None:
            x_next = x_next + h * d_cur
        else:
            x_next = x_next + h * (1.5 * d_cur - 0.5 * d_prev)
        d_prev = d_cur

        x_list.append(x_next)
        if return_inters:
            inters.append(x_next.unsqueeze(0))

    if denoise_to_zero:
        x_next = get_denoised(net, x_next, t_n,
                              class_labels=class_labels,
                              condition=condition,
                              unconditional_condition=unconditional_condition)
        if return_inters:
            inters.append(x_next.unsqueeze(0))

    if return_inters:
        return torch.cat(inters, dim=0).to(latents.device)
    return x_next, x_list


# ----------------------------------------------------------------------------
# (2) DPM-Solver 二阶：多步 + 无 inner_steps（x0̂→epŝ 等价）
# ----------------------------------------------------------------------------
def dpm_solver2_multistep(
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
    denoise_to_zero=False,
    return_inters=False,
    **kwargs
):
    t_steps = get_schedule(num_steps, sigma_min, sigma_max, device=latents.device,
                           schedule_type=schedule_type, schedule_rho=schedule_rho)
    x_next = latents * t_steps[0]
    inters = [x_next.unsqueeze(0)]
    x_list = [x_next]

    d_prev = None

    for i_out, (t_c, t_n) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x0_hat_c = get_denoised(net, x_next, t_c,
                                class_labels=class_labels,
                                condition=condition,
                                unconditional_condition=unconditional_condition)
        d_cur = (x_next - x0_hat_c) / t_c  # ≡ epŝ
        h = (t_n - t_c)
        if d_prev is None:
            x_next = x_next + h * d_cur
        else:
            x_next = x_next + h * (1.5 * d_cur - 0.5 * d_prev)
        d_prev = d_cur

        x_list.append(x_next)
        if return_inters:
            inters.append(x_next.unsqueeze(0))

    if denoise_to_zero:
        x_next = get_denoised(net, x_next, t_n,
                              class_labels=class_labels,
                              condition=condition,
                              unconditional_condition=unconditional_condition)
        if return_inters:
            inters.append(x_next.unsqueeze(0))

    if return_inters:
        return torch.cat(inters, dim=0).to(latents.device)
    return x_next, x_list
