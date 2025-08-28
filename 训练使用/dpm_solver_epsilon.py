import torch

# =========================
# 1) 单步 + 需要 inner_steps
#    DPM-Solver 二阶（噪声预测 2S / RK2-r）
# =========================
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
    # Time step discretization.
    t_steps = get_schedule(num_steps, sigma_min, sigma_max, device=latents.device,
                           schedule_type=schedule_type, schedule_rho=schedule_rho)
    # Main sampling loop.
    x_next = latents * t_steps[0]
    inters = [x_next.unsqueeze(0)]
    x_list = [x_next]

    for i_out, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):  # 0, ..., N-1
        x_cur = x_next
        # Inner schedule
        t_s = get_schedule(inner_steps, t_next, t_cur, device=latents.device,
                           schedule_type='polynomial', schedule_rho=7)
        for j_in, (t_c, t_n) in enumerate(zip(t_s[:-1], t_s[1:])):
            # 起点斜率（eps）；AFS 仅首子步启用
            use_afs = (afs and j_in == 0)
            if use_afs:
                d_cur = x_cur / ((1 + t_c**2).sqrt())          # 解析近似（与原代码一致）
            else:
                # 直接从网络取 eps
                d_cur = net(x_cur, t_c,
                            class_labels=class_labels,
                            condition=condition,
                            unconditional_condition=unconditional_condition)

            # 阶段点（几何插值），预测到阶段点
            t_mid = (t_n ** r) * (t_c ** (1 - r))
            x_pred = x_cur + (t_mid - t_c) * d_cur

            # 阶段点斜率（eps）
            d_mid = net(x_pred, t_mid,
                        class_labels=class_labels,
                        condition=condition,
                        unconditional_condition=unconditional_condition)

            # RK2-r 合成到 t_n
            x_cur = x_cur + (t_n - t_c) * ((1 / (2 * r)) * d_mid + (1 - 1 / (2 * r)) * d_cur)

        x_next = x_cur
        x_list.append(x_next)
        if return_inters:
            inters.append(x_cur.unsqueeze(0))

    # 可选：denoise to zero（把 eps 转为 x0）
    if denoise_to_zero:
        eps_final = net(x_next, t_next,
                        class_labels=class_labels,
                        condition=condition,
                        unconditional_condition=unconditional_condition)
        x_next = x_next - t_next * eps_final
        if return_inters:
            inters.append(x_next.unsqueeze(0))

    if return_inters:
        return torch.cat(inters, dim=0).to(latents.device)
    return x_next, x_list


# =========================
# 2) 单步 + 不需要 inner_steps
#    DPM-Solver 二阶（噪声预测 2S / RK2-r）
# =========================
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
    # Time step discretization.
    t_steps = get_schedule(num_steps, sigma_min, sigma_max, device=latents.device,
                           schedule_type=schedule_type, schedule_rho=schedule_rho)
    # Main sampling loop.
    x_next = latents * t_steps[0]
    inters = [x_next.unsqueeze(0)]
    x_list = [x_next]

    for i_out, (t_c, t_n) in enumerate(zip(t_steps[:-1], t_steps[1:])):  # 0, ..., N-1
        # 起点斜率（eps）
        d_cur = net(x_next, t_c,
                    class_labels=class_labels,
                    condition=condition,
                    unconditional_condition=unconditional_condition)

        # 阶段点与预测
        t_mid = (t_n ** r) * (t_c ** (1 - r))
        x_pred = x_next + (t_mid - t_c) * d_cur

        # 阶段点斜率（eps）
        d_mid = net(x_pred, t_mid,
                    class_labels=class_labels,
                    condition=condition,
                    unconditional_condition=unconditional_condition)

        # RK2-r 合成
        x_next = x_next + (t_n - t_c) * ((1 / (2 * r)) * d_mid + (1 - 1 / (2 * r)) * d_cur)

        x_list.append(x_next)
        if return_inters:
            inters.append(x_next.unsqueeze(0))

    if denoise_to_zero:
        eps_final = net(x_next, t_n,
                        class_labels=class_labels,
                        condition=condition,
                        unconditional_condition=unconditional_condition)
        x_next = x_next - t_n * eps_final
        if return_inters:
            inters.append(x_next.unsqueeze(0))

    if return_inters:
        return torch.cat(inters, dim=0).to(latents.device)
    return x_next, x_list


# =========================
# 3) 多步 + 需要 inner_steps
#    DPM-Solver 二阶（噪声预测 2M / AB2），首子步欧拉起步
# =========================
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
    # Time step discretization.
    t_steps = get_schedule(num_steps, sigma_min, sigma_max, device=latents.device,
                           schedule_type=schedule_type, schedule_rho=schedule_rho)
    # Main sampling loop.
    x_next = latents * t_steps[0]
    inters = [x_next.unsqueeze(0)]
    x_list = [x_next]

    d_prev = None  # 保存上一子步的 eps

    for i_out, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):  # 0, ..., N-1
        x_cur = x_next
        # Inner schedule
        t_s = get_schedule(inner_steps, t_next, t_cur, device=latents.device,
                           schedule_type='polynomial', schedule_rho=7)

        for j_in, (t_c, t_n) in enumerate(zip(t_s[:-1], t_s[1:])):
            # 当前子步斜率（eps）
            d_cur = net(x_cur, t_c,
                        class_labels=class_labels,
                        condition=condition,
                        unconditional_condition=unconditional_condition)
            h = (t_n - t_c)
            if d_prev is None:
                # Euler 起步
                x_cur = x_cur + h * d_cur
            else:
                # AB2：x_{n+1} = x_n + h * (1.5 d_n - 0.5 d_{n-1})
                x_cur = x_cur + h * (1.5 * d_cur - 0.5 * d_prev)
            d_prev = d_cur

        x_next = x_cur
        x_list.append(x_next)
        if return_inters:
            inters.append(x_cur.unsqueeze(0))

    if denoise_to_zero:
        eps_final = net(x_next, t_next,
                        class_labels=class_labels,
                        condition=condition,
                        unconditional_condition=unconditional_condition)
        x_next = x_next - t_next * eps_final
        if return_inters:
            inters.append(x_next.unsqueeze(0))

    if return_inters:
        return torch.cat(inters, dim=0).to(latents.device)
    return x_next, x_list


# =========================
# 4) 多步 + 不需要 inner_steps
#    DPM-Solver 二阶（噪声预测 2M / AB2），直接在外层步
# =========================
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
    # Time step discretization.
    t_steps = get_schedule(num_steps, sigma_min, sigma_max, device=latents.device,
                           schedule_type=schedule_type, schedule_rho=schedule_rho)
    # Main sampling loop.
    x_next = latents * t_steps[0]
    inters = [x_next.unsqueeze(0)]
    x_list = [x_next]

    d_prev = None

    for i_out, (t_c, t_n) in enumerate(zip(t_steps[:-1], t_steps[1:])):  # 0, ..., N-1
        # 当前外层步斜率（eps）
        d_cur = net(x_next, t_c,
                    class_labels=class_labels,
                    condition=condition,
                    unconditional_condition=unconditional_condition)
        h = (t_n - t_c)
        if d_prev is None:
            x_next = x_next + h * d_cur            # Euler 起步
        else:
            x_next = x_next + h * (1.5 * d_cur - 0.5 * d_prev)  # AB2
        d_prev = d_cur

        x_list.append(x_next)
        if return_inters:
            inters.append(x_next.unsqueeze(0))

    if denoise_to_zero:
        eps_final = net(x_next, t_n,
                        class_labels=class_labels,
                        condition=condition,
                        unconditional_condition=unconditional_condition)
        x_next = x_next - t_n * eps_final
        if return_inters:
            inters.append(x_next.unsqueeze(0))

    if return_inters:
        return torch.cat(inters, dim=0).to(latents.device)
    return x_next, x_list
