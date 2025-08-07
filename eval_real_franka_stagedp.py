
"""
python eval_real_franka_rawdp.py -i  data/outputs/2025.07.14/17.54.12_train_diffusion_unet_image_real_image/checkpoints/epoch=0550-train_loss=0.001.ckpt -o data/test_box0601 --robot_ip 192.168.0.168

Usage:
(robodiff)$ python eval_real_robot.py -i <ckpt_path> -o <save_dir> --robot_ip <ip_of_ur5>
================ Human in control ==============
Recording control:
Click the opencv window (make sure it's in focus).
Press "w" to start evaluation (hand control over to policy).
Press "Q" to exit program.
================ Policy in control ==============
Make sure you can hit the robot hardware emergency-stop button quickly! 
Recording control:
Press "S" to stop evaluation and gain control back.
"""

# %%
import time
from multiprocessing.managers import SharedMemoryManager
import click
import cv2
import numpy as np
import torch
import dill
import hydra
import pathlib
import skvideo.io
from omegaconf import OmegaConf
import scipy.spatial.transform as st
# from diffusion_policy.real_world.real_env_franka            import RealEnvFranka as RealEnv
from diffusion_policy.real_world.real_env_franka_eval       import RealEnvFranka as RealEnv
from diffusion_policy.real_world.spacemouse_shared_memory   import Spacemouse
from diffusion_policy.common.precise_sleep                  import precise_wait
from diffusion_policy.real_world.real_inference_util        import (
    get_real_obs_resolution, 
    get_real_obs_dict,
    mask_get_real_obs_dict)
from diffusion_policy.common.pytorch_util                   import dict_apply
from diffusion_policy.workspace.base_workspace              import BaseWorkspace
from diffusion_policy.policy.base_image_policy              import BaseImagePolicy
from diffusion_policy.common.cv2_util                       import get_image_transform
from diffusion_policy.real_world.keystroke_counter          import ( KeystrokeCounter, Key, KeyCode )


OmegaConf.register_new_resolver("eval", eval, replace=True)

@click.command()
@click.option('--input', '-i', required=True, help='Path to checkpoint')
@click.option('--output', '-o', required=True, help='Directory to save recording')
@click.option('--robot_ip', '-ri', required=True, help="UR5's IP address e.g. 192.168.0.204")
@click.option('--match_dataset', '-m', default=None, help='Dataset used to overlay and adjust initial condition')
@click.option('--match_episode', '-me', default=None, type=int, help='Match specific episode from the match dataset')
@click.option('--vis_camera_idx', default=0, type=int, help="Which RealSense camera to visualize.")
@click.option('--init_joints', '-j', is_flag=True, default=False, help="Whether to initialize robot joint configuration in the beginning.")
@click.option('--steps_per_inference', '-si', default=6, type=int, help="Action horizon for inference.")
@click.option('--max_duration', '-md', default=60, help='Max duration for each epoch in seconds.')
@click.option('--frequency', '-f', default=10, type=float, help="Control frequency in Hz.")
@click.option('--command_latency', '-cl', default=0.01, type=float, help="Latency between receiving SapceMouse command to executing on Robot in Sec.")
def main(input, output, robot_ip, match_dataset, match_episode,
    vis_camera_idx, init_joints, 
    steps_per_inference, max_duration,
    frequency, command_latency):
    # load match_dataset
    match_camera_idx = 0
    episode_first_frame_map = dict()
    if match_dataset is not None:
        match_dir = pathlib.Path(match_dataset)
        match_video_dir = match_dir.joinpath('videos')
        for vid_dir in match_video_dir.glob("*/"):
            episode_idx = int(vid_dir.stem)
            match_video_path = vid_dir.joinpath(f'{match_camera_idx}.mp4')
            if match_video_path.exists():
                frames = skvideo.io.vread(
                    str(match_video_path), num_frames=1)
                episode_first_frame_map[episode_idx] = frames[0]
    print(f"Loaded initial frame for {len(episode_first_frame_map)} episodes")
    
    # load checkpoint
    ckpt_path = input
    payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill)
    cfg = payload['cfg']

    # === 添加以下代码 ===
    print("="*50)
    # print(f"Workspace _target_: {cfg.defaults._target_}")
    print(f"Workspace _target_: {cfg._target_}")
    print(f"Policy _target_: {cfg.policy._target_}")
    print(f"Obs_encoder _target_: {cfg.policy.obs_encoder._target_}")
    print("="*50)
    # === 添加结束 ===

    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    # hacks for method-specific setup.
    action_offset = 0
    delta_action = False
    # delta_action = True
    if 'diffusion' in cfg.name:
        torch.set_grad_enabled(False)  # 全局禁用梯度计算
        policy = workspace.model
        if cfg.training.use_ema:
            policy = workspace.ema_model

        # 移动到设备
        device = torch.device('cuda')
        policy.eval().to(device)

        # set inference params
        policy.num_inference_steps = 16 # DDIM inference iterations
        # policy.n_action_steps = policy.horizon - policy.n_obs_steps + 1
    else:
        print("Warning: Using non-diffusion policy, please check the configuration.")

    # setup experiment
    dt = 1/frequency


    return_begin =False
    # init_position = np.array([0.3551,-0.0026,0.6935,-2.8691,1.2213,0.0139])
    # init_position = np.array([0.566, -0.2,0.33,-1.44,-0.659,-0.593])
    init_position = np.array([0.48, 0.0, 0.7,-2.8691,1.2213,0.0139])
    epsilon = 0.02                                      # 位置误差容忍阈值 (单位：米)0.004
    step_size = 0.02                                    # 调整步长0.01
    epsilon_rot = 0.0175                                # 角度死区阈值 (约1度，单位：弧度)
    max_step_angle = 0.07                               # 最大单步调整角度 (约5度，单位：弧度)

    # stage预测用
    current_stage = 0  # 当前阶段值
    # stage_buffer = []   # 历史阶段值记录
    stage_history = []  # 历史阶段值记录
    # 在人类控制循环和策略控制循环之前添加
    stage_buffer = [current_stage] * cfg.n_obs_steps  # 初始化缓冲区

    # # 确保shape_meta包含stage配置
    # if 'stage' not in cfg.task.shape_meta['obs']:
    #     print("Warning: 'stage' not found in shape_meta['obs'], initializing with default values.")
    #     cfg.task.shape_meta['obs']['stage'] = [cfg.n_obs_steps, 1]


    obs_res = get_real_obs_resolution(cfg.task.shape_meta)
    n_obs_steps = cfg.n_obs_steps
    print("n_obs_steps: ", n_obs_steps)
    print("steps_per_inference:", steps_per_inference)
    print("action_offset:", action_offset)
    gripper_state_exe = 0.08
    with SharedMemoryManager() as shm_manager:
        with KeystrokeCounter() as key_counter, \
            Spacemouse(shm_manager=shm_manager) as sm, \
            RealEnv(
                    output_dir          = output, 
                    robot_ip            = robot_ip, 
                    frequency           = frequency,
                    n_obs_steps         = n_obs_steps,
                    obs_image_resolution= obs_res,
                    obs_float32         = True,
                    init_joints         = init_joints,
                    enable_multi_cam_vis= True,
                    record_raw_video    = True,
                    thread_per_video    = 3,            # 每个相机视图的线程数用于视频录制(H.264)
                    video_crf           = 21,           # 视频录制质量，越低越好（但速度较慢）
                    shm_manager         = shm_manager,
                    enable_sam2         = False,        # 是否启用SAM2 True
                    # enable_sam2         = False,        # 是否启用SAM2
            ) as env:
            cv2.setNumThreads(1)

            env_enable_sam2 = env.enable_sam2
            policy_control = False

            return_begin = False
            # init_position = np.array([0.2,0.0,0.85,-2.743,1.127,-0.485])
            init_position = np.array([0.48, 0.0, 0.7,-2.8691,1.2213,0.0139])

            epsilon = 0.01                                      # 位置误差容忍阈值 (单位：米)0.004
            step_size = 0.015                                    # 调整步长0.01
            epsilon_rot = 0.02                                # 角度死区阈值 (约1度，单位：弧度)
            max_step_angle = 0.03                               # 最大单步调整角度 (约5度，单位：弧度)
            pos_arrived = [False, False, False, False]          # 归位标志符
            stage_gripper_map = {
                0: 0.0,    # 运动
                1: 0.0,   # 抓紫圆柱+运动
                2: 0.0,    # 放紫圆柱+返回
            }# 卷尺0.05


            print("Waiting for realsense")
            time.sleep(1.0)

            print("Warming up policy inference")
            
            # 测试模型
            state = env.get_robot_state()                       # 获取机器人状态
            target_pose = state['ActualTCPPose']                # 获取目标姿态
            gripper_state = state['ActualGripperstate']         # 获取夹爪状态
            print('Initial pose:', target_pose)                 # 打印初始姿态
            print('Initial gripper state:', gripper_state)      # 打印初始夹爪状态

            obs = env.get_obs()
            # 更新阶段历史记录
            if stage_history:
                stage_history = stage_history[-n_obs_steps+1:]
            stage_history.append(current_stage)
            
            # 确保有足够的阶段值
            if len(stage_history) < n_obs_steps:
                stage_arr = np.array([stage_history[0]] * (n_obs_steps - len(stage_history)) + stage_history)
            else:
                stage_arr = np.array(stage_history)
            # 转换为适当的形状注入观测数据
            obs['stage'] = stage_arr.astype(np.float32).reshape(-n_obs_steps, 1)# n_obs_steps应该是2
            # print("obs:", obs['stage'])
            print("obs:", obs)

            print("env_enable_sam2 == False, 全局条件rgb")
            with torch.no_grad():
                policy.reset()
                obs_dict_np = get_real_obs_dict(
                    env_obs=obs, shape_meta=cfg.task.shape_meta)
                obs_dict = dict_apply(obs_dict_np, 
                    lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                result = policy.predict_action(obs_dict)
                action = result['action'][0].detach().to('cpu').numpy()
                del result

            print('Ready!')
            stage = 0
            while True:
                # ========= human control loop ==========
                print("Human in control!")
                state = env.get_robot_state()                       # 获取机器人状态
                target_pose = state['ActualTCPPose']                # 获取目标姿态
                gripper_state = state['ActualGripperstate']         # 获取夹爪状态
                
                t_start = time.monotonic()                          # 获取当前时间
                iter_idx = 0                                        # 迭代次数 
                while True:
                    # calculate timing
                    t_cycle_end = t_start + (iter_idx + 1) * dt
                    t_sample = t_cycle_end - command_latency
                    t_command_target = t_cycle_end + dt

                    obs = env.get_obs()                # visualize
                    if not obs or 'camera_0' not in obs:
                        print("⚠️ 获取观测失败！检查环境连接")
                        time.sleep(0.1)  # 短暂等待避免忙循环
                        continue
                    # 1：处理RGB图像数据（480x640x3）
                    vis_img = obs[f'camera_{vis_camera_idx}'][-1,:,:,::-1].copy()   # 获取可视化图像并转换颜色空间

                    # cv2.putText(vis_img, f'Stage: {current_stage}', (10, 30), 
                    # cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                    
                    cv2.imshow('RGB Image', vis_img) 
                        
                    cv2.pollKey()                                           # OpenCV键盘事件处理
                    press_events = key_counter.get_press_events()           # 获取按键事件
                    for key_stroke in press_events:                         # 遍历按键事件

                        if key_stroke == KeyCode(char='q'):
                            env.end_episode()
                            key_counter.clear()                             # 清除按键计数器
                            cv2.destroyAllWindows()                         # 关闭所有OpenCV窗口 
                            exit(0)   
                        elif key_stroke == KeyCode(char='w'):               # 结束人类控制循环，切换策略控制循环
                            stage = 0
                            # env.start_episode(t_start + (iter_idx + 2) * dt - time.monotonic() + time.time()) # 开始新集
                            key_counter.clear()                             # 清除按键计数器
                            return_begin = False
                            print('Recording!')                             # 打印录制消息
                            cv2.waitKey(1)                                  # 强制刷新事件队列
                            policy_control = True
                            break
                        elif key_stroke == KeyCode(char='s'):
                            env.end_episode()                               # 结束当前集
                            key_counter.clear()                             # 清除按键计数器
                            return_begin = False
                            print('Stopped.')                               # 打印停止消息
                        elif key_stroke == Key.backspace:
                            if click.confirm('Are you sure to drop an episode?'): # 确认删除
                                env.drop_episode()                          # 删除集
                                key_counter.clear()                         # 清除按键计数器
                        elif key_stroke == KeyCode(char='r'):
                            return_begin = True
                            # 保证爪子还原
                            key_counter.clear()
                            print("r被按下, Return to initial pose triggered")
                        elif key_stroke == Key.space:  # 同样响应空格
                            current_stage = (current_stage + 1) % 5         # 0-4循环
                            print(f'策略控制中更新阶段={current_stage}')
                        # 检测数字按键 (0-9),解决Key对象没有char属性的问题
                        elif isinstance(key_stroke, KeyCode) and key_stroke.char and key_stroke.char in '0123456789':
                            current_stage = int(key_stroke.char)
                            print(f'数字键 {key_stroke.char} 被按下，设置 stage={current_stage}')
                        else:
                            print(f"按键 {key_stroke} 未处理或不在预设范围内，stage={current_stage}")
                    if current_stage in stage_gripper_map:
                        gripper_state_exe = stage_gripper_map[current_stage]
                    else:
                        gripper_state_exe = (current_stage + 1) % 2 * 0.08 # 默认行为
                        print(f"警告: stage={current_stage}不在预设范围，自动计算夹爪状态为 {gripper_state_exe:.2f}")
                    gripper_state = obs['robot_gripper'][-1][0]             # 获取当前夹爪状态
                    precise_wait(t_sample)
                    
                    if return_begin:
                        # ===== 位置控制 =====
                        for i in range(3):
                            # 计算当前位置与目标的差值
                            error = init_position[i] - target_pose[i]
                            abs_error = abs(error)
                            if abs_error > epsilon:                         # 差值超过阈值时，按比例调整步长
                                adjustment = np.sign(error) * min(step_size, abs_error)
                                target_pose[i] += adjustment
                            else:
                                pos_arrived[i] = abs_error < epsilon * 0.1
                                target_pose[i] = init_position[i]            # 差值小于阈值时，直接设为目标值
                        # ===== 旋转控制 =====
                        current_rot = st.Rotation.from_rotvec(target_pose[3:6])
                        target_rot = st.Rotation.from_rotvec(init_position[3:6])
                        rel_rot = target_rot * current_rot.inv()
                        rel_angle = rel_rot.magnitude()
                        if rel_angle > epsilon_rot:
                            step_angle = min(rel_angle, max_step_angle)
                            adjust_rot = st.Rotation.from_rotvec(
                                rel_rot.as_rotvec() * (step_angle / rel_angle)
                            )
                            new_rot = adjust_rot * current_rot
                            target_pose[3:6] = new_rot.as_rotvec()
                        else:
                            # 检查旋转到位状态（角度差小于阈值的10%）
                            pos_arrived[3] = rel_angle < epsilon_rot * 0.1
                            target_pose[3:6] = init_position[3:6]
                        
                        gripper_state_exe = 0.08                                        # 夹爪状态
                        # ===== 旋转控制结束 =====
                        if all(pos_arrived):                                            # 有滞后
                            print("归位成功,当前关节位置为",obs['robot_joint'][-1],"释放控制权")
                            pos_arrived = [False, False, False, False]                  # 归位标志符
                            return_begin = False
                    
                    else: # ===== 空间鼠标控制 =====
                        sm_state = sm.get_motion_state_transformed()                    # 获取SpaceMouse的运动状态
                        dpos = sm_state[:3] * (env.max_pos_speed / frequency)           # 计算位置增量
                        drot_xyz = sm_state[3:] * (env.max_rot_speed / frequency)       # 计算旋转增量
                        drot = st.Rotation.from_euler('xyz', drot_xyz)                  # 计算旋转
                        target_pose[:3] += dpos                                         # 更新目标位置
                        target_pose[3:] = (drot * st.Rotation.from_rotvec(
                            target_pose[3:])).as_rotvec()                               # 更新目标旋转

                    # 位置拼接
                    actions =np.zeros(7)
                    actions[:6] = target_pose
                    actions[6] = gripper_state_exe

                    #  命令执行
                    env.exec_actions(                                                   # 执行动作
                        actions=[actions], 
                        timestamps=[t_command_target-time.monotonic()+time.time()],
                        # stages=[[stage]],
                        )
                    precise_wait(t_cycle_end)                                           # 精确等待循环结束时间
                    iter_idx += 1                                                       # 增加迭代索引
                    
                    
                    if policy_control:                                                  # 如果策略控制
                        break
                # ========== policy control loop ==============
                try:
                    # 添加的初始测试
                    print("Entered Policy Control Loop!")
                    current_stage = 0

                    gripper_state   = state['ActualGripperstate']       # 获取夹爪状态
                    target_pose     = state['ActualTCPPose']            # 获取目标姿态
                    print("init_position:",init_position)
                    print("gripper_state",gripper_state,"target_pose",target_pose)
                    actions =np.zeros(7)
                    actions[:6]     = target_pose
                    actions[6]      = 0.08                              # 夹爪状态
                    print("gripper_state_exe",actions[6])
                    env.exec_actions(
                        actions=[actions],
                        timestamps=[1.0]                # start_delay = 1.0
                    )

                    # start episode
                    policy.reset()
                    start_delay = 1.0
                    eval_t_start = time.time() + start_delay
                    t_start = time.monotonic() + start_delay
                    env.start_episode(eval_t_start)
                    # reduces overall latency
                    frame_latency = 1/30
                    precise_wait(eval_t_start - frame_latency, time_func=time.time)
                    print("Started!")
                    iter_idx = 0
                    perv_target_pose = None
                    while policy_control:
                        t_cycle_end = t_start + (iter_idx + steps_per_inference) * dt

                        obs = env.get_obs()

                        # === 添加阶段信息到观测数据 ===
                        # 更新阶段历史记录
                        if stage_history:
                            stage_history = stage_history[-n_obs_steps+1:]
                        stage_history.append(current_stage)
                        
                        # 确保有足够的阶段值
                        if len(stage_history) < n_obs_steps:
                            stage_arr = np.array([stage_history[0]] * (n_obs_steps - len(stage_history)) + stage_history)
                        else:
                            stage_arr = np.array(stage_history)
                        # 转换为适当的形状注入观测数据
                        obs['stage'] = stage_arr.astype(np.float32).reshape(-n_obs_steps, 1)# n_obs_steps应该是2
                        # 
                        # obs['stage'] = stage_array
                        print("obs:", obs['stage'])

                        # 以下为调试用
                        gripper_state = obs['robot_gripper'][-1]
                        obs_timestamps = obs['timestamp']
                        # 推理run inference
                        if env_enable_sam2:
                            print("error")
                            with torch.no_grad():
                                s = time.time()
                                obs_dict_np = mask_get_real_obs_dict(
                                    env_obs=obs, 
                                    shape_meta=cfg.task.shape_meta)
                        else:
                            with torch.no_grad():
                                s = time.time()
                                obs_dict_np = get_real_obs_dict(
                                    env_obs=obs, 
                                    shape_meta=cfg.task.shape_meta)
                        
                        obs_dict = dict_apply(obs_dict_np, 
                            lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                                
                        result = policy.predict_action(obs_dict)
                        # this action starts from the first obs step
                        action = result['action'][0].detach().to('cpu').numpy()

                        # 转换动作convert policy action to env actions
                        if delta_action:
                            # assert len(action) == 1,"增量模式需要1维动作"
                            if perv_target_pose is None:
                                perv_target_pose = obs['robot_eef_pose'][-1]
                            this_target_pose = perv_target_pose.copy()
                            this_target_pose += action[-1]
                            perv_target_pose = this_target_pose
                            this_target_poses = np.expand_dims(this_target_pose, axis=0)
                        else:                                           # 7替换掉len(target_pose)
                            assert action.shape[1] == 7, "绝对模式需要7维动作(x,y,z,rx,ry,rz,gripper)"
                            # assert action.shape[1] == 8, "绝对模式需要8维动作(x,y,z,rx,ry,rz,gripper,stage)"
                            # print("action[7]:", action[7])
                            # action = action[:, :7]  # 只取前7个元素
                            # 使用更安全的初始化方式
                            this_target_poses = np.empty_like(action)
                            # 方法0：直接赋值执行
                            this_target_poses = np.zeros((len(action), len(actions)), dtype=np.float64)
                            this_target_poses[:] = action
                        # deal with timing
                        # the same step actions are always the target for
                        action_timestamps = (np.arange(len(action), dtype=np.float64) + action_offset
                            ) * dt + obs_timestamps[-1]
                        action_exec_latency = 0.01
                        curr_time = time.time()
                        is_new = action_timestamps > (curr_time + action_exec_latency)
                        if np.sum(is_new) == 0:
                            # exceeded time budget, still do something
                            this_target_poses = this_target_poses[[-1]]
                            # schedule on next available step
                            next_step_idx = int(np.ceil((curr_time - eval_t_start) / dt))
                            action_timestamp = eval_t_start + (next_step_idx) * dt
                            print('Over budget', action_timestamp - curr_time)
                            action_timestamps = np.array([action_timestamp])
                        else:
                            this_target_poses = this_target_poses[is_new]
                            action_timestamps = action_timestamps[is_new]

                        env.exec_actions(
                            actions=this_target_poses,
                            timestamps=action_timestamps
                        )

                        # 可视化visualize
                        # 1：处理RGB图像数据（480x640x3）
                        vis_img = obs[f'camera_{vis_camera_idx}'][-1,:,:,::-1].copy()   # 获取可视化图像并转换颜色空间

                        # cv2.putText(vis_img, f'Stage: {current_stage}', (10, 30), 
                        # cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)


                        cv2.imshow('RGB Image', vis_img) 
                        cv2.pollKey() 

                        # key_stroke = cv2.pollKey()
                        # if key_stroke == ord('s'):
                        #     env.end_episode()
                        #     print('Stopped.')
                        #     break

                        press_events = key_counter.get_press_events()           # 获取按键事件
                        for key_stroke in press_events:                         # 遍历按键事件
                            if key_stroke == KeyCode(char='q'):
                                env.end_episode()
                                key_counter.clear()                             # 清除按键计数器
                                cv2.destroyAllWindows()                         # 关闭所有OpenCV窗口 
                                exit(0)   
                            elif key_stroke == KeyCode(char='s'):
                                env.end_episode()                               # 结束当前集
                                key_counter.clear()                             # 清除按键计数器
                                policy_control = False   # 关键！切换回人类控制
                                return_begin = False
                                print('Stopped.')                               # 打印停止消息
                            # 检测数字按键 (0-9),解决Key对象没有char属性的问题
                            elif isinstance(key_stroke, KeyCode) and key_stroke.char and key_stroke.char in '0123456789':
                                current_stage = int(key_stroke.char)
                                print(f'数字键 {key_stroke.char} 被按下，设置 stage={current_stage}')
                            elif key_stroke == Key.space:  # 同样响应空格
                                current_stage = (current_stage + 1) % 5         # 0-4循环
                                print(f'策略控制中更新阶段={current_stage}')
                            else:
                                print(f"按键 {key_stroke} 未处理或不在预设范围内，stage={stage}")

                        # auto termination
                        terminate = False
                        if time.monotonic() - t_start > max_duration:
                            terminate = True
                            print('Terminated by the timeout!')

                        if terminate:
                            env.end_episode()
                            break

                        # wait for execution
                        precise_wait(t_cycle_end - frame_latency)
                        iter_idx += steps_per_inference

                except KeyboardInterrupt:
                    print("Interrupted!")
                    # stop robot.
                    env.end_episode()
                
                print("Stopped.")



# %%
if __name__ == '__main__':
    main()
