
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
from diffusion_policy.real_world.real_env_franka_pro_eval       import RealEnvFranka as RealEnv
from diffusion_policy.real_world.spacemouse_shared_memory   import Spacemouse
from diffusion_policy.common.precise_sleep                  import precise_wait
from diffusion_policy.real_world.real_inference_util        import (
    get_real_obs_resolution, 
    get_real_obs_dict,
    mask_get_real_obs_dict,
    get_real_umi_obs_dict,
    get_real_umi_action
    )
from diffusion_policy.common.pytorch_util                   import dict_apply
from diffusion_policy.workspace.base_workspace              import BaseWorkspace
from diffusion_policy.policy.base_image_policy              import BaseImagePolicy
from diffusion_policy.common.cv2_util                       import get_image_transform
from diffusion_policy.real_world.keystroke_counter          import ( KeystrokeCounter, Key, KeyCode )

import yaml
from diffusion_policy.common.pose_util import pose_to_mat, mat_to_pose
import os
from scipy.spatial.transform import Rotation as R


OmegaConf.register_new_resolver("eval", eval, replace=True)

def solve_table_collision(ee_pose, gripper_width, height_threshold):
    finger_thickness = 25.5 / 1000
    keypoints = list()
    for dx in [-1, 1]:
        for dy in [-1, 1]:
            keypoints.append((dx * gripper_width / 2, dy * finger_thickness / 2, 0))
    keypoints = np.asarray(keypoints)
    rot_mat = st.Rotation.from_rotvec(ee_pose[3:6]).as_matrix()
    transformed_keypoints = np.transpose(rot_mat @ np.transpose(keypoints)) + ee_pose[:3]
    delta = max(height_threshold - np.min(transformed_keypoints[:, 2]), 0)
    ee_pose[2] += delta

def solve_sphere_collision(ee_poses, robots_config):
    num_robot = len(robots_config)
    this_that_mat = np.identity(4)
    this_that_mat[:3, 3] = np.array([0, 0.89, 0]) # TODO: very hacky now!!!!

    for this_robot_idx in range(num_robot):
        for that_robot_idx in range(this_robot_idx + 1, num_robot):
            this_ee_mat = pose_to_mat(ee_poses[this_robot_idx][:6])
            this_sphere_mat_local = np.identity(4)
            this_sphere_mat_local[:3, 3] = np.asarray(robots_config[this_robot_idx]['sphere_center'])
            this_sphere_mat_global = this_ee_mat @ this_sphere_mat_local
            this_sphere_center = this_sphere_mat_global[:3, 3]

            that_ee_mat = pose_to_mat(ee_poses[that_robot_idx][:6])
            that_sphere_mat_local = np.identity(4)
            that_sphere_mat_local[:3, 3] = np.asarray(robots_config[that_robot_idx]['sphere_center'])
            that_sphere_mat_global = this_that_mat @ that_ee_mat @ that_sphere_mat_local
            that_sphere_center = that_sphere_mat_global[:3, 3]

            distance = np.linalg.norm(that_sphere_center - this_sphere_center)
            threshold = robots_config[this_robot_idx]['sphere_radius'] + robots_config[that_robot_idx]['sphere_radius']
            # print(that_sphere_center, this_sphere_center)
            if distance < threshold:
                print('avoid collision between two arms')
                half_delta = (threshold - distance) / 2
                normal = (that_sphere_center - this_sphere_center) / distance
                this_sphere_mat_global[:3, 3] -= half_delta * normal
                that_sphere_mat_global[:3, 3] += half_delta * normal
                
                ee_poses[this_robot_idx][:6] = mat_to_pose(this_sphere_mat_global @ np.linalg.inv(this_sphere_mat_local))
                ee_poses[that_robot_idx][:6] = mat_to_pose(np.linalg.inv(this_that_mat) @ that_sphere_mat_global @ np.linalg.inv(that_sphere_mat_local))


@click.command()
@click.option('--input', '-i', required=True, help='Path to checkpoint')
@click.option('--output', '-o', required=True, help='Directory to save recording')
@click.option('--robot_config', '-rc', required=True, help='Path to robot_config yaml file')
@click.option('--robot_ip', '-ri', required=True, help="UR5's IP address e.g. 192.168.0.204")
@click.option('--match_dataset', '-m', default=None, help='Dataset used to overlay and adjust initial condition')
@click.option('--match_episode', '-me', default=None, type=int, help='Match specific episode from the match dataset')
@click.option('--vis_camera_idx', default=0, type=int, help="Which RealSense camera to visualize.")
@click.option('--init_joints', '-j', is_flag=True, default=False, help="Whether to initialize robot joint configuration in the beginning.")
@click.option('--steps_per_inference', '-si', default=6, type=int, help="Action horizon for inference.")
@click.option('--max_duration', '-md', default=60, help='Max duration for each epoch in seconds.')
@click.option('--frequency', '-f', default=10, type=float, help="Control frequency in Hz.")
@click.option('--command_latency', '-cl', default=0.01, type=float, help="Latency between receiving SapceMouse command to executing on Robot in Sec.")
def main(input, 
         output, 
         robot_config, 
         robot_ip, 
         match_dataset, 
         match_episode,
         vis_camera_idx, 
         init_joints, 
         steps_per_inference, 
         max_duration,
         frequency, 
         command_latency
         ):

    # load robot config file
    robot_config_data = yaml.safe_load(open(os.path.expanduser(robot_config), 'r'))
    
    # load left-right robot relative transform
    tx_left_right = np.array(robot_config_data['tx_left_right'])
    tx_robot1_robot0 = tx_left_right

    robots_config = robot_config_data['robots']
    grippers_config = robot_config_data['grippers']



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
    cls = hydra.utils.get_class(cfg._target_)

    print(f"从 {ckpt_path} 加载 {cls} ")

    workspace = cls(cfg)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    # 添加详细的维度检查
    print("\n===== 模型配置检查 =====")
    print(f"模型名称: {cfg.name}")
    print(f"使用 EMA: {cfg.training.use_ema}")
    print(f"观察步数 (n_obs_steps): {cfg.task.img_obs_horizon}")
    print(f"姿态表示 - 观测姿态表示: {cfg.task.pose_repr.obs_pose_repr}")
    print(f"姿态表示 - 动作姿态表示: {cfg.task.pose_repr.action_pose_repr}")

    # hacks for method-specific setup.
    action_offset = 0
    delta_action = False
    # delta_action = True
    if 'diffusion' in cfg.name:
        policy = workspace.model
        if cfg.training.use_ema:
            policy = workspace.ema_model
        
        Trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        print(f'模型可训练参数 Trainable params: {Trainable_params/ 1e6}M') # 171.27361M

        # 移动到设备
        device = torch.device('cuda')
        policy.eval().to(device)

        # 从配置中获取预测范围和观察步数
        n_obs_steps = cfg.task.img_obs_horizon

        # 设置推理参数
        policy.num_inference_steps = 16  # DDIM inference iterations
        obs_pose_rep = cfg.task.pose_repr.obs_pose_repr
        action_pose_repr = cfg.task.pose_repr.action_pose_repr

    else:
        print("Warning: Using non-diffusion policy, please check the configuration.")

    # setup experiment
    dt = 1/frequency

    obs_res = get_real_obs_resolution(cfg.task.shape_meta)
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
            policy_control = False

            return_begin = False
            # init_position = np.array([0.566, -0.2,0.33,-1.44,-0.659,-0.593])
            init_position = np.array([0.50, 0.033,0.562,2.88,-1.16,-0.104])# 0.458
            epsilon = 0.01                                      # 位置误差容忍阈值 (单位：米)0.004
            step_size = 0.015                                    # 调整步长0.01
            epsilon_rot = 0.02                                # 角度死区阈值 (约1度，单位：弧度)
            max_step_angle = 0.03                               # 最大单步调整角度 (约5度，单位：弧度)
            pos_arrived = [False, False, False, False]          # 归位标志符
            stage_gripper_map = {
                0: 0.08,    # 运动
                1: 0.035,   # 抓紫圆柱+运动
                2: 0.08,    # 放紫圆柱+返回
                3: 0.08,    # 运动
                4: 0.048,   # 抓绿方块+运动
                5: 0.08,    # 放绿方块+返回
                6: 0.08,    # 运动
                7: 0.035,    # 抓黄圆柱+运动
                8: 0.08,    # 放黄圆柱+返回
                9: 0.08,    # 运动
            }# 卷尺0.05


            print("Waiting for realsense")
            time.sleep(1.0)

            print("Warming up policy inference")
            
            # 测试模型
            state = env.get_robot_state()                       # 获取机器人状态
            target_pose = state['ActualTCPPose']                # 获取目标姿态
            gripper_state = state['ActualGripperstate']         # 获取夹爪状态
            print(f"ActualTCPPose 形状: {state['ActualTCPPose'].shape}", 'Initial pose:', target_pose)
            print(f"ActualGripperstate 形状: {state['ActualGripperstate'].shape}", 'Initial gripper state:', gripper_state)


            obs = env.get_obs_umi()
            print(f"\n观测数据键: {list(obs.keys())}")
            for key, value in obs.items():
                print(f"{key}: {value.shape} ({value.dtype})")
            
            # 添加详细的维度检查
            print("\n===== 创建观测字典前 =====")
            print(f"形状元数据 (shape_meta):")
            print(yaml.dump(OmegaConf.to_container(cfg.task.shape_meta)))


            episode_start_pose = list()
            for robot_id in range(len(robots_config)):
                pose = np.concatenate([
                    obs[f'robot{robot_id}_eef_pos'],
                    obs[f'robot{robot_id}_eef_rot_axis_angle']
                ], axis=-1)[-1]
                # pose = obs['robot_eef_pose']# robot_eef_pose
                episode_start_pose.append(pose)

            
            print("env_enable_sam2 == False, 全局条件rgb")
            with torch.no_grad():
                policy.reset()
                # dp
                # obs_dict_np = get_real_obs_dict(
                #     env_obs=obs, 
                #     shape_meta=cfg.task.shape_meta
                #     )
                # umi
                obs_dict_np = get_real_umi_obs_dict(
                    env_obs=obs, 
                    shape_meta=cfg.task.shape_meta,
                    obs_pose_repr=cfg.task.pose_repr.obs_pose_repr,
                    tx_robot1_robot0=tx_robot1_robot0,
                    episode_start_pose=episode_start_pose)

                # print("\n===== 观测字典内容 =====")
                # for key, value in obs_dict_np.items():
                #     print(f"{key}: {value.shape} ({value.dtype})")

                obs_dict = dict_apply(obs_dict_np, 
                    lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                # print("\n===== 张量观测字典内容 =====")
                # for key, value in obs_dict.items():
                #     print(f"{key}: {value.shape} ({value.dtype})")
                
                result = policy.predict_action(obs_dict)
                # print(f"预测结果键: {list(result.keys())}")
                # for key, value in result.items():
                #     print(f"{key}: {value.shape} ({value.dtype})")

                action = result['action'][0].detach().to('cpu').numpy()
                # print("result.shape", result['action'].shape, "\n action.shape:", action.shape)
                # print(f"动作形状: {action.shape}")
                # print("\n===== 转换为真实动作 =====")
                # print(f"动作姿态表示: {action_pose_repr}")
                # print(f"观测数据键: {list(obs.keys())}")

                # # 在转换前打印动作内容
                # print("原始动作前5个样本:")
                # for i in range(min(5, action.shape[0])):
                #     print(f"样本 {i}: {action[i]}")
                
                assert action.shape[-1] == 10 * len(robots_config)
                action = get_real_umi_action(action, obs, action_pose_repr)# (16, 7)
                # print(f"转换后动作形状: {action.shape}")
                # print("转换后动作前5个样本:")
                # for i in range(min(5, action.shape[0])):
                #     print(f"样本 {i}: {action[i]}")
                
                action_horizon = action.shape[0]
                action_dim = action.shape[-1]
                # print("action_horizon:", action_horizon, "action_dim:", action_dim)

                assert action.shape[-1] == 7 * len(robots_config)
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

                    obs = env.get_obs_umi()                # visualize
                    
                    # 1：处理RGB图像数据（480x640x3）
                    # vis_img = obs[f'camera_{vis_camera_idx}'][-1,:,:,::-1].copy()   # 获取可视化图像并转换颜色空间
                    vis_img = obs[f'camera{vis_camera_idx}_rgb'][-1,:,:,::-1].copy()   # 获取可视化图像并转换颜色空间

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
                            policy_control = False   # 关键！切换回人类控制
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
                        # 空格切换爪子状态
                        elif key_stroke == KeyCode(char=' '):
                            policy_control = not policy_control
                            print(f"Policy control {'enabled' if policy_control else 'disabled'}")
                        # 检测数字按键 (0-9)
                        # 检测数字按键 (0-9),解决Key对象没有char属性的问题
                        elif isinstance(key_stroke, KeyCode) and key_stroke.char and key_stroke.char in '0123456789':
                            stage = int(key_stroke.char)
                            print(f'数字键 {key_stroke.char} 被按下, 设置 stage={stage}')
                        else:
                            print(f"按键 {key_stroke} 未处理或不在预设范围内, stage={stage}")

                    if stage in stage_gripper_map:
                        gripper_state_exe = stage_gripper_map[stage]
                    else:
                        gripper_state_exe = (stage + 1) % 2 * 0.08 # 默认行为
                        print(f"警告: stage={stage}不在预设范围，自动计算夹爪状态为 {gripper_state_exe:.2f}")
                    # gripper_state = obs['robot_gripper'][-1][0]             # 获取当前夹爪状态
                    gripper_state = obs['robot0_gripper_width'][-1][0]             # 获取当前夹爪状态

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
                        # print("dpos:",dpos,"drot_xyz:",drot_xyz)
                        drot = st.Rotation.from_euler('xyz', drot_xyz)                  # 计算旋转
                        # print("dpos:",dpos,"drot:",drot.as_rotvec())
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
                    frame_latency = 1/30
                    precise_wait(eval_t_start - frame_latency, time_func=time.time)
                    print("Started!")
                    iter_idx = 0
                    perv_target_pose = None

                    
                    max_timesteps = 500
                    all_time_actions = np.zeros((max_timesteps, max_timesteps + action_horizon, action_dim))
                    inference_idx = steps_per_inference

                    while True:
                        # calculate timing
                        t_cycle_end = t_start + (iter_idx + steps_per_inference) * dt

                        obs = env.get_obs_umi()
                        # 以下为调试用
                        gripper_state = obs['robot_gripper'][-1]
                        obs_timestamps = obs['timestamp']
                        # 推理run inference

                        # with torch.no_grad():
                        #     s = time.time()
                        #     obs_dict_np = get_real_obs_dict(
                        #         env_obs=obs, 
                        #         shape_meta=cfg.task.shape_meta)
                    
                            
                        # obs_dict = dict_apply(obs_dict_np, 
                        #     lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                                
                        # result = policy.predict_action(obs_dict)
                        # # this action starts from the first obs step
                        # # dp
                        # # action = result['action'][0].detach().to('cpu').numpy()
                        # # 使用原始实现中的函数
                        # raw_action = result['action_pred'][0].detach().to('cpu').numpy()
                        # action = get_real_umi_action(raw_action, obs, cfg.task.pose_repr.action_pose_repr)


                        # run inference
                        with torch.no_grad():
                            s = time.time()
                            obs_dict_np = get_real_umi_obs_dict(
                                env_obs=obs, shape_meta=cfg.task.shape_meta, 
                                obs_pose_repr=obs_pose_rep,
                                tx_robot1_robot0=tx_robot1_robot0,
                                episode_start_pose=episode_start_pose)
                            obs_dict = dict_apply(obs_dict_np, 
                                lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                            result = policy.predict_action(obs_dict)
                            raw_action = result['action_pred'][0].detach().to('cpu').numpy()
                            raw_action2 = result['action'][0].detach().to('cpu').numpy()

                            # print(f"obs_dict_np['robot0_eef_pose'].shape: {obs_dict_np['robot0_eef_pose'].shape}")
                            # print(f"obs[robot0_eef_pos]: {obs['robot0_eef_pos']}")
                            # print(f"raw_action.shape: {raw_action.shape}, raw_action2.shape: {raw_action2.shape}")
                            # print(f"raw_action: {raw_action}, raw_action2: {raw_action2}")

                            # raw_action = result['action'][0].detach().to('cpu').numpy()
                            # print('Inference latency:', time.time() - s)
                            # print("result.shape", result['action'].shape, "\n action.shape:", action.shape)
                            # print(f"动作形状: {action.shape}")
                            # print("\n===== 转换为真实动作 =====")
                            # print(f"动作姿态表示: {action_pose_repr}")
                            # print(f"观测数据键: {list(obs.keys())}")

                            # # 在转换前打印动作内容
                            # print("原始动作前5个样本:")
                            # for i in range(min(5, action.shape[0])):
                            #     print(f"样本 {i}: {action[i]}")
                            
                            action = get_real_umi_action(raw_action, obs, action_pose_repr)# (16, 7)
                            # print(f"转换后动作形状: {action.shape}")
                            # print("转换后动作前5个样本:")
                            # for i in range(min(5, action.shape[0])):
                            #     print(f"样本 {i}: {action[i]}")

                            # 只使用位置部分 (XYZ)，保持姿态和夹爪不变
                            print("修改动作：只使用位置部分，保持姿态和夹爪不变")
                            for i in range(action.shape[0]):
                                # 获取当前姿态和夹爪状态
                                current_pose = obs['robot_eef_pose'][-1]  # 当前姿态
                                current_gripper = obs['robot0_gripper_width'][-1]  # 当前夹爪状态
                                
                                # 只保留预测的位置 (前3个元素)
                                action[i, :3] = action[i, :3]  # 位置保持不变
                                action[i, 3:6] = current_pose[3:6]  # 姿态保持当前值
                                action[i, 6] = current_gripper  # 夹爪保持当前值
                                # action[i, 6] = action[i, 6]  # 夹爪保持当前值

                            all_time_actions[[iter_idx], iter_idx:iter_idx + action_horizon] = action

                        if inference_idx == steps_per_inference:
                            inference_idx = 0
                            ensemble_steps = 8
                            # if temporal_agg:    # True
                            # temporal ensemble
                            action_seq_for_curr_step = all_time_actions[:, iter_idx:iter_idx + action_horizon]
                            target_pose_list = []
                            for i in range(action_horizon):
                                actions_for_curr_step = action_seq_for_curr_step[max(0, iter_idx - ensemble_steps + 1):iter_idx + 1, i]
                                actions_populated = np.all(actions_for_curr_step != 0, axis=1)
                                actions_for_curr_step = actions_for_curr_step[actions_populated]

                                k = -0.01
                                exp_weights = np.exp(k * np.arange(len(actions_for_curr_step)))
                                exp_weights = exp_weights / exp_weights.sum()
                                weighted_rotvec = R.from_rotvec(np.array(actions_for_curr_step)[:, 3:6]).mean(weights=exp_weights).as_rotvec()
                                weighted_action = (actions_for_curr_step * exp_weights[:, np.newaxis]).sum(axis=0, keepdims=True)
                                weighted_action[0][3:6] = weighted_rotvec
                                target_pose_list.append(weighted_action)
                            this_target_poses = np.concatenate(target_pose_list, axis=0)
                        else:
                            this_target_poses = action

                        assert this_target_poses.shape[1] == len(robots_config) * 7
                        for target_pose in this_target_poses:
                            for robot_idx in range(len(robots_config)):
                                solve_table_collision(
                                    ee_pose=target_pose[robot_idx * 7: robot_idx * 7 + 6],
                                    gripper_width=target_pose[robot_idx * 7 + 6],
                                    height_threshold=robots_config[robot_idx]['height_threshold']
                                )

                            # solve collison between two robots
                            solve_sphere_collision(
                                ee_poses=target_pose.reshape([len(robots_config), -1]),
                                robots_config=robots_config
                            )



                        # # 转换动作convert policy action to env actions
                        # if delta_action:
                        #     # assert len(action) == 1,"增量模式需要1维动作"
                        #     if perv_target_pose is None:
                        #         perv_target_pose = obs['robot_eef_pose'][-1]
                        #     this_target_pose = perv_target_pose.copy()
                        #     this_target_pose += action[-1]
                        #     perv_target_pose = this_target_pose
                        #     this_target_poses = np.expand_dims(this_target_pose, axis=0)
                        # else:                                           # 7替换掉len(target_pose)
                        #     assert action.shape[1] == 7, "绝对模式需要7维动作(x,y,z,rx,ry,rz,gripper)"
                        #     # 使用更安全的初始化方式
                        #     this_target_poses = np.empty_like(action)
                        #     # 方法0：直接赋值执行
                        #     this_target_poses = np.zeros((len(action), len(actions)), dtype=np.float64)
                        #     this_target_poses[:] = action
                        
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
                        # vis_img = obs[f'camera_{vis_camera_idx}'][-1,:,:,::-1].copy()   # 获取可视化图像并转换颜色空间
                        vis_img = obs[f'camera{vis_camera_idx}_rgb'][-1,:,:,::-1].copy()   # 获取可视化图像并转换颜色空间
                        cv2.imshow('RGB Image', vis_img) 

                        key_stroke = cv2.pollKey()
                        if key_stroke == ord('s'):
                            env.end_episode()
                            print('Stopped.')
                            break

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
                        iter_idx += 1
                        inference_idx += 1

                except KeyboardInterrupt:
                    print("Interrupted!")
                    # stop robot.
                    env.end_episode()
                
                print("Stopped.")



# %%
if __name__ == '__main__':
    main()
