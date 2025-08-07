# 版本：2025.07.14
# 作者：黄一
# conda activate dp0628
# CUDA_VISIBLE_DEVICES=0 python demo_real_franka_test.py -o data/column_0714 -ri 192.168.0.168
import time
from multiprocessing.managers                               import SharedMemoryManager
import click
import cv2
import numpy as np
import scipy.spatial.transform as st
from diffusion_policy.real_world.real_env_franka            import RealEnvFranka as RealEnv
from diffusion_policy.real_world.spacemouse_shared_memory   import Spacemouse
from diffusion_policy.common.precise_sleep                  import precise_wait
from diffusion_policy.real_world.keystroke_counter          import ( KeystrokeCounter, Key, KeyCode )

@click.command()
@click.option('--output', '-o', required=True, help="Directory to save demonstration dataset.")
@click.option('--robot_ip', '-ri', required=True, help="Franka's IP address e.g. 172.16.0.1")
@click.option('--vis_camera_idx', default=0, type=int, help="Which RealSense camera to visualize.")
@click.option('--init_joints', '-j', is_flag=True, default=False, help="Whether to initialize robot joint configuration in the beginning.")
@click.option('--frequency', '-f', default=10, type=float, help="Control frequency in Hz.")
@click.option('--command_latency', '-cl', default=0.01, type=float, help="Latency between receiving SapceMouse command to executing on Robot in Sec.")
def main(output, robot_ip, vis_camera_idx, init_joints, frequency, command_latency):
    dt = 1 / frequency
    with SharedMemoryManager() as shm_manager:
        with KeystrokeCounter() as key_counter, \
                Spacemouse(shm_manager=shm_manager) as sm, \
                RealEnv(
                    output_dir=output,                          # 实验结果输出目录
                    robot_ip=robot_ip,                          # 机器人IP地址
                    obs_image_resolution=(640, 480),            # 图像分辨率1280, 720    # recording resolution
                    frequency=frequency,                        # 控制频率
                    init_joints=init_joints,                    # 初始化关节
                    enable_multi_cam_vis=True,                  # 多相机可视化
                    record_raw_video    =True,                  # 记录原始视频,number of threads per camera view for video recording (H.264)
                    thread_per_video    =3,                     # 每个相机视图的线程数用于视频录制(H.264)
                    video_capture_resolution=(640, 480),        # 视频捕获分辨率
                    video_crf=21,                               # 视频录制质量，越低越好（但速度较慢）
                    multi_cam_vis_resolution=(640,480),         # 多相机可视化分辨率
                    shm_manager=shm_manager,                    # 共享内存管理器
                    enable_depth  = True,                      # False True是否启用深度图像
                    # enable_sam2 = True,                         # 是否启用SAM2
                    enable_sam2 = False,                        # 是否启用SAM2
                    # enable_predictor=True,                      # 是否启用stage预测器
                ) as env:                                       # 实例化RealEnvFranka类
            cv2.setNumThreads(1)                                # 设置OpenCV线程数，可修改
            env_enable_predictor = env.enable_predictor
            stage_gripper_map = {
                0: 0.0,    # 运动
                1: 0.0,   # 抓紫圆柱+运动
                2: 0.0,    # 放紫圆柱+返回
            }# 卷尺0.05
            time.sleep(1.0)                                     # 休眠1秒
            print(' ✅ Ready!')
            state = env.get_robot_state()                       # 获取机器人状态
            target_pose = state['ActualTCPPose']                # 获取目标姿态
            gripper_state = state['ActualGripperstate']         # 获取夹爪状态
            print('Initial pose:', target_pose)                 # 打印初始姿态
            print('Initial gripper state:', gripper_state)      # 打印初始夹爪状态
            t_start = time.monotonic()                          # 获取当前时间
            iter_idx = 0                                        # 迭代次数 
            stop = False                                        # 停止标志 
            is_recording = False                                # 录制标志

            return_begin = False
            init_position = np.array([0.48, 0.0, 0.7,-2.8691,1.2213,0.0139])

            epsilon = 0.01                                      # 位置误差容忍阈值 (单位：米)0.004
            step_size = 0.015                                   # 调整步长0.01
            epsilon_rot = 0.02                                  # 角度死区阈值 (约1度，单位：弧度)
            max_step_angle = 0.03                               # 最大单步调整角度 (约5度，单位：弧度)
            pos_arrived = [False, False, False, False]          # 归位标志符
            stage = 0
            
            while not stop:
                # 1.计算时间
                t_cycle_end = t_start + (iter_idx + 1) * dt     # 计算循环结束时间
                t_sample = t_cycle_end - command_latency        # 计算采样时间
                t_command_target = t_cycle_end + dt             # 计算命令目标人
                # 2.获取观察结果                
                obs = env.get_obs()
                # 3.处理按键事件
                #     如果按下的是'q'键，退出程序
                #     如果按下的是'c'键，开始录制
                #     如果按下的是's'键，停止录制
                #     如果按下的是退格键，# 删除最近录制的集
                press_events = key_counter.get_press_events()   # 获取按键事件
                for key_stroke in press_events:                 # 遍历按键事件
                    print("key_stroke",key_stroke)
                    if key_stroke == KeyCode(char='q'):
                        env.end_episode()
                        key_counter.clear()                     # 清除按键计数器
                        stage = 0                               # 重置阶段
                        stop = True                             # 设置停止标志为True
                        cv2.destroyAllWindows()                 # 关闭所有OpenCV窗口    
                    elif key_stroke == KeyCode(char='j'): 
                        env.start_episode(t_start + (iter_idx + 2) * dt - time.monotonic() + time.time()) # 开始新集
                        key_counter.clear()                     # 清除按键计数器
                        stage = 0                               # 重置阶段
                        is_recording = True                     # 设置录制标志为True
                        return_begin = False
                        print('Recording!')                     # 打印录制消息
                    elif key_stroke == KeyCode(char='s'):
                        env.end_episode()                       # 结束当前集
                        key_counter.clear()                     # 清除按键计数器
                        stage = 0                               # 重置阶段
                        is_recording = False                    # 设置录制标志为False
                        return_begin = False
                        print('Stopped.')                       # 打印停止消息
                    elif key_stroke == Key.backspace:
                        if click.confirm('Are you sure to drop an episode?'): # 确认删除
                            env.drop_episode()                  # 删除集
                            key_counter.clear()                 # 清除按键计数器
                            stage = 0                           # 重置阶段
                            is_recording = False                # 设置录制标志为False
                            print('Episode dropped.')           # 打印删除消息
                    elif key_stroke == KeyCode(char='k'):
                        return_begin = True
                        print("r被按下, Return to initial pose triggered")
                    elif key_stroke == Key.space:
                        stage += 1                              # 严格单向递增
                        break                                   # 每帧仅处理一次空格事件
                    elif key_stroke == KeyCode(char='f'):
                        print("f被按下")
                    # 检测数字按键 (0-9),解决Key对象没有char属性的问题
                    elif isinstance(key_stroke, KeyCode) and key_stroke.char and key_stroke.char in '0123456789':
                        stage = int(key_stroke.char)
                        print(f'数字键 {key_stroke.char} 被按下，设置 stage={stage}')
                    else:
                        print(f"按键 {key_stroke} 未处理或不在预设范围内，stage={stage}")
                if stage in stage_gripper_map:
                    gripper_state_exe = stage_gripper_map[stage]
                else:
                    gripper_state_exe = (stage + 1) % 2 * 0.08 # 默认行为
                    print(f"警告: stage={stage}不在预设范围，自动计算夹爪状态为 {gripper_state_exe:.2f}")

                # 获取实际夹爪状态
                gripper_state = obs['robot_gripper'][-1][0]                # 获取当前夹爪状态

                # 4.可视化：处理RGB图像数据（480x640x3）
                # vis_img = obs[f'camera_{vis_camera_idx}'][-1,:,:,::-1].copy()   # 获取可视化图像并转换颜色空间
                vis_img = cv2.resize(
                        obs[f'camera_{vis_camera_idx}'][-1,:,:,::-1].copy(), 
                        (1280, 960)  # (width, height)
                    )

                # 处理显示图像
                def draw_common_text(img_input):
                    episode_id = env.replay_buffer.n_episodes                               # 获取当前集ID
                    text = f'Episode: {episode_id}, Stage: {stage}'                         # 设置文本为当前集ID和阶段
                    if is_recording:                                                        # 如果正在录制
                        text += ', Recording!'                                              # 添加录制文本
                    cv2.putText(                                                            # 在可视化图像上绘制文本
                        img_input,
                        text,
                        (10, 30),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1,
                        thickness=2,
                        color=(255, 255, 255)
                    )
                    return img_input
                vis_img = draw_common_text(vis_img)   
                cv2.imshow('RGB Image', vis_img)

                cv2.pollKey()                                                           # OpenCV键盘事件处理
                precise_wait(t_sample)
                
                if not env.is_saving:  # 仅在非保存状态响应空间鼠标
                    # 模拟空间鼠标
                    if return_begin:
                        # state = env.get_robot_state()
                        target_pose = state['ActualTCPPose']
                        init = target_pose
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
                        print("target_pose ",target_pose)
                        print("init_pose ",init)
                        gripper_state_exe = 0.0
                        # gripper_state_exe = 0.08                                        # 夹爪状态
                        # ===== 旋转控制结束 =====
                        if all(pos_arrived):                                            # 有滞后
                            print("归位成功,当前关节位置为",obs['robot_joint'][-1],"释放控制权")
                            pos_arrived = [False, False, False, False]                  # 归位标志符
                            return_begin = False
                    
                    # 5.遥操作Resized stage: (20291,) -> (20202,
                    # 5.1 获取遥操作命令
                    else:
                        # print("state['ActualTCPPose']",state['ActualTCPPose'])
                        sm_state = sm.get_motion_state_transformed()                    # 获取SpaceMouse的运动状态
                        if sm.is_button_pressed(12):
                            print("memu:进度+1")
                            stage += 1
                        elif sm.is_button_pressed(13):
                            print("w(2):开始写入")
                            env.start_episode(t_start + (iter_idx + 2) * dt - time.monotonic() + time.time()) # 开始新集
                            key_counter.clear()                     # 清除按键计数器
                            stage = 0                               # 重置阶段
                            is_recording = True                     # 设置录制标志为True
                            return_begin = False
                            print('Recording!')       
                        elif sm.is_button_pressed(14):
                            print("r(3):归位")
                            return_begin = True
                        elif sm.is_button_pressed(15):
                            print("s(4):停止")
                            env.end_episode()                       # 结束当前集
                            key_counter.clear()                     # 清除按键计数器
                            stage = 0                               # 重置阶段
                            is_recording = False                    # 设置录制标志为False
                            return_begin = False
                            print('Stopped.')                       # 打印停止消息
                        elif sm.is_button_pressed(1):
                            if click.confirm('Are you sure to drop an episode?'): # 确认删除
                                print("删除(menu):删除")
                                env.drop_episode()                  # 删除集
                                key_counter.clear()                 # 清除按键计数器
                                stage = 0                           # 重置阶段
                                is_recording = False                # 设置录制标志为False
                                print('Episode dropped.')           # 打印删除消息
                        elif sm.is_button_pressed(0):
                            print("q(fit):离开")
                            env.end_episode()
                            key_counter.clear()                     # 清除按键计数器
                            stage = 0                               # 重置阶段
                            stop = True                             # 设置停止标志为True
                            cv2.destroyAllWindows()                 # 关闭所有OpenCV窗口  
                        dpos = sm_state[:3] * (env.max_pos_speed / frequency)           # 计算位置增量
                        # dpos[:] = 0  # 禁止移动
                        drot_xyz = sm_state[3:] * (env.max_rot_speed / frequency)       # 计算旋转增量
                        drot_xyz[:] = 0  # 禁止旋转
                        drot = st.Rotation.from_euler('xyz', drot_xyz)                  # 计算旋转
                        target_pose[:3] += dpos                                         # 更新目标位置
                        target_pose[3:6] = (drot * st.Rotation.from_rotvec(target_pose[3:6])).as_rotvec()# 更新目标旋转
                else:
                    # 保存期间发送零动作保持机器人静止
                    print("Saving, sending zero action.")
                actions =np.zeros(7)
                actions[:6] = target_pose
                actions[6] = gripper_state_exe
                # 5.2 执行遥操作命令
                env.exec_actions(                                                       # 执行动作
                    actions=[actions], 
                    timestamps=[t_command_target - time.monotonic() + time.time()],
                    stages=np.array([stage], dtype=np.int64)                            # 显式转换当前阶段值为 np.ndarray 一维数组
                    )
                precise_wait(t_cycle_end)                                               # 精确等待循环结束时间
                iter_idx += 1                                                           # 增加迭代索引
            
            cv2.destroyAllWindows()                                                     # 关闭所有OpenCV窗口

# %%
if __name__ == '__main__':
    main()