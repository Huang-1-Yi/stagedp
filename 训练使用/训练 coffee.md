python diffusion_policy/scripts/robomimic_dataset_conversion.py -i data/robomimic/datasets/coffee_preparation/coffee_preparation_d0.hdf5 -o data/robomimic/datasets/coffee_preparation/coffee_preparation_d0_abs.hdf5 -n 32

## Baseline：预测噪声改为状态
###  demo=200   50次测试，成功率(epoch30开始到260 ，ckpt=0.68 0.64 0.66未更新，即成功率没大变化)
训练：
python train_sim.py --config-name=robomimic_train_dp_diffusion_unet_abs_sample task_name=coffee_preparation_d0 n_demo=200
9月5日 19.31.37_diff_c_coffee_preparation_d0

###  demo=400  50次测试，成功率(epoch200开始到440 ，ckpt=0.92未更新，即成功率没大变化)
训练：


###  demo=400 84改为224，改yaml文件  50次测试，成功率(epoch70开始到 ，ckpt=94+-2)
训练：
CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python train_sim.py 

### demo=400 512降维度到256  50次测试，成功率(epoch100开始 ，ckpt=90-94) 选定为baseline
训练：


## 加速改进：以robomimic_acdp_ddpm为baseline  84*84 256 64 300 400 
### ddpm
python train_sim.py --config-name=robomimic_acdp_ddpm task_name=stack_d1 n_demo=400

### ddim 
#### 推理步长20
训练：
CUDA_VISIBLE_DEVICES=1 python train_sim.py --config-name=robomimic_acdp_ddim task_name=coffee_preparation_d0 n_demo=200
9月5日 20.15.36_diff_c_coffee_preparation_d0

测试：
epoch:270 成功率 
python eval_sim.py --checkpoint /home/hy/Desktop/stagedp/data/outputs/2025.09.05/20.15.36_diff_c_coffee_preparation_d0/checkpoints/epoch=0270-test_mean_score=0.9200.ckpt -o data/coffee_preparation_d0_eval_output_09052015_270

#### 推理步长10
训练：
python train_sim.py --config-name=robomimic_acdp_ddim_infer10 task_name=stack_d1 n_demo=400
8月27日 
测试：

#### 推理步长5
80-90 100-90 240-88 250-90 290-92
训练：
CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python train_sim.py --config-name=robomimic_acdp_ddim_infer5 task_name=stack_d1 n_demo=400
8月30日 23.27.42_diff_c_stack_d1

测试：
epoch:290 成功率 88
python eval_sim.py --checkpoint /home/hy/Desktop/stagedp/data/outputs/2025.08.30/23.27.42_diff_c_stack_d1/checkpoints/epoch=0290-test_mean_score=0.9200.ckpt -o data/stack_d1_eval_output_08302327_290

epoch:latest 成功率 88
python eval_sim.py --checkpoint /home/hy/Desktop/stagedp/data/outputs/2025.08.30/23.27.42_diff_c_stack_d1/checkpoints/latest.ckpt -o data/stack_d1_eval_output_08302327_latest

###  dpm预测噪声epsilon
#### 预测噪声
#### inner层循环数量2 步长20 单步 2阶
python train_sim.py --config-name=robomimic_acdp_dpm_solver_2_2s task_name=stack_d1 n_demo=400
8月28日 11.18.45_diff_c_stack_d1

#### inner层循环数量2 步长20 多步 2阶
CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python train_sim.py --config-name=robomimic_acdp_dpm_solver_2_2m task_name=stack_d1 n_demo=400
8月28日 11.18.47_diff_c_stack_d1

#### 无inner层       步长20 单步 2阶
python train_sim.py --config-name=robomimic_acdp_dpm_solver_0_2s task_name=stack_d1 n_demo=400
8月28日 

#### 无inner层       步长20 多步 2阶
python train_sim.py --config-name=robomimic_acdp_dpm_solver_0_2m task_name=stack_d1 n_demo=400
8月28日 

#### 降步长

#### 

#### 预测状态(最后测)


### dpm++预测状态sample：提高基线成功率
#### inner层循环数量2 步长20 单步 2阶  50-90 90-90 150-92 160-88 220-88
训练：
python train_sim.py --config-name=robomimic_acdp_dpm_solver++ task_name=stack_d1 n_demo=400
8月27日 16.35.20_diff_c_stack_d1
测试：
epoch:50 成功率
python eval_sim.py --checkpoint /media/disk7t/outputs/2025.08.27/16.35.20_diff_c_stack_d1/checkpoints/epoch=0050-test_mean_score=0.9000.ckpt -o data/stack_d1_eval_output_08271635_50

epoch:90 成功率 78
python eval_sim.py --checkpoint /media/disk7t/outputs/2025.08.27/16.35.20_diff_c_stack_d1/checkpoints/epoch=0090-test_mean_score=0.9000.ckpt -o data/stack_d1_eval_output_08271635_90

epoch:90_es1.05 成功率 78
python eval_sim.py --checkpoint /media/disk7t/outputs/2025.08.27/16.35.20_diff_c_stack_d1/checkpoints/epoch=0090-test_mean_score=0.9000.ckpt -o data/stack_d1_eval_output_08271635_90_es

epoch:90_es1.1 成功率 86
python eval_sim.py --checkpoint /media/disk7t/outputs/2025.08.27/16.35.20_diff_c_stack_d1/checkpoints/epoch=0090-test_mean_score=0.9000.ckpt -o data/stack_d1_eval_output_08271635_90_es11

epoch:90_es1.15 成功率 80
python eval_sim.py --checkpoint /media/disk7t/outputs/2025.08.27/16.35.20_diff_c_stack_d1/checkpoints/epoch=0090-test_mean_score=0.9000.ckpt -o data/stack_d1_eval_output_08271635_90_es115

epoch:90_es1.15 成功率 82
python eval_sim.py --checkpoint /media/disk7t/outputs/2025.08.27/16.35.20_diff_c_stack_d1/checkpoints/epoch=0090-test_mean_score=0.9000.ckpt -o data/stack_d1_eval_output_08271635_90_es12

epoch:150 成功率
python eval_sim.py --checkpoint /media/disk7t/outputs/2025.08.27/16.35.20_diff_c_stack_d1/checkpoints/epoch=0150-test_mean_score=0.9200.ckpt -o data/stack_d1_eval_output_08271635_150

epoch:160 成功率
python eval_sim.py --checkpoint /media/disk7t/outputs/2025.08.27/16.35.20_diff_c_stack_d1/checkpoints/epoch=0160-test_mean_score=0.8800.ckpt -o data/stack_d1_eval_output_08271635_160


#### inner层循环数量2 步长20 多步 2阶 130-90 170-96 180-88 190-88 290-90
python train_sim.py --config-name=robomimic_acdp_dpm_solver++_2_2m task_name=stack_d1 n_demo=400
8月27日 21.09.14_diff_c_stack_d1
测试：
epoch: 130 成功率 90
python eval_sim.py --checkpoint /media/disk7t/outputs/2025.08.27/21.09.14_diff_c_stack_d1/checkpoints/epoch=0130-test_mean_score=0.9000.ckpt -o data/stack_d1_eval_output_08272109

epoch: 130_es95 成功率 88
python eval_sim.py --checkpoint /media/disk7t/outputs/2025.08.27/21.09.14_diff_c_stack_d1/checkpoints/epoch=0130-test_mean_score=0.9000.ckpt -o data/stack_d1_eval_output_08272109_es95

epoch: 130_es105 成功率 86
python eval_sim.py --checkpoint /media/disk7t/outputs/2025.08.27/21.09.14_diff_c_stack_d1/checkpoints/epoch=0130-test_mean_score=0.9000.ckpt -o data/stack_d1_eval_output_08272109_es105

epoch: 130_es11 成功率 80
python eval_sim.py --checkpoint /media/disk7t/outputs/2025.08.27/21.09.14_diff_c_stack_d1/checkpoints/epoch=0130-test_mean_score=0.9000.ckpt -o data/stack_d1_eval_output_08272109_es11

epoch:170_es95 成功率 84
python eval_sim.py --checkpoint /media/disk7t/outputs/2025.08.27/21.09.14_diff_c_stack_d1/checkpoints/epoch=0170-test_mean_score=0.9600.ckpt -o data/stack_d1_eval_output_08272109_170_es95

epoch:170 成功率 90
python eval_sim.py --checkpoint /media/disk7t/outputs/2025.08.27/21.09.14_diff_c_stack_d1/checkpoints/epoch=0170-test_mean_score=0.9600.ckpt -o data/stack_d1_eval_output_08272109_170

epoch:170_es101 成功率 86
python eval_sim.py --checkpoint /media/disk7t/outputs/2025.08.27/21.09.14_diff_c_stack_d1/checkpoints/epoch=0170-test_mean_score=0.9600.ckpt -o data/stack_d1_eval_output_08272109_170_es101

epoch:170_es102 成功率 90
python eval_sim.py --checkpoint /media/disk7t/outputs/2025.08.27/21.09.14_diff_c_stack_d1/checkpoints/epoch=0170-test_mean_score=0.9600.ckpt -o data/stack_d1_eval_output_08272109_170_es102

epoch:170_es103 成功率 88
python eval_sim.py --checkpoint /media/disk7t/outputs/2025.08.27/21.09.14_diff_c_stack_d1/checkpoints/epoch=0170-test_mean_score=0.9600.ckpt -o data/stack_d1_eval_output_08272109_170_es103

epoch:170_es105 成功率 92
python eval_sim.py --checkpoint /media/disk7t/outputs/2025.08.27/21.09.14_diff_c_stack_d1/checkpoints/epoch=0170-test_mean_score=0.9600.ckpt -o data/stack_d1_eval_output_08272109_170_es105

epoch:170_es11 成功率 90
python eval_sim.py --checkpoint /media/disk7t/outputs/2025.08.27/21.09.14_diff_c_stack_d1/checkpoints/epoch=0170-test_mean_score=0.9600.ckpt -o data/stack_d1_eval_output_08272109_170_es11

epoch:170_es115 成功率 90
python eval_sim.py --checkpoint /media/disk7t/outputs/2025.08.27/21.09.14_diff_c_stack_d1/checkpoints/epoch=0170-test_mean_score=0.9600.ckpt -o data/stack_d1_eval_output_08272109_170_es115

epoch:180 成功率 84
python eval_sim.py --checkpoint /media/disk7t/outputs/2025.08.27/21.09.14_diff_c_stack_d1/checkpoints/epoch=0180-test_mean_score=0.8800.ckpt -o data/stack_d1_eval_output_08272109_180

epoch:180_es105 成功率 86
python eval_sim.py --checkpoint /media/disk7t/outputs/2025.08.27/21.09.14_diff_c_stack_d1/checkpoints/epoch=0180-test_mean_score=0.8800.ckpt -o data/stack_d1_eval_output_08272109_180_es105

#### 无inner层       步长20 单步 2阶
python train_sim.py --config-name=robomimic_acdp_dpm_solver++_0_2s task_name=stack_d1 n_demo=400
8月27日 21.09.31_diff_c_stack_d1
测试：
epoch: 成功率
python eval_sim.py --checkpoint /latest.ckpt -o data/stack_d1_eval_output_0827210931


#### 无inner层       步长20 多步 2阶
python train_sim.py --config-name=robomimic_acdp_dpm_solver++_0_2m task_name=stack_d1 n_demo=400
8月28日 06.09.22_diff_c_stack_d1
测试：
epoch: 成功率
python eval_sim.py --checkpoint /latest.ckpt -o data/stack_d1_eval_output_08271635


#### inner层循环数量2 步长10 单步 2阶
130-94 160-92 190-90 200-90 260-92
python train_sim.py --config-name=robomimic_acdp_dpm_solver++_infer10 task_name=stack_d1 n_demo=400
8月28日 06.13.58_diff_c_stack_d1
测试：
epoch:130 成功率 90
python eval_sim.py --checkpoint /media/disk7t/outputs/2025.08.28/06.13.58_diff_c_stack_d1/checkpoints/epoch=0130-test_mean_score=0.9400.ckpt -o data/stack_d1_eval_output_08280613_130

epoch:130_es120 成功率 84
python eval_sim.py --checkpoint /media/disk7t/outputs/2025.08.28/06.13.58_diff_c_stack_d1/checkpoints/epoch=0130-test_mean_score=0.9400.ckpt -o data/stack_d1_eval_output_08280613_130_es120

epoch:130_es105 成功率 82
python eval_sim.py --checkpoint /media/disk7t/outputs/2025.08.28/06.13.58_diff_c_stack_d1/checkpoints/epoch=0130-test_mean_score=0.9400.ckpt -o data/stack_d1_eval_output_08280613_130_es105

epoch:130_es103 成功率 84
python eval_sim.py --checkpoint /media/disk7t/outputs/2025.08.28/06.13.58_diff_c_stack_d1/checkpoints/epoch=0130-test_mean_score=0.9400.ckpt -o data/stack_d1_eval_output_08280613_130_es103

epoch:130_es101 成功率 88
python eval_sim.py --checkpoint /media/disk7t/outputs/2025.08.28/06.13.58_diff_c_stack_d1/checkpoints/epoch=0130-test_mean_score=0.9400.ckpt -o data/stack_d1_eval_output_08280613_130_es101

epoch:130_es99 成功率 88
python eval_sim.py --checkpoint /media/disk7t/outputs/2025.08.28/06.13.58_diff_c_stack_d1/checkpoints/epoch=0130-test_mean_score=0.9400.ckpt -o data/stack_d1_eval_output_08280613_130_es99

epoch:160 成功率 
python eval_sim.py --checkpoint /media/disk7t/outputs/2025.08.28/06.13.58_diff_c_stack_d1/checkpoints/epoch=0160-test_mean_score=0.9200.ckpt -o data/stack_d1_eval_output_08280613_160

#### inner层循环数量2 步长5 单步 2阶 eps_scaler=1.01
170-92 210-88 220-98 250-94 260-94
python train_sim.py --config-name=robomimic_acdp_dpm_solver++_infer5 task_name=stack_d1 n_demo=400
8月30日 23.21.23_diff_c_stack_d1

测试：
epoch:250 成功率 90
python eval_sim.py --checkpoint /home/hy/Desktop/stagedp/data/outputs/2025.08.30/23.21.23_diff_c_stack_d1/checkpoints/epoch=0250-test_mean_score=0.9400.ckpt -o data/stack_d1_eval_output_08302321

epoch:250 成功率 88
python eval_sim.py --checkpoint /home/hy/Desktop/stagedp/data/outputs/2025.08.30/23.21.23_diff_c_stack_d1/checkpoints/epoch=0250-test_mean_score=0.9400.ckpt -o data/stack_d1_eval_output_08302321_es101

epoch:250 成功率 0
python eval_sim.py --checkpoint /home/hy/Desktop/stagedp/data/outputs/2025.08.30/23.21.23_diff_c_stack_d1/checkpoints/epoch=0250-test_mean_score=0.9400.ckpt -o data/stack_d1_eval_output_08302321_es98

epoch:250 成功率 36
python eval_sim.py --checkpoint /home/hy/Desktop/stagedp/data/outputs/2025.08.30/23.21.23_diff_c_stack_d1/checkpoints/epoch=0250-test_mean_score=0.9400.ckpt -o data/stack_d1_eval_output_08302321_es99

epoch:250 成功率 2
python eval_sim.py --checkpoint /home/hy/Desktop/stagedp/data/outputs/2025.08.30/23.21.23_diff_c_stack_d1/checkpoints/epoch=0250-test_mean_score=0.9400.ckpt -o data/stack_d1_eval_output_08302321_es105

epoch:250 成功率 66
python eval_sim.py --checkpoint /home/hy/Desktop/stagedp/data/outputs/2025.08.30/23.21.23_diff_c_stack_d1/checkpoints/epoch=0250-test_mean_score=0.9400.ckpt -o data/stack_d1_eval_output_08302321_es102

#### inner层循环数量2 步长5 单步 2阶 eps_scaler=1
210-94 220-92 230-88 280-90 290-90
CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python train_sim.py --config-name=robomimic_acdp_dpm_solver++_infer5 task_name=stack_d1 n_demo=400
8月31日 15.56.17_diff_c_stack_d1

测试：
epoch:210 成功率 84
python eval_sim.py --checkpoint /home/hy/Desktop/stagedp/data/outputs/2025.08.31/15.56.17_diff_c_stack_d1/checkpoints/epoch=0210-test_mean_score=0.9400.ckpt -o data/stack_d1_eval_output_08311556

epoch:210 成功率 86
python eval_sim.py --checkpoint /home/hy/Desktop/stagedp/data/outputs/2025.08.31/15.56.17_diff_c_stack_d1/checkpoints/epoch=0210-test_mean_score=0.9400.ckpt -o data/stack_d1_eval_output_08311556_es101

epoch:210 成功率 64
python eval_sim.py --checkpoint /home/hy/Desktop/stagedp/data/outputs/2025.08.31/15.56.17_diff_c_stack_d1/checkpoints/epoch=0210-test_mean_score=0.9400.ckpt -o data/stack_d1_eval_output_08311556_es102

epoch:latest 成功率 86
python eval_sim.py --checkpoint /home/hy/Desktop/stagedp/data/outputs/2025.08.31/15.56.17_diff_c_stack_d1/checkpoints/latest.ckpt -o data/stack_d1_eval_output_08311556_latest

epoch:latest 成功率 72
python eval_sim.py --checkpoint /home/hy/Desktop/stagedp/data/outputs/2025.08.31/15.56.17_diff_c_stack_d1/checkpoints/latest.ckpt -o data/stack_d1_eval_output_08311556_latest_es102

epoch:latest 成功率 0
python eval_sim.py --checkpoint /home/hy/Desktop/stagedp/data/outputs/2025.08.31/15.56.17_diff_c_stack_d1/checkpoints/latest.ckpt -o data/stack_d1_eval_output_08311556_latest_es105

epoch:latest 成功率 0
python eval_sim.py --checkpoint /home/hy/Desktop/stagedp/data/outputs/2025.08.31/15.56.17_diff_c_stack_d1/checkpoints/latest.ckpt -o data/stack_d1_eval_output_08311556_latest_es99

#### inner层循环数量2 步长10 单步 2阶 heun 130-92 170-86 190-86 230-90 250-88
python train_sim.py --config-name=robomimic_acdp_dpm_solver++_heun_2_2s task_name=stack_d1 n_demo=400
9月4日 15.39.05_diff_c_stack_d1

#### 无inner层       步长10 单步 2阶 heun 90-90 100-92 180-92 210-90 240-90
CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python train_sim.py --config-name=robomimic_acdp_dpm_solver++_heun_0_2s task_name=stack_d1 n_demo=400
9月4日 15.42.27_diff_c_stack_d1


#### 外层循环数量2 步长10


#### 外层循环数量2改到3 步长20 

#### 外层循环数量2改到1 步长20


### 多种采样方法测试
#### heun
python train_sim.py --config-name=robomimic_acdp_dpm_solver++_heun task_name=coffee_preparation_d0 n_demo=200
9月6日 10.57.53_diff_c_coffee_preparation_d0

#### dpm
python train_sim.py --config-name=robomimic_acdp_dpm_solver++_dpm task_name=coffee_preparation_d0 n_demo=200
9月6日 

#### ipndm
python train_sim.py --config-name=robomimic_acdp_ddim task_name=coffee_preparation_d0 n_demo=200
9月6日

#### epd
python train_sim.py --config-name=robomimic_acdp_ddim task_name=coffee_preparation_d0 n_demo=200
9月6日

#### epd_parallel
python train_sim.py --config-name=robomimic_acdp_ddim task_name=coffee_preparation_d0 n_demo=200
9月6日





### 方案1：修改config和文件值失败
num_inference_steps: 20 
    # 修改后（DPM-Solver++多步版）
    noise_scheduler:
        _target_: diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler
        algorithm_type: dpmsolver++
        num_train_timesteps: 100
        beta_start: 0.0001
        beta_end: 0.02
        beta_schedule: squaredcos_cap_v2
        solver_type: midpoint  # 或者 "heun"
        lower_order_final: True
        # 注意：DPMSolverMultistepScheduler 没有 variance_type 参数
        # 它使用固定的方差类型
        prediction_type: epsilon  # 与您之前使用的相同

且
  noise_scheduler:
    _target_: diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler
    algorithm_type: "dpmsolver++"  # 指定DPM-Solver++算法[6,7](@ref)
    solver_order: 2                # 二阶求解（平衡速度与质量）
    thresholding: True             # 启用动态阈值截断（提升稳定性）[1](@ref)
    beta_schedule: squaredcos_cap_v2  # 保持与DDIM一致的噪声计划
    beta_start: 0.0001
    beta_end: 0.02
    num_train_timesteps: 20
    prediction_type: epsilon       # 需与模型预测类型匹配
policy中代码：
dpm_solver在conditional_sample方法中修改采样循环
        for t in scheduler.timesteps:
            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                **kwargs
                ).prev_sample
改为
            # DPM-Solver++需使用scheduler.step的return_dict形式[7](@ref)
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                return_dict=True,   # 必须启用字典返回
                **kwargs
            ).prev_sample           # 从字典获取结果

        
        # 全局种子确保实验完全可复现
        def set_seed(seed=42):
            import random, numpy as np, torch
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True  # 消除CUDA随机性 [6,8](@ref)

        # 仅固定 generator无法控制数据加载、模型初始化等随机源，在模型初始化后调用
        set_seed(42)  # 与generator种子一致

### 方案2：直接修改config
#### 140个epoch后仍然为0.0
训练：
python train_sim.py --config-name=robomimic_train_dp_diffusion_unet_abs_dpmsolver++0821 task_name=stack_d1 n_demo=200
8月21日
测试：

_target_: diffusion_policy.policy.robomimic.diffusion_unet_hybrid_image_policy.DiffusionUnetHybridImagePolicy
  noise_scheduler:
    _target_: diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler
    algorithm_type: dpmsolver++
    num_train_timesteps: 100           # num_inference_steps < num_train_timesteps
    beta_start: 0.0001
    beta_end: 0.02
    beta_schedule: squaredcos_cap_v2
    solver_type: midpoint  # 或者 "heun"
    lower_order_final: True
    # 注意：DPMSolverMultistepScheduler 没有 variance_type 参数
    # 它使用固定的方差类型
    prediction_type: epsilon  # 与您之前使用的相同

#### 140个epoch后仍然为0.0
_target_: acdp.policy.robomimic.diffusion_unet_hybrid_image_policy.DiffusionUnetHybridImagePolicy
  noise_scheduler:
    _target_: diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler
    algorithm_type: dpmsolver++
    num_train_timesteps: 100           # num_inference_steps < num_train_timesteps
    beta_start: 0.0001
    beta_end: 0.02
    beta_schedule: squaredcos_cap_v2
    solver_type: midpoint  # 或者 "heun"
    lower_order_final: True
    # 注意：DPMSolverMultistepScheduler 没有 variance_type 参数
    # 它使用固定的方差类型
    prediction_type: epsilon  # 与您之前使用的相同


### 方案3：修改
#### 版本1代码
python train_sim.py --config-name=robomimic_train_dp_diffusion_unet_abs_dpmsolver++epd task_name=stack_d1 n_demo=200
26s一个epoch 设定600个epoch， 15gb
  _target_: acdp.policy.robomimic.diffusion_unet_hybrid_image_policy_dpmsolver_epd.DiffusionUnetHybridImagePolicy

  shape_meta: ${shape_meta}
  
###### 修改后（DPM-Solver++多步版）
  noise_scheduler:
    _target_: diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler
    algorithm_type: dpmsolver++       # 指定DPM-Solver++算法
    num_train_timesteps: 100          # 训练步数100步，推理步压缩至20步
    beta_start: 0.0001
    beta_end: 0.02
    beta_schedule: squaredcos_cap_v2  # 保持与DDIM一致的噪声计划
    solver_type: midpoint
    solver_order: 2                   # 显式添加二阶求解器（平衡速度与质量）
    lower_order_final: True
    prediction_type: epsilon

#### 版本2代码运行，824凌晨版本
CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python train_sim.py --config-name=robomimic_train_dp_diffusion_unet_abs_dpmsolver++epd_zxr task_name=stack_d1 n_demo=200

外层循环为3
02.47.12_diff_c_stack_d1
外层循环为2
13.31.42_diff_c_stack_d1
外层循环为2，但是预测真实值而不是噪声
15.42.36_diff_c_stack_d1
/epoch=0030-test_mean_score=0.5800.ckpt
epoch=0040-test_mean_score=0.6000.ckpt
epoch=0070-test_mean_score=0.5800.ckpt
epoch=0080-test_mean_score=0.6800.ckpt
epoch=0090-test_mean_score=0.5600.ckpt
epoch=0040-test_mean_score=0.6000.ckpt
epoch=0080-test_mean_score=0.6800.ckpt
epoch=0110-test_mean_score=0.6600.ckpt
epoch=0130-test_mean_score=0.6000.ckpt
epoch=0140-test_mean_score=0.6000.ckpt