python diffusion_policy/scripts/robomimic_dataset_conversion.py -i data/robomimic/datasets/coffee_preparation/coffee_preparation_d0.hdf5 -o data/robomimic/datasets/coffee_preparation/coffee_preparation_d0_abs.hdf5 -n 32

## 0907修改为基于AMED

### ddpm 170-86 180-94 210-92 220-98 230-90
python train_sim.py --config-name=robomimic_acdp_ddim task_name=coffee_preparation_d0 n_demo=100
9月7日 04.10.32_diff_c_coffee_preparation_d0

### ddim 
#### 推理步长20 使用dp3的unet网络 condition_type: cross_attention

训练：
python train_sim.py --config-name=robomimic_acdp_ddim_dp3 task_name=coffee_preparation_d0 n_demo=200
9月7日 14.54.59_diff_c_coffee_preparation_d0


### euler               stack d1 60-86 70-90 100-84 110-84 120-86
9月8日 16.37.58_diff_c_stack_d1


### ddim  demo=100      stack d1 70-82 170-80 340-78 390-80 550-80
9月7日 04.10.32_diff_c_coffee_preparation_d0

### eler demo=100 预测sample
CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python train_sim.py --config-name=robomimic_train_dp_diffusion_unet_abs_euler_a task_name=stack_d1 n_demo=100
9月8日 23.35.14_diff_c_stack_d1



### eler demo=100 优化后的噪声指标
CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python train_sim.py --config-name=robomimic_train_dp_diffusion_unet_abs_euler task_name=stack_d1 n_demo=100
9月8日 18.00.46_diff_c_stack_d1

## Baseline：预测噪声改为状态
###  demo=200   50次测试，成功率(epoch30开始到260 ，ckpt=0.68 0.64 0.66未更新，即成功率没大变化)
训练：
python train_sim.py --config-name=robomimic_train_dp_diffusion_unet_abs_sample task_name=coffee_preparation_d0 n_demo=200
9月5日 19.31.37_diff_c_coffee_preparation_d0

### ddim 
#### 推理步长20
训练：
CUDA_VISIBLE_DEVICES=1 python train_sim.py --config-name=robomimic_acdp_ddim task_name=coffee_preparation_d0 n_demo=200
9月5日 20.15.36_diff_c_coffee_preparation_d0

测试：
epoch:270 成功率 
python eval_sim.py --checkpoint /home/hy/Desktop/stagedp/data/outputs/2025.09.05/20.15.36_diff_c_coffee_preparation_d0/checkpoints/epoch=0270-test_mean_score=0.9200.ckpt -o data/coffee_preparation_d0_eval_output_09052015_270

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
CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python train_sim.py --config-name=robomimic_acdp_dpm_solver++_dpm task_name=coffee_preparation_d0 n_demo=200
9月6日 23.58.46_diff_c_coffee_preparation_d0

#### ipndm
python train_sim.py --config-name=robomimic_acdp_ddim task_name=coffee_preparation_d0 n_demo=200
9月6日

#### epd
python train_sim.py --config-name=robomimic_acdp_ddim task_name=coffee_preparation_d0 n_demo=200
9月6日

#### epd_parallel
python train_sim.py --config-name=robomimic_acdp_ddim task_name=coffee_preparation_d0 n_demo=200
9月6日

