# Equivariant Diffusion Policy
[Project Website](https://stagedp.github.io) | [Github]https://github.com/Huang-1-Yi/stagedp | [Paper](https:/) | [Video](https://y)


## 环境
    ```bash
    conda activate equidiff
    ```
    
## 数据集生成
### 生成点云和体素观测
1.  用cpu生成
    ```bash
    python diffusion_policy/scripts/dataset_states_to_obs.py --input data/robomimic/datasets/stack_d1/stack_d1.hdf5 --output data/robomimic/datasets/stack_d1/stack_d1_voxel.hdf5 --num_workers=32 
    ```
2.  用gpu加速
    ```bash
    python diffusion_policy/scripts/dataset_states_to_obs_gpu.py --input data/robomimic/datasets/stack_d1/stack_d1.hdf5 --output data/robomimic/datasets/stack_d1/stack_d1_voxel.hdf5 --num_workers=32 
    ```
### 原始dp的transformer（代码带“_gpu”默认用gpu加速）
1.  转换stack_d1（非体素）
    ```bash
    python diffusion_policy/scripts/robomimic_dataset_conversion_gpu.py -i data/robomimic/datasets/stack_d1/stack_d1.hdf5 -o data/robomimic/datasets/stack_d1/stack_d1_abs.hdf5 -n 32 --use_gpu --suppress_warnings
    ```
2.  转换stack_d1_voxel（体素）
    ```bash
    python diffusion_policy/scripts/robomimic_dataset_conversion_gpu.py -i data/robomimic/datasets/stack_d1/stack_d1_voxel.hdf5 -o data/robomimic/datasets/stack_d1/stack_d1_voxel_abs.hdf5 -n 32 --use_gpu --suppress_warnings
    ```

## 仿真可用
仿真使用CUDA_VISIBLE_DEVICES=0有概率报错，可带前缀“MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa”
1.  原始dp的unet的abs
    ```bash
    HYDRA_FULL_ERROR=1 python train_sim.py --config-name=robomimic_train_dp_diffusion_unet_abs task_name=stack_d1 n_demo=400
    ```
2. 原始dp的transformer的abs
    ```bash
    HYDRA_FULL_ERROR=1 python train_sim.py --config-name=robomimic_train_dp_diffusion_transformer_abs task_name=stack_d1 n_demo=400
    ```
3. 原始dp的体素的abs
    ```bash
    HYDRA_FULL_ERROR=1 python train_sim.py --config-name=robomimic_train_dp_diffusion_unet_voxel_abs task_name=stack_d1 n_demo=400
    ```
4.  eqdp的unet的abs
    ```bash
    HYDRA_FULL_ERROR=1 python train_sim.py --config-name=robomimic_train_equi_diffusion_unet_abs task_name=stack_d1 n_demo=400
    ```
5. eqdp的unet的rel
    ```bash
    HYDRA_FULL_ERROR=1 python train_sim.py --config-name=robomimic_train_equi_diffusion_unet_rel task_name=stack_d1 n_demo=400
    ```
## 真实使用
### 原始dp的unet训练
    ```bash
    HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=0 python train.py --config-name=dp_train_diffusion_unet_real_image_workspace task.dataset_path=/home/hy/Desktop/dp_0314/data/pen_0807
    ```
### 原始dp的unet使用
    ```bash
    python eval_real_franka_rawdp_pro.py -i /media/disk7t/outputs/2025.08.08/01.52.33_dp_train_diffusion_unet_image_real_image/checkpoints/latest.ckpt -o data/0808 -ri 192.168.0.168
    ```




## 备注
abs和rel的区别只有两个
1.  shape是[10]还是[7]
2.  abs_action是true还是false














<sup>1</sup>Northeastern Univeristy, <sup>2</sup>Boston Dynamics AI Institute  
Conference on Robot Learning 2024 (Oral)
![](img/equi.gif) | 
## Installation
1.  Install the following apt packages for mujoco:
    ```bash
    sudo apt install -y libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf
    ```
1. Install gfortran (dependancy for escnn) 
    ```bash
    sudo apt install -y gfortran
    ```

1. Install [Mambaforge](https://github.com/conda-forge/miniforge#mambaforge) (strongly recommended) or Anaconda
1. Clone this repo
    ```bash
    git clone https://github.com/pointW/equidiff.git
    cd equidiff
    ```
1. Install environment:
    Use Mambaforge (strongly recommended):
    ```bash
    mamba env create -f conda_environment.yaml
    conda activate equidiff
    ```
    or use Anaconda (not recommended): 
    ```bash
    conda env create -f conda_environment.yaml
    conda activate equidiff
    ```
1. Install mimicgen:
    ```bash
    cd ..
    git clone https://github.com/NVlabs/mimicgen_environments.git
    cd mimicgen_environments
    # This project was developed with Mimicgen v0.1.0. The latest version should work fine, but it is not tested
    git checkout 081f7dbbe5fff17b28c67ce8ec87c371f32526a9
    pip install -e .
    cd ../equidiff
    ```
1. Make sure mujoco version is 2.3.2 (required by mimicgen)
    ```bash
    pip list | grep mujoco
    ```

## Dataset
### Download Dataset
Download dataset from MimicGen's hugging face: https://huggingface.co/datasets/amandlek/mimicgen_datasets/tree/main/core  
Make sure the dataset is kept under `/path/to/equidiff/data/robomimic/datasets/[dataset]/[dataset].hdf5`

### Generating Voxel and Point Cloud Observation

```bash
# Template
python equi_diffpo/scripts/dataset_states_to_obs.py --input data/robomimic/datasets/[dataset]/[dataset].hdf5 --output data/robomimic/datasets/[dataset]/[dataset]_voxel.hdf5 --num_workers=[n_worker]
# Replace [dataset] and [n_worker] with your choices.
# E.g., use 24 workers to generate point cloud and voxel observation for stack_d1
python equi_diffpo/scripts/dataset_states_to_obs.py --input data/robomimic/datasets/stack_d1/stack_d1.hdf5 --output data/robomimic/datasets/stack_d1/stack_d1_voxel.hdf5 --num_workers=24
```

### Convert Action Space in Dataset
The downloaded dataset has a relative action space. To train with absolute action space, the dataset needs to be converted accordingly
```bash
# Template
python equi_diffpo/scripts/robomimic_dataset_conversion.py -i data/robomimic/datasets/[dataset]/[dataset].hdf5 -o data/robomimic/datasets/[dataset]/[dataset]_abs.hdf5 -n [n_worker]
# Replace [dataset] and [n_worker] with your choices.
# E.g., convert stack_d1 (non-voxel) with 12 workers
python equi_diffpo/scripts/robomimic_dataset_conversion.py -i data/robomimic/datasets/stack_d1/stack_d1_voxel.hdf5 -o data/robomimic/datasets/stack_d1/stack_d1_abs.hdf5 -n 12
# E.g., convert stack_d1_voxel (voxel) with 12 workers
python equi_diffpo/scripts/robomimic_dataset_conversion.py -i data/robomimic/datasets/stack_d1/stack_d1_voxel.hdf5 -o data/robomimic/datasets/stack_d1/stack_d1_voxel_abs.hdf5 -n 12
```

## Training with image observation
To train Equivariant Diffusion Policy (with absolute pose control) in Stack D1 task:
```bash
# Make sure you have the non-voxel converted dataset with absolute action space from the previous step 
python train.py --config-name=train_equi_diffusion_unet_abs task_name=stack_d1 n_demo=100
```
To train with relative pose control instead:
```bash
python train.py --config-name=train_equi_diffusion_unet_rel task_name=stack_d1 n_demo=100
```
To train in other tasks, replace `stack_d1` with `stack_three_d1`, `square_d2`, `threading_d2`, `coffee_d2`, `three_piece_assembly_d2`, `hammer_cleanup_d1`, `mug_cleanup_d1`, `kitchen_d1`, `nut_assembly_d0`, `pick_place_d0`, `coffee_preparation_d1`. Notice that the corresponding dataset should be downloaded already. If training absolute pose control, the data conversion is also needed.

To run environments on CPU (to save GPU memory), use `osmesa` instead of `egl` through `MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa`, e.g.,
```bash
MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa python train.py --config-name=train_equi_diffusion_unet_abs task_name=stack_d1
```

Equivariant Diffusion Policy requires around 22G GPU memory to run with batch size of 128 (default). To reduce the GPU usage, consider training with smaller batch size and/or reducing the hidden dimension
```bash
# to train with batch size of 64 and hidden dimension of 64
MUJOCO_GL=osmesa PYOPENGL_PLATTFORM=osmesa python train.py --config-name=train_equi_diffusion_unet_abs task_name=stack_d1 policy.enc_n_hidden=64 dataloader.batch_size=64
```

## Training with voxel observation
To train Equivariant Diffusion Policy (with absolute pose control) in Stack D1 task:
```bash
# Make sure you have the voxel converted dataset with absolute action space from the previous step 
python train.py --config-name=train_equi_diffusion_unet_voxel_abs task_name=stack_d1 n_demo=100
```

## License
This repository is released under the MIT license. See [LICENSE](LICENSE) for additional details.

## Acknowledgement
* Our repo is built upon the origional [Diffusion Policy](https://github.com/real-stanford/diffusion_policy)
* Our ACT baseline is adaped from its [original repo](https://github.com/tonyzhaozh/act)
* Our DP3 baseline is adaped from its [original repo](https://github.com/YanjieZe/3D-Diffusion-Policy)
