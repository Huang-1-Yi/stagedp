import zarr
import numpy as np
import os
import json
import pickle
from pathlib import Path
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs, JpegXl
import shutil
import cv2
import av
from tqdm import tqdm
import multiprocessing
from concurrent.futures import ThreadPoolExecutor

# 注册图像编解码器
register_codecs()

def create_zarr_zip(output_path, source_zarr_path, target_resolution=(224, 224), 
                   compression_level=99, num_workers=None):
    """
    创建与代码0-7相同格式的Zarr zip文件
    
    参数:
    output_path: 输出Zarr zip文件路径
    source_zarr_path: 源Zarr数据集路径
    target_resolution: 目标图像分辨率 (高度, 宽度)
    compression_level: JpegXL压缩级别 (1-99)
    num_workers: 并行处理的工作线程数
    """
    # 确保输出目录存在
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 创建临时目录用于构建Zarr存储
    temp_dir = Path("temp_zarr_build")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True)
    
    try:
        # 打开源Zarr数据集
        source_store = zarr.open(str(source_zarr_path), mode='r')
        
        # 获取episode_ends
        episode_ends = source_store['meta']['episode_ends'][:]
        
        # 创建新的Zarr存储 - 使用直接方法
        store = zarr.DirectoryStore(str(temp_dir))
        root = zarr.group(store=store)
        
        # 创建meta组和episode_ends数组
        meta_group = root.create_group('meta')
        meta_group.array(
            'episode_ends', 
            episode_ends, 
            dtype='int64',
            chunks=(len(episode_ends),)
        )
        
        # 创建data组
        data_group = root.create_group('data')
        
        # 定义压缩器
        img_compressor = JpegXl(level=compression_level, numthreads=1)
        
        # 处理所有相机数据
        print("处理相机数据...")
        camera_keys = [key for key in source_store['data'].keys() if key.startswith('camera')]
        
        for camera_key in camera_keys:
            if not camera_key.endswith('_rgb'):
                continue
                
            camera_array = source_store['data'][camera_key]
            T, H, W, C = camera_array.shape
            
            # 创建目标图像数组
            target_shape = (T, target_resolution[0], target_resolution[1], 3)
            img_array = data_group.zeros(
                name=camera_key,
                shape=target_shape,
                chunks=(1, target_resolution[0], target_resolution[1], 3),
                compressor=img_compressor,
                dtype='uint8'
            )
            
            # 调整图像大小 - 使用多线程并行处理
            def process_frame(i):
                img = camera_array[i]
                resized_img = cv2.resize(img, (target_resolution[1], target_resolution[0]))
                return i, resized_img
            
            # 设置并行工作线程数
            if num_workers is None:
                num_workers = multiprocessing.cpu_count()
            
            print(f"调整{camera_key}图像大小 ({T}帧, {num_workers}线程)...")
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                results = list(tqdm(executor.map(process_frame, range(T)), total=T))
                
            # 按顺序存储结果
            for i, img in results:
                img_array[i] = img
        
        # 复制所有非图像数据
        print("复制非图像数据...")
        for key in source_store['data'].keys():
            if key in camera_keys:
                continue  # 跳过已处理的相机数据
                
            if key == 'meta':
                continue  # 跳过meta组
                
            array = source_store['data'][key]
            
            # 处理不同形状的数组
            if array.shape:
                chunk_size = min(1000, array.shape[0])
                chunks = (chunk_size,) + array.shape[1:]
            else:
                chunks = None
                
            # 创建数组并复制数据
            data_group.array(
                name=key,
                data=array[:],
                chunks=chunks,
                dtype=array.dtype
            )
            print(f"复制: data/{key} (形状: {array.shape}, 类型: {array.dtype})")
        
        # 复制其他组（如calibration, dataset_plan等）
        print("复制其他组数据...")
        for group_name in source_store.keys():
            if group_name == 'data' or group_name == 'meta':
                continue  # 已处理
                
            source_group = source_store[group_name]
            dest_group = root.create_group(group_name)
            
            # 递归复制组内容
            def copy_group(src, dest):
                for key in src.keys():
                    item = src[key]
                    if isinstance(item, zarr.Array):
                        dest.array(
                            name=key,
                            data=item[:],
                            chunks=item.chunks,
                            dtype=item.dtype
                        )
                    elif isinstance(item, zarr.Group):
                        new_dest = dest.create_group(key)
                        copy_group(item, new_dest)
            
            copy_group(source_group, dest_group)
            print(f"复制组: {group_name}")
        
        # 保存到zip文件
        print("保存到zip文件...")
        with zarr.ZipStore(str(output_path), mode='w') as zip_store:
            # 递归复制整个存储到zip文件
            def copy_to_zip(src, dest):
                for key in src.keys():
                    item = src[key]
                    if isinstance(item, zarr.Array):
                        dest.array(
                            name=key,
                            data=item[:],
                            chunks=item.chunks,
                            dtype=item.dtype
                        )
                    elif isinstance(item, zarr.Group):
                        new_dest = dest.create_group(key)
                        copy_to_zip(item, new_dest)
            
            # 创建目标存储并复制数据
            dest_root = zarr.group(store=zip_store)
            copy_to_zip(root, dest_root)
        
        print(f"成功创建Zarr zip文件: {output_path}")
        return True
    
    except Exception as e:
        print(f"创建Zarr zip文件时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # 清理临时目录
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

# 示例用法
if __name__ == "__main__":
    # 源Zarr数据集路径
    source_zarr_path = "/home/hy/Desktop/dp_0314/data/cup_and_saucer0805/replay_buffer_new.zarr"
    
    # 目标Zarr zip文件路径
    output_path = "/home/hy/Desktop/dp_0314/data/cup_and_saucer0805/replay_buffer.zarr.zip"
    
    # 目标图像分辨率
    target_resolution = (224, 224)  # 高度, 宽度
    
    # 创建Zarr zip文件
    create_zarr_zip(
        output_path=output_path,
        source_zarr_path=source_zarr_path,
        target_resolution=target_resolution,
        compression_level=99,
        num_workers=multiprocessing.cpu_count()  # 使用所有可用核心
    )