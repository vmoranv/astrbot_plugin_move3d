#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
综合测试脚本
用于测试 jb、jb3d 和 c3d 命令，并遍历不同参数组合以找到最佳效果
"""

import os
import sys
import cv2
import numpy as np
from image_processor import sketch_effect, line_art_effect, sketch_3d_effect, c3d_effect
import argparse

def test_jb_parameters(input_image_path, output_dir='jb_test_results'):
    """
    测试 jb 命令的不同参数组合
    
    参数:
    input_image_path: 输入图像路径
    output_dir: 输出目录
    """
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print("测试 jb 命令的参数组合...")
    
    # 遍历 blur_kernel_size 的值 (1-51)，步长为2（确保是奇数）
    for blur_kernel_size in range(1, 52, 2):
        try:
            output_path = os.path.join(output_dir, f'jb_kernel_{blur_kernel_size}.png')
            # 将模糊核大小参数传递给 sketch_effect 函数
            sketch_effect(input_image_path, output_path, blur_kernel_size)
            print(f"已生成: {output_path} (模糊核大小={blur_kernel_size})")
        except Exception as e:
            print(f"处理参数 (模糊核大小={blur_kernel_size}) 时出错: {e}")

def test_jb3d_parameters(input_image_path, output_dir='jb3d_test_results'):
    """
    测试 jb3d 命令的不同参数组合
    
    参数:
    input_image_path: 输入图像路径
    output_dir: 输出目录
    """
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print("测试 jb3d 命令的参数组合...")
    
    # 遍历 threshold1 和 threshold2 的所有组合 (0-255)，步长为10
    for threshold1 in range(0, 256, 10):
        for threshold2 in range(0, 256, 10):
            # 确保阈值2大于阈值1
            if threshold2 <= threshold1:
                continue
            
            try:
                output_path = os.path.join(output_dir, f'jb3d_t1_{threshold1}_t2_{threshold2}.png')
                # 将 Canny 参数传递给 sketch_3d_effect 函数
                sketch_3d_effect(input_image_path, output_path, threshold1, threshold2)
                print(f"已生成: {output_path} (阈值1={threshold1}, 阈值2={threshold2})")
            except Exception as e:
                print(f"处理参数组合 (阈值1={threshold1}, 阈值2={threshold2}) 时出错: {e}")
    
def test_c3d_parameters(input_image_path, output_dir='c3d_test_results'):
    """
    测试 c3d 命令的不同参数组合
    
    参数:
    input_image_path: 输入图像路径
    output_dir: 输出目录
    """
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print("测试 c3d 命令的参数组合...")
    
    # 测试不同的对比度值
    for contrast in [0.8, 1.0, 1.2, 1.5, 2.0]:
        try:
            output_path = os.path.join(output_dir, f'c3d_contrast_{contrast}.png')
            c3d_effect(input_image_path, output_path, contrast=contrast)
            print(f"已生成: {output_path} (对比度={contrast})")
        except Exception as e:
            print(f"处理参数 (对比度={contrast}) 时出错: {e}")
    
    # 测试不同的阴影放大倍数
    for shadow_scale in [1.05, 1.1, 1.2, 1.3]:
        try:
            output_path = os.path.join(output_dir, f'c3d_shadow_scale_{shadow_scale}.png')
            c3d_effect(input_image_path, output_path, shadow_scale=shadow_scale)
            print(f"已生成: {output_path} (阴影放大={shadow_scale})")
        except Exception as e:
            print(f"处理参数 (阴影放大={shadow_scale}) 时出错: {e}")
    
    # 测试不同的阴影模糊强度
    for shadow_blur in [1.5, 2.0, 2.5, 3.0, 5.0]:
        try:
            output_path = os.path.join(output_dir, f'c3d_shadow_blur_{shadow_blur}.png')
            c3d_effect(input_image_path, output_path, shadow_blur=shadow_blur)
            print(f"已生成: {output_path} (阴影模糊={shadow_blur})")
        except Exception as e:
            print(f"处理参数 (阴影模糊={shadow_blur}) 时出错: {e}")
    
    # 测试不同的阴影偏移值
    for shadow_offset_x in [-20, -15, -10, -5, 0, 5, 10]:
        try:
            output_path = os.path.join(output_dir, f'c3d_offset_x_{shadow_offset_x}.png')
            c3d_effect(input_image_path, output_path, shadow_offset_x=shadow_offset_x)
            print(f"已生成: {output_path} (阴影偏移X={shadow_offset_x})")
        except Exception as e:
            print(f"处理参数 (阴影偏移X={shadow_offset_x}) 时出错: {e}")


def test_with_real_image(image_path):
    """
    使用真实图像进行测试
    """
    if not os.path.exists(image_path):
        print(f"图像文件不存在: {image_path}")
        return
    
    print(f"使用图像进行测试: {image_path}")
    
    # 测试所有命令
    test_jb_parameters(image_path, 'jb_test_results_real')
    test_jb3d_parameters(image_path, 'jb3d_test_results_real')
    test_c3d_parameters(image_path, 'c3d_test_results_real')

def main():
    parser = argparse.ArgumentParser(description='测试 jb、jb3d 和 c3d 命令')
    parser.add_argument('--input', '-i', type=str, required=True, help='输入图像路径')
    parser.add_argument('--test-type', '-t', type=str, choices=['jb', 'jb3d', 'c3d', 'all'],
                       default='all', help='测试类型: jb、jb3d、c3d 或 all')
    
    args = parser.parse_args()
    
    # 检查输入图像是否存在
    if not os.path.exists(args.input):
        print(f"错误: 输入图像不存在: {args.input}")
        sys.exit(1)
        
    input_image_path = args.input
    print(f"使用提供的图像: {input_image_path}")
    
    # 根据测试类型执行相应的测试
    if args.test_type == 'jb':
        print("开始测试 jb 命令...")
        test_jb_parameters(input_image_path)
    elif args.test_type == 'jb3d':
        print("开始测试 jb3d 命令...")
        test_jb3d_parameters(input_image_path)
    elif args.test_type == 'c3d':
        print("开始测试 c3d 命令...")
        test_c3d_parameters(input_image_path)
    elif args.test_type == 'all':
        print("开始测试所有命令...")
        test_jb_parameters(input_image_path)
        test_jb3d_parameters(input_image_path)
        test_c3d_parameters(input_image_path)
    
    print("测试完成！请查看生成的结果图像以选择最佳参数。")

if __name__ == '__main__':
    main()