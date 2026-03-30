#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验室安全识别系统 - 模型训练脚本
使用YOLOv8训练自定义数据集（GPU版）
"""

import os
import sys
import torch
from pathlib import Path
from ultralytics import YOLO
import yaml

def train_model():
    """训练YOLOv8模型"""
    print("开始训练YOLOv8模型...")
    
    # 检查GPU
    if torch.cuda.is_available():
        print(f"使用GPU训练: {torch.cuda.get_device_name(0)}")
        device = '0'  # 使用第一块GPU
    else:
        print("没有检测到GPU，将使用CPU训练（会很慢）")
        device = 'cpu'
    
    # 设置项目路径
    project_root = Path(__file__).parent
    data_yaml_path = project_root / "yolov8" / "data.yaml"
    
    # 检查数据配置文件
    if not data_yaml_path.exists():
        print(f"错误：找不到数据配置文件 {data_yaml_path}")
        return False
    
    # 读取数据配置
    with open(data_yaml_path, 'r', encoding='utf-8') as f:
        data_config = yaml.safe_load(f)
    
    print(f"数据集配置：")
    print(f"  类别数量: {data_config['nc']}")
    print(f"  类别名称: {data_config['names']}")
    
    # 创建模型输出目录
    output_dir = project_root / "models"
    output_dir.mkdir(exist_ok=True)
    
    try:
        # 加载预训练模型
        # model = YOLO('yolov8m.pt') # 原版
        
        # 使用自定义的 P2 结构模型 (专门针对小目标优化)
        # 注意：因为网络结构变了，不能直接加载 yolov8m.pt 的权重，只能加载 yolov8m.yaml 结构
        # 但为了利用预训练权重，我们通常先加载 .yaml 构建结构，再尝试加载 .pt (部分匹配)
        # 或者直接从头训练 (如果数据量够大)
        # 这里我们采用 "加载 .yaml 构建新网络 + 迁移 yolov8m.pt 权重" 的方式
        
        # print("构建 YOLOv8-P2 小目标检测模型...")
        # p2_yaml_path = project_root / "yolov8" / "yolov8-p2.yaml"
        # model = YOLO(str(p2_yaml_path)) # 从 yaml 构建
        
        # 切换到最新的 YOLO26-P2-m 模型 (基于官方 YOLO26 架构 + P2层)
        print("构建 YOLO26-P2-m (NMS-Free, P2 Layer) 模型...")
        
        # 使用新创建的 yolo26-p2.yaml
        p2_yaml_path = project_root / "yolov8" / "yolo26-p2.yaml"
        
        try:
            # 构建模型结构
            model = YOLO(str(p2_yaml_path))
            
            # 尝试加载预训练权重
            # 优先尝试 YOLO26m (如果已发布)
            try:
                model.load('yolo26n.pt')
                print("成功加载 YOLO26n 权重!")
            except:
                print("YOLO26m.pt 未找到，尝试加载 yolo11m.pt 作为初始化权重...")
                # YOLO11 和 YOLO26 架构可能相似 (C3k2)，尝试加载以加速收敛
                try:
                    model.load('yolo11m.pt')
                except Exception as e:
                    print(f"权重加载失败 ({e})，将从头训练...")
                
        except Exception as e:
            print(f"构建失败 ({e})，回退到官方 yolo11m.pt...")
            model = YOLO('yolo11m.pt')
        
        # 尝试加载预训练权重 (可选，但这会显著加快收敛)
        # model.load('yolov8m.pt') 
        
        # 开始训练 - 使用GPU
        results = model.train(
            data=str(data_yaml_path),
            epochs=300,
            imgsz=640,
            batch=8,       # P2层消耗显存较大，降低 batch size 以防 OOM
            device=device,  # 使用GPU
            project=str(output_dir),
            name='lab_safety_detection_yolo26_p2', # 修改名字以区分
            
            # YOLO26 特定优化
            # close_mosaic=20, 
            
            save=True,
            save_period=20,
            patience=50,
            workers=4,      # 降低 worker 数量以防止共享内存不足导致的崩溃
            verbose=True,
            plots=True,
            # 优化超参数
            lr0=0.01,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3,
            # 数据增强参数 (针对小目标优化)
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=0.0,    # 关闭旋转，保持重力方向 (实验室内人通常是直立的)
            translate=0.1,  # 减小平移幅度，避免小目标被移出边界
            scale=0.5,      # 调整缩放比例
            shear=0.0,
            perspective=0.0,
            flipud=0.0,
            fliplr=0.5,
            mosaic=1.0,     # 保持 Mosaic
            mixup=0.0,      # 关闭 Mixup (User Requested)
            copy_paste=0.0, # 如果没有分割数据，设为0
            # 损失函数权重优化
            box=7.5,        # 保持 Box 损失权重
            cls=0.5,        # 保持 Class 损失权重
            dfl=1.5,        # Distribution Focal Loss
        )
        
        print("训练完成！")
        
        # 获取最佳模型路径
        best_model_path = output_dir / "lab_safety_detection" / "weights" / "best.pt"
        if best_model_path.exists():
            print(f"最佳模型已保存到: {best_model_path}")
            
            # 复制到项目根目录
            import shutil
            target_path = project_root / "best.pt"
            shutil.copy2(best_model_path, target_path)
            print(f"模型已复制到: {target_path}")
            
            return True
        else:
            print("警告：未找到训练好的模型文件")
            return False
            
    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        return False

def validate_model():
    """验证训练好的模型 - 在验证集上验证"""
    project_root = Path(__file__).parent
    model_path = project_root / "best.pt"
    
    if not model_path.exists():
        print("错误：找不到训练好的模型文件")
        return False
    
    try:
        # 加载模型
        model = YOLO(str(model_path))
        
        # 读取数据配置，明确告诉用户在验证集上验证
        data_yaml_path = project_root / "yolov8" / "data.yaml"
        with open(data_yaml_path, 'r', encoding='utf-8') as f:
            data_config = yaml.safe_load(f)
        
        print(f"\n{'='*60}")
        print("在验证集上进行模型验证...")
        print(f"验证集路径: {data_config.get('val', '未指定')}")
        print(f"{'='*60}")
        
        # 使用GPU验证
        device = '0' if torch.cuda.is_available() else 'cpu'
        
        # 在验证集上验证
        results = model.val(
            data=str(data_yaml_path),
            device=device  # 使用GPU验证
        )
        
        print("\n验证完成！")
        print(f"mAP50: {results.box.map50:.4f} ({results.box.map50*100:.1f}%)")
        print(f"mAP50-95: {results.box.map:.4f}")
        print(f"精确度: {results.box.mp:.4f}")
        print(f"召回率: {results.box.mr:.4f}")
        
        # 简单性能评估
        if results.box.map50 >= 0.70:
            print("\n模型性能优秀！")
        elif results.box.map50 >= 0.50:
            print("\n模型性能中等，可以考虑进一步优化")
        else:
            print("\n模型性能较差，建议重新训练或调整参数")
        
        return True
        
    except Exception as e:
        print(f"模型验证出现错误: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("实验室安全识别系统 - 模型训练（GPU版）")
    print("=" * 60)
    
    # 训练模型
    if train_model():
        print("\n" + "=" * 60)
        print("开始验证模型...")
        validate_model()
    else:
        print("训练失败，请检查错误信息")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("训练完成！现在可以使用训练好的模型进行检测了。")
    print("=" * 60)