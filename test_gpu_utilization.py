"""测试 GPU 利用率"""
import torch
import time
from torch.utils.data import DataLoader
from src.data.dataset import TextImageDataset
from src.utils.config import load_config

config = load_config("configs/train_config.yaml")

# 创建数据集
dataset = TextImageDataset(
    dataset_name="custom",
    dataset_path="./data/test_data",
    image_size=256,
    num_samples=100,
    is_train=True,
)

# 测试不同配置
configs = [
    {"batch_size": 2, "num_workers": 2, "name": "原始配置"},
    {"batch_size": 16, "num_workers": 8, "name": "优化配置"},
]

for cfg in configs:
    print(f"\n测试 {cfg['name']}: batch_size={cfg['batch_size']}, num_workers={cfg['num_workers']}")
    dataloader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=True,
        prefetch_factor=2,
    )
    
    # 模拟训练步骤
    start_time = time.time()
    for i, batch in enumerate(dataloader):
        if i >= 10:  # 只测试前 10 个批次
            break
        # 模拟 GPU 操作
        pixel_values = batch["pixel_values"].cuda()
        _ = pixel_values.mean()  # 简单计算
    end_time = time.time()
    
    print(f"  处理 10 个批次耗时: {end_time - start_time:.2f} 秒")
    print(f"  平均每批次: {(end_time - start_time) / 10:.3f} 秒")
