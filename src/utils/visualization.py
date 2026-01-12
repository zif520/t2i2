"""可视化工具模块"""

import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Optional, Tuple

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def save_image_grid(
    images: List[Image.Image],
    output_path: str,
    ncols: int = 4,
    padding: int = 2,
) -> None:
    """
    保存图像网格
    
    Args:
        images: 图像列表
        output_path: 输出路径
        ncols: 列数
        padding: 图像间距
    """
    n_images = len(images)
    nrows = (n_images + ncols - 1) // ncols
    
    # 计算网格尺寸
    img_width, img_height = images[0].size
    grid_width = ncols * img_width + (ncols - 1) * padding
    grid_height = nrows * img_height + (nrows - 1) * padding
    
    # 创建网格图像
    grid = Image.new("RGB", (grid_width, grid_height), color=(255, 255, 255))
    
    # 放置图像
    for idx, img in enumerate(images):
        row = idx // ncols
        col = idx % ncols
        x = col * (img_width + padding)
        y = row * (img_height + padding)
        grid.paste(img, (x, y))
    
    # 保存
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(output_path)
    print(f"图像网格已保存到: {output_path}")


def plot_training_curves(
    losses: List[float],
    output_path: str,
    title: str = "Training Loss",
) -> None:
    """
    绘制训练曲线
    
    Args:
        losses: 损失值列表
        output_path: 输出路径
        title: 图表标题
    """
    if not HAS_MATPLOTLIB:
        print("警告: matplotlib 未安装，无法绘制训练曲线")
        return
    
    plt.figure(figsize=(10, 6))
    plt.plot(losses, label="Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"训练曲线已保存到: {output_path}")


def tensor_to_pil_image(tensor: torch.Tensor) -> Image.Image:
    """
    将 PyTorch 张量转换为 PIL 图像
    
    Args:
        tensor: 形状为 (C, H, W) 或 (B, C, H, W) 的张量，值范围 [0, 1]
        
    Returns:
        PIL 图像或图像列表
    """
    if tensor.dim() == 4:
        # 批次维度
        images = []
        for i in range(tensor.shape[0]):
            img = tensor_to_pil_image(tensor[i])
            images.append(img)
        return images
    elif tensor.dim() == 3:
        # 单张图像
        tensor = tensor.clamp(0, 1)
        array = tensor.cpu().numpy()
        array = (array * 255).astype(np.uint8)
        
        # 转换维度顺序 (C, H, W) -> (H, W, C)
        if array.shape[0] == 3 or array.shape[0] == 1:
            array = array.transpose(1, 2, 0)
        
        if array.shape[2] == 1:
            array = array.squeeze(2)
            return Image.fromarray(array, mode="L")
        else:
            return Image.fromarray(array, mode="RGB")
    else:
        raise ValueError(f"不支持的张量维度: {tensor.dim()}")


def pil_image_to_tensor(image: Image.Image) -> torch.Tensor:
    """
    将 PIL 图像转换为 PyTorch 张量
    
    Args:
        image: PIL 图像
        
    Returns:
        形状为 (C, H, W) 的张量，值范围 [0, 1]
    """
    array = np.array(image).astype(np.float32) / 255.0
    
    # 转换维度顺序 (H, W, C) -> (C, H, W)
    if array.ndim == 2:
        array = array[np.newaxis, :, :]
    else:
        array = array.transpose(2, 0, 1)
    
    return torch.from_numpy(array)

