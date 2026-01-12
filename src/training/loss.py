"""损失函数模块"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def diffusion_loss(
    pred_noise: torch.Tensor,
    target_noise: torch.Tensor,
    loss_type: str = "mse",
) -> torch.Tensor:
    """
    扩散模型损失函数
    
    Args:
        pred_noise: 预测的噪声，形状为 (B, C, H, W)
        target_noise: 目标噪声，形状为 (B, C, H, W)
        loss_type: 损失类型 ("mse" 或 "l1")
        
    Returns:
        损失值
    """
    if loss_type == "mse":
        return F.mse_loss(pred_noise, target_noise)
    elif loss_type == "l1":
        return F.l1_loss(pred_noise, target_noise)
    else:
        raise ValueError(f"不支持的损失类型: {loss_type}")


class DiffusionLoss(nn.Module):
    """扩散模型损失函数（可学习的）"""
    
    def __init__(self, loss_type: str = "mse"):
        super().__init__()
        self.loss_type = loss_type
    
    def forward(
        self,
        pred_noise: torch.Tensor,
        target_noise: torch.Tensor,
    ) -> torch.Tensor:
        return diffusion_loss(pred_noise, target_noise, self.loss_type)



