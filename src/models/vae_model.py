"""VAE 模型模块（使用预训练 VAE）"""

import os
import torch
import torch.nn as nn
from diffusers import AutoencoderKL
from typing import Optional

# 设置下载超时（通过环境变量）
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")  # 增加到 300 秒


class VAEEncoder:
    """VAE 编码器（将图像编码到潜在空间）"""
    
    def __init__(
        self,
        pretrained_model_name: str = "stabilityai/sd-vae-ft-mse-original",
        use_slicing: bool = True,
    ):
        """
        初始化 VAE 编码器
        
        Args:
            pretrained_model_name: 预训练 VAE 模型名称
            use_slicing: 是否使用切片以节省内存
        """
        # 尝试从 subfolder 加载，如果失败则直接加载
        try:
            self.vae = AutoencoderKL.from_pretrained(
                pretrained_model_name,
                subfolder="vae",
            )
        except Exception:
            # 如果没有 subfolder，直接加载
            self.vae = AutoencoderKL.from_pretrained(
                pretrained_model_name,
            )
        self.vae.eval()
        
        if use_slicing:
            self.vae.enable_slicing()
    
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """
        将图像编码到潜在空间
        
        Args:
            images: 形状为 (B, C, H, W) 的图像张量，值范围 [-1, 1]
            
        Returns:
            潜在表示，形状为 (B, 4, H//8, W//8)
        """
        with torch.no_grad():
            # VAE 编码
            latent_dist = self.vae.encode(images).latent_dist
            latent = latent_dist.sample()
            # 缩放因子
            latent = latent * self.vae.config.scaling_factor
        return latent
    
    def to(self, device: torch.device):
        """移动到指定设备"""
        self.vae = self.vae.to(device)
        return self


class VAEDecoder:
    """VAE 解码器（将潜在表示解码为图像）"""
    
    def __init__(
        self,
        pretrained_model_name: str = "stabilityai/sd-vae-ft-mse-original",
        use_slicing: bool = True,
    ):
        """
        初始化 VAE 解码器
        
        Args:
            pretrained_model_name: 预训练 VAE 模型名称
            use_slicing: 是否使用切片以节省内存
        """
        # 尝试从 subfolder 加载，如果失败则直接加载
        try:
            self.vae = AutoencoderKL.from_pretrained(
                pretrained_model_name,
                subfolder="vae",
            )
        except Exception:
            # 如果没有 subfolder，直接加载
            self.vae = AutoencoderKL.from_pretrained(
                pretrained_model_name,
            )
        self.vae.eval()
        
        if use_slicing:
            self.vae.enable_slicing()
    
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """
        将潜在表示解码为图像
        
        Args:
            latents: 形状为 (B, 4, H, W) 的潜在表示
            
        Returns:
            图像，形状为 (B, 3, H*8, W*8)，值范围 [-1, 1]
        """
        with torch.no_grad():
            # 缩放因子
            latents = latents / self.vae.config.scaling_factor
            # VAE 解码
            images = self.vae.decode(latents).sample
        return images
    
    def to(self, device: torch.device):
        """移动到指定设备"""
        self.vae = self.vae.to(device)
        return self

