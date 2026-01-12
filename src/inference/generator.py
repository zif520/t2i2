"""图像生成器模块"""

import sys
import torch
from diffusers import DDPMScheduler, DDIMScheduler
from typing import Optional, List
from PIL import Image
import numpy as np
from pathlib import Path

# 添加项目根目录到路径
if str(Path(__file__).parent.parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.dit_model import DiTModel
from src.models.vae_model import VAEDecoder
from src.utils.visualization import tensor_to_pil_image, denormalize_image


class ImageGenerator:
    """图像生成器"""
    
    def __init__(
        self,
        model: DiTModel,
        vae_decoder: VAEDecoder,
        text_encoder: Any,  # CLIP 文本编码器
        tokenizer: Any,  # CLIP tokenizer
        scheduler_type: str = "ddpm",
        device: Optional[torch.device] = None,
    ):
        """
        初始化图像生成器
        
        Args:
            model: DiT 模型
            vae_decoder: VAE 解码器
            text_encoder: 文本编码器
            tokenizer: 文本分词器
            scheduler_type: 调度器类型 ("ddpm" 或 "ddim")
            device: 设备
        """
        self.model = model
        self.vae_decoder = vae_decoder
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        
        # 初始化调度器
        self.scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            prediction_type="epsilon",
        )
        
        if scheduler_type == "ddim":
            self.scheduler = DDIMScheduler(
                num_train_timesteps=1000,
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule="scaled_linear",
                prediction_type="epsilon",
            )
        
        # 设置为评估模式
        self.model.eval()
        self.text_encoder.eval()
    
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        height: int = 256,
        width: int = 256,
        seed: Optional[int] = None,
    ) -> Image.Image:
        """
        生成图像
        
        Args:
            prompt: 文本提示
            num_inference_steps: 推理步数
            guidance_scale: 引导强度
            height: 图像高度
            width: 图像宽度
            seed: 随机种子
            
        Returns:
            生成的图像
        """
        if seed is not None:
            torch.manual_seed(seed)
        
        # 设置调度器步数
        self.scheduler.set_timesteps(num_inference_steps)
        
        # 编码文本
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        
        text_outputs = self.text_encoder(text_inputs.input_ids)
        # CLIP 文本编码器返回 last_hidden_state，取平均池化
        text_embeddings = text_outputs.last_hidden_state.mean(dim=1)  # (B, 768)
        
        # 计算潜在空间尺寸
        latent_height = height // 8
        latent_width = width // 8
        
        # 初始化潜在表示（随机噪声）
        latents = torch.randn(
            (1, 4, latent_height, latent_width),
            device=self.device,
        )
        
        # 扩散采样循环
        for t in self.scheduler.timesteps:
            # 预测噪声
            noise_pred = self.model(
                latents,
                t.unsqueeze(0).to(self.device),
                text_embeddings,
            )
            
            # 调度器步骤
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
        
        # 使用 VAE 解码
        images = self.vae_decoder.decode(latents)
        
        # 转换为 PIL 图像
        images = denormalize_image(images[0])
        image = tensor_to_pil_image(images)
        
        return image
    
    @torch.no_grad()
    def generate_batch(
        self,
        prompts: List[str],
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        height: int = 256,
        width: int = 256,
        seed: Optional[int] = None,
    ) -> List[Image.Image]:
        """
        批量生成图像
        
        Args:
            prompts: 文本提示列表
            num_inference_steps: 推理步数
            guidance_scale: 引导强度
            height: 图像高度
            width: 图像宽度
            seed: 随机种子
            
        Returns:
            生成的图像列表
        """
        images = []
        for i, prompt in enumerate(prompts):
            current_seed = seed + i if seed is not None else None
            image = self.generate(
                prompt,
                num_inference_steps,
                guidance_scale,
                height,
                width,
                current_seed,
            )
            images.append(image)
        return images

