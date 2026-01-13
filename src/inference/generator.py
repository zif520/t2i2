"""图像生成器模块"""

import sys
import torch
from diffusers import DDPMScheduler, DDIMScheduler
from typing import Optional, List, Any
from PIL import Image
import numpy as np
from pathlib import Path

# 添加项目根目录到路径
if str(Path(__file__).parent.parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.dit_model import DiTModel
from src.models.vae_model import VAEDecoder
from src.utils.visualization import tensor_to_pil_image


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
        if hasattr(text_outputs, 'pooler_output') and text_outputs.pooler_output is not None:
            text_embeddings = text_outputs.pooler_output
        else:
            text_embeddings = text_outputs.last_hidden_state.mean(dim=1)
        
        # 确保文本嵌入是连续的
        text_embeddings = text_embeddings.contiguous()
        
        # 计算潜在空间尺寸
        latent_height = height // 8
        latent_width = width // 8
        
        # 初始化潜在表示（随机噪声）
        latents = torch.randn(
            (1, 4, latent_height, latent_width),
            device=self.device,
        )
        # 应用 scaling_factor（与训练时编码的缩放一致）
        # 训练时：latent = latent * scaling_factor
        # 推理时：初始噪声也需要按相同方式缩放
        scaling_factor = self.vae_decoder.vae.config.scaling_factor
        latents = latents * scaling_factor
        
        # 扩散采样循环
        for i, t in enumerate(self.scheduler.timesteps):
            # Classifier-Free Guidance (CFG)
            if guidance_scale > 1.0:
                # 条件预测（有文本条件）
                noise_pred_cond = self.model(
                    latents,
                    t.unsqueeze(0).to(self.device),
                    text_embeddings,
                )
                
                # 无条件预测（空文本条件）
                # 创建空文本嵌入（全零）
                uncond_embeddings = torch.zeros_like(text_embeddings)
                noise_pred_uncond = self.model(
                    latents,
                    t.unsqueeze(0).to(self.device),
                    uncond_embeddings,
                )
                
                # CFG: 引导预测 = 无条件预测 + guidance_scale * (条件预测 - 无条件预测)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            else:
                # 不使用 CFG
                noise_pred = self.model(
                    latents,
                    t.unsqueeze(0).to(self.device),
                    text_embeddings,
                )
            
            # 调度器步骤
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
        
        # 使用 VAE 解码
        images = self.vae_decoder.decode(latents)
        
        # VAE 解码后的图像值范围是 [-1, 1]，需要转换到 [0, 1]
        images = (images + 1.0) / 2.0  # 从 [-1, 1] 到 [0, 1]
        images = images.clamp(0, 1)
        
        # 转换为 PIL 图像
        image = tensor_to_pil_image(images[0])
        
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

