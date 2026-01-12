"""训练器模块"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from accelerate import Accelerator
from tqdm import tqdm
from pathlib import Path
from typing import Optional, Dict, Any
import json

# 添加项目根目录到路径
if str(Path(__file__).parent.parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.dit_model import DiTModel
from src.models.vae_model import VAEEncoder
from src.utils.logger import get_logger
from src.utils.config import Config
from src.training.scheduler import get_scheduler
from src.training.loss import DiffusionLoss


class Trainer:
    """训练器类"""
    
    def __init__(
        self,
        model: DiTModel,
        vae_encoder: VAEEncoder,
        text_encoder: Any,  # CLIP 文本编码器
        train_dataloader: DataLoader,
        config: Config,
        accelerator: Optional[Accelerator] = None,
    ):
        """
        初始化训练器
        
        Args:
            model: DiT 模型
            vae_encoder: VAE 编码器
            text_encoder: 文本编码器
            train_dataloader: 训练数据加载器
            config: 配置对象
            accelerator: Accelerate 对象（可选）
        """
        self.model = model
        self.vae_encoder = vae_encoder
        self.text_encoder = text_encoder
        self.train_dataloader = train_dataloader
        self.config = config
        
        # 初始化 Accelerator
        if accelerator is None:
            self.accelerator = Accelerator(
                mixed_precision=config.training.get("mixed_precision", "fp16"),
                gradient_accumulation_steps=config.training.get("gradient_accumulation_steps", 1),
            )
        else:
            self.accelerator = accelerator
        
        # 初始化调度器
        self.noise_scheduler = get_scheduler(
            scheduler_type="ddpm",
            num_train_timesteps=config.scheduler.get("num_train_timesteps", 1000),
            beta_start=config.scheduler.get("beta_start", 0.00085),
            beta_end=config.scheduler.get("beta_end", 0.012),
            beta_schedule=config.scheduler.get("beta_schedule", "scaled_linear"),
            prediction_type=config.scheduler.get("prediction_type", "epsilon"),
        )
        
        # 初始化优化器
        optimizer_config = config.optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.training.get("learning_rate", 1e-4),
            betas=optimizer_config.get("betas", [0.9, 0.999]),
            weight_decay=optimizer_config.get("weight_decay", 0.01),
        )
        
        # 初始化学习率调度器
        lr_scheduler_config = config.lr_scheduler
        warmup_steps = lr_scheduler_config.get("warmup_steps", 500)
        total_steps = len(train_dataloader) * config.training.get("num_epochs", 50)
        
        if lr_scheduler_config.get("type", "cosine") == "cosine":
            self.lr_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps - warmup_steps,
            )
        else:
            self.lr_scheduler = None
        
        # 初始化损失函数
        self.criterion = DiffusionLoss(loss_type="mse")
        
        # 准备模型和数据
        self.model, self.optimizer, self.train_dataloader = self.accelerator.prepare(
            self.model, self.optimizer, self.train_dataloader
        )
        
        # 优化：将 VAE 和文本编码器也移到 GPU（如果还没移动）
        if hasattr(self.vae_encoder, 'vae'):
            self.vae_encoder.vae = self.vae_encoder.vae.to(self.accelerator.device)
        if hasattr(self.text_encoder, 'to'):
            self.text_encoder = self.text_encoder.to(self.accelerator.device)
        
        # 日志记录器
        self.logger = get_logger()
        
        # 训练状态
        self.global_step = 0
        self.current_epoch = 0
        
        # EMA（指数移动平均）
        self.use_ema = config.training.get("use_ema", False)
        if self.use_ema:
            self.ema_decay = config.training.get("ema_decay", 0.9999)
            self.ema_model = None
            self._init_ema()
    
    def _init_ema(self):
        """初始化 EMA 模型"""
        self.ema_model = type(self.model)(
            **self.config.model
        ).to(self.accelerator.device)
        self.ema_model.load_state_dict(self.model.state_dict())
        self.ema_model.eval()
    
    def _update_ema(self):
        """更新 EMA 模型"""
        if not self.use_ema:
            return
        
        with torch.no_grad():
            for ema_param, model_param in zip(
                self.ema_model.parameters(),
                self.model.parameters()
            ):
                ema_param.data.mul_(self.ema_decay).add_(
                    model_param.data, alpha=1 - self.ema_decay
                )
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        执行一个训练步骤
        
        Args:
            batch: 批次数据
            
        Returns:
            损失字典
        """
        self.model.train()
        
        # 获取数据（使用非阻塞传输，已在 DataLoader 中 pin_memory）
        pixel_values = batch["pixel_values"].to(self.accelerator.device, non_blocking=True)
        input_ids = batch["input_ids"].to(self.accelerator.device, non_blocking=True)
        
        # 使用统一的 autocast 上下文（优化：减少上下文切换开销）
        with torch.amp.autocast(device_type="cuda", enabled=self.accelerator.mixed_precision != "no"):
            # VAE 编码图像到潜在空间
            with torch.no_grad():
                latents = self.vae_encoder.encode(pixel_values)
            
            # 文本编码器编码文本
            with torch.no_grad():
                text_outputs = self.text_encoder(input_ids)
                # CLIP 文本编码器返回 last_hidden_state，取平均池化
                if hasattr(text_outputs, 'pooler_output') and text_outputs.pooler_output is not None:
                    text_embeddings = text_outputs.pooler_output
                else:
                    text_embeddings = text_outputs.last_hidden_state.mean(dim=1)
            
            # 确保文本嵌入是连续的（避免编译模式下的问题）
            text_embeddings = text_embeddings.contiguous()
            
            # 采样时间步（在 GPU 上生成，避免 CPU-GPU 传输）
            timesteps = torch.randint(
                0,
                self.noise_scheduler.config.num_train_timesteps,
                (latents.shape[0],),
                device=latents.device,
                dtype=torch.long,
            )
            
            # 添加噪声（使用 torch.randn_like 在 GPU 上生成）
            noise = torch.randn_like(latents)
            noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
            
            # 预测噪声（在 autocast 上下文中）
            pred_noise = self.model(noisy_latents, timesteps, text_embeddings)
            
            # 计算损失（在 autocast 上下文中）
            loss = self.criterion(pred_noise, noise)
        
        # 反向传播
        self.accelerator.backward(loss)
        
        # 梯度裁剪
        if self.config.training.get("max_grad_norm", 0) > 0:
            self.accelerator.clip_grad_norm_(
                self.model.parameters(),
                self.config.training.get("max_grad_norm", 1.0),
            )
        
        # 更新参数
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)  # 优化：set_to_none 更快且节省内存
        
        # 定期清理缓存以减少内存碎片（每100步，减少频率以提升性能）
        if self.global_step % 100 == 0:
            torch.cuda.empty_cache()
        
        # 更新学习率
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        
        # 更新 EMA
        self._update_ema()
        
        return {"loss": loss.item()}
    
    def train(self):
        """执行训练循环"""
        num_epochs = self.config.training.get("num_epochs", 50)
        save_steps = self.config.training.get("save_steps", 500)
        logging_steps = self.config.training.get("logging_steps", 50)
        output_dir = Path(self.config.training.get("output_dir", "./outputs"))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"开始训练，共 {num_epochs} 个 epoch")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            self.model.train()
            
            progress_bar = tqdm(
                self.train_dataloader,
                desc=f"Epoch {epoch+1}/{num_epochs}",
                disable=not self.accelerator.is_local_main_process,
            )
            
            epoch_loss = 0.0
            num_steps = 0
            
            for batch in progress_bar:
                # 训练步骤
                loss_dict = self.train_step(batch)
                loss = loss_dict["loss"]
                epoch_loss += loss
                num_steps += 1
                self.global_step += 1
                
                # 更新进度条
                progress_bar.set_postfix({"loss": f"{loss:.4f}"})
                
                # 记录日志
                if self.global_step % logging_steps == 0:
                    self.logger.info(
                        f"Step {self.global_step}, Loss: {loss:.4f}, "
                        f"LR: {self.optimizer.param_groups[0]['lr']:.2e}"
                    )
                
                # 保存检查点
                if self.global_step % save_steps == 0:
                    self.save_checkpoint(output_dir / f"checkpoint-{self.global_step}")
            
            # Epoch 结束
            avg_loss = epoch_loss / num_steps
            self.logger.info(f"Epoch {epoch+1} 完成，平均损失: {avg_loss:.4f}")
            
            # 保存 epoch 检查点
            self.save_checkpoint(output_dir / f"checkpoint-epoch-{epoch+1}")
        
        self.logger.info("训练完成！")
        # 保存最终模型
        self.save_checkpoint(output_dir / "final")
    
    def save_checkpoint(self, checkpoint_dir: Path):
        """
        保存检查点
        
        Args:
            checkpoint_dir: 检查点目录
        """
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存模型
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        torch.save(
            unwrapped_model.state_dict(),
            checkpoint_dir / "model.pt",
        )
        
        # 保存优化器
        torch.save(
            self.optimizer.state_dict(),
            checkpoint_dir / "optimizer.pt",
        )
        
        # 保存 EMA 模型
        if self.use_ema and self.ema_model is not None:
            torch.save(
                self.ema_model.state_dict(),
                checkpoint_dir / "ema_model.pt",
            )
        
        # 保存训练状态
        training_state = {
            "global_step": self.global_step,
            "current_epoch": self.current_epoch,
        }
        with open(checkpoint_dir / "training_state.json", "w") as f:
            json.dump(training_state, f, indent=2)
        
        self.logger.info(f"检查点已保存到: {checkpoint_dir}")
    
    def load_checkpoint(self, checkpoint_dir: Path):
        """
        加载检查点
        
        Args:
            checkpoint_dir: 检查点目录
        """
        checkpoint_dir = Path(checkpoint_dir)
        
        # 加载模型
        model_path = checkpoint_dir / "model.pt"
        if model_path.exists():
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            unwrapped_model.load_state_dict(torch.load(model_path))
            self.logger.info(f"模型已从 {model_path} 加载")
        
        # 加载优化器
        optimizer_path = checkpoint_dir / "optimizer.pt"
        if optimizer_path.exists():
            self.optimizer.load_state_dict(torch.load(optimizer_path))
            self.logger.info(f"优化器已从 {optimizer_path} 加载")
        
        # 加载训练状态
        state_path = checkpoint_dir / "training_state.json"
        if state_path.exists():
            with open(state_path, "r") as f:
                training_state = json.load(f)
            self.global_step = training_state.get("global_step", 0)
            self.current_epoch = training_state.get("current_epoch", 0)
            self.logger.info(f"训练状态已从 {state_path} 加载")

