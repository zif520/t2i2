"""训练脚本"""

import argparse
import sys
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import CLIPTextModel, CLIPTokenizer

# 添加项目根目录到路径，支持直接运行脚本
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.dit_model import DiTModel
from src.models.vae_model import VAEEncoder
from src.data.dataset import TextImageDataset
from src.training.trainer import Trainer
from src.utils.config import load_config
from src.utils.logger import setup_logger


def main():
    parser = argparse.ArgumentParser(description="训练 DiT 模型")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_config.yaml",
        help="配置文件路径",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="恢复训练的检查点路径",
    )
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 设置日志
    logger = setup_logger(
        log_file=Path(config.training.get("output_dir", "./outputs")) / "train.log"
    )
    logger.info("开始训练 DiT 模型")
    logger.info(f"配置: {config.to_dict()}")
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    
    # 加载文本编码器
    logger.info("加载文本编码器...")
    text_encoder_name = config.text_encoder.get("pretrained_model_name", "openai/clip-vit-base-patch32")
    text_encoder = CLIPTextModel.from_pretrained(text_encoder_name)
    text_encoder = text_encoder.to(device)
    text_encoder.eval()
    
    # 加载 VAE 编码器
    logger.info("加载 VAE 编码器...")
    vae_config = config.vae or {}
    vae_encoder = VAEEncoder(
        pretrained_model_name=vae_config.get("pretrained_model_name", "stabilityai/sd-vae-ft-mse"),
        use_slicing=vae_config.get("use_slicing", True),
    )
    vae_encoder = vae_encoder.to(device)
    
    # 创建数据集
    logger.info("创建数据集...")
    dataset = TextImageDataset(
        dataset_name=config.data.get("dataset_name", "coco"),
        dataset_path=config.data.get("dataset_path"),
        image_size=config.data.get("image_size", 256),
        tokenizer_name=text_encoder_name,
        max_length=77,
        num_samples=config.data.get("num_samples"),
        is_train=True,
    )
    
    # 创建数据加载器（深度优化 GPU 利用率）
    batch_size = config.training.get("batch_size", 4)
    num_workers = config.training.get("num_workers", 8)  # 增加数据加载线程
    prefetch_factor = config.training.get("prefetch_factor", 4)  # 增加预取因子
    
    train_dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,  # 加速 CPU-GPU 传输
        prefetch_factor=prefetch_factor,  # 预取更多批次
        persistent_workers=True if num_workers > 0 else False,  # 保持 worker 进程，减少创建开销
        drop_last=True,  # 丢弃最后不完整的批次，确保批次大小一致
    )
    
    # 创建模型
    logger.info("创建 DiT 模型...")
    model = DiTModel(
        hidden_size=config.model.get("hidden_size", 384),
        num_layers=config.model.get("num_layers", 8),
        num_heads=config.model.get("num_heads", 6),
        patch_size=config.model.get("patch_size", 2),
        in_channels=config.model.get("in_channels", 4),
        out_channels=config.model.get("out_channels", 4),
        attention_head_dim=config.model.get("attention_head_dim", 64),
        mlp_ratio=config.model.get("mlp_ratio", 4.0),
        dropout=config.model.get("dropout", 0.1),
        input_size=config.data.get("image_size", 256) // 8,  # 潜在空间尺寸
    )
    model = model.to(device)
    
    # 编译模型以优化性能（PyTorch 2.0+）
    # 注意：编译模式使用 "default" 而不是 "reduce-overhead" 以避免 CUDA graph 问题
    if config.training.get("compile_model", False) and hasattr(torch, "compile"):
        try:
            logger.info("编译模型以优化性能...")
            # 使用 "default" 模式，避免 CUDA graphs 导致的问题
            model = torch.compile(model, mode="default")
            logger.info("✓ 模型编译成功（使用 default 模式）")
        except Exception as e:
            logger.warning(f"模型编译失败，继续使用未编译版本: {e}")
    
    # 创建训练器
    logger.info("创建训练器...")
    trainer = Trainer(
        model=model,
        vae_encoder=vae_encoder,
        text_encoder=text_encoder,
        train_dataloader=train_dataloader,
        config=config,
    )
    
    # 恢复训练（如果指定）
    if args.resume:
        logger.info(f"从检查点恢复训练: {args.resume}")
        trainer.load_checkpoint(Path(args.resume))
    
    # 开始训练
    trainer.train()
    
    logger.info("训练完成！")


if __name__ == "__main__":
    main()

