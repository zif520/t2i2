"""推理脚本"""

import argparse
import os
import sys
from pathlib import Path

# 在导入 torch 之前设置临时目录（如果系统临时目录不可用）
project_root = Path(__file__).parent.parent.parent
if not os.environ.get("TMPDIR"):
    # 使用 outputs 目录下的 tmp 子目录
    tmp_dir = project_root / "outputs" / "tmp"
    try:
        tmp_dir.mkdir(parents=True, exist_ok=True)
        os.environ["TMPDIR"] = str(tmp_dir.absolute())
        os.environ["TMP"] = str(tmp_dir.absolute())
        os.environ["TEMP"] = str(tmp_dir.absolute())
    except (OSError, PermissionError):
        # 如果无法创建，尝试使用 outputs 目录本身
        outputs_dir = project_root / "outputs"
        os.environ["TMPDIR"] = str(outputs_dir.absolute())
        os.environ["TMP"] = str(outputs_dir.absolute())
        os.environ["TEMP"] = str(outputs_dir.absolute())

# 现在可以安全导入 torch
import torch
from transformers import CLIPTextModel, CLIPTokenizer

# 添加项目根目录到路径，支持直接运行脚本
sys.path.insert(0, str(project_root))

from src.models.dit_model import DiTModel
from src.models.vae_model import VAEDecoder
from src.inference.generator import ImageGenerator
from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.utils.visualization import save_image_grid


def main():
    parser = argparse.ArgumentParser(description="使用 DiT 模型生成图像")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_config.yaml",
        help="配置文件路径",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="模型检查点路径",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="文本提示",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./outputs/generated",
        help="输出目录",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="推理步数",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=256,
        help="图像高度",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=256,
        help="图像宽度",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="随机种子",
    )
    args = parser.parse_args()
    
    # 设置日志
    logger = setup_logger()
    logger.info("开始生成图像")
    
    # 加载配置
    config = load_config(args.config)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    
    # 加载文本编码器
    logger.info("加载文本编码器...")
    # 设置下载超时（通过环境变量）
    import os
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
    
    text_encoder_name = config.text_encoder.get("pretrained_model_name", "openai/clip-vit-base-patch32")
    text_encoder = CLIPTextModel.from_pretrained(text_encoder_name)
    text_encoder = text_encoder.to(device)
    text_encoder.eval()
    
    tokenizer = CLIPTokenizer.from_pretrained(text_encoder_name)
    
    # 加载 VAE 解码器
    logger.info("加载 VAE 解码器...")
    vae_config = config.vae or {}
    vae_decoder = VAEDecoder(
        pretrained_model_name=vae_config.get("pretrained_model_name", "stabilityai/sd-vae-ft-mse"),
        use_slicing=vae_config.get("use_slicing", True),
    )
    vae_decoder = vae_decoder.to(device)
    
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
        input_size=args.height // 8,
    )
    model = model.to(device)
    
    # 加载检查点
    logger.info(f"加载模型检查点: {args.checkpoint}")
    checkpoint_path = Path(args.checkpoint) / "model.pt"
    if not checkpoint_path.exists():
        checkpoint_path = Path(args.checkpoint)
    
    state_dict = torch.load(checkpoint_path, map_location=device)
    
    # 处理 torch.compile 编译后的键名（移除 _orig_mod. 前缀）
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        logger.info("检测到编译后的模型，移除 _orig_mod. 前缀")
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("_orig_mod."):
                new_state_dict[k[10:]] = v  # 移除 "_orig_mod." 前缀
            else:
                new_state_dict[k] = v
        state_dict = new_state_dict
    
    model.load_state_dict(state_dict)
    logger.info("模型加载完成")
    
    # 创建生成器
    generator = ImageGenerator(
        model=model,
        vae_decoder=vae_decoder,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        scheduler_type="ddpm",
        device=device,
    )
    
    # 生成图像
    logger.info(f"生成图像，提示: {args.prompt}")
    image = generator.generate(
        prompt=args.prompt,
        num_inference_steps=args.num_inference_steps,
        height=args.height,
        width=args.width,
        seed=args.seed,
    )
    
    # 保存图像
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成文件名
    prompt_safe = "".join(c for c in args.prompt if c.isalnum() or c in (" ", "-", "_")).rstrip()[:50]
    output_path = output_dir / f"{prompt_safe}.png"
    
    image.save(output_path)
    logger.info(f"图像已保存到: {output_path}")
    
    # 如果生成多张，保存网格
    logger.info("生成完成！")


if __name__ == "__main__":
    main()
