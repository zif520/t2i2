"""测试完整训练流程，确保能正常运行"""

import torch
import sys
from pathlib import Path
from torch.utils.data import DataLoader

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from src.models.dit_model import DiTModel
from src.models.vae_model import VAEEncoder
from src.data.dataset import TextImageDataset
from src.utils.config import load_config
from src.utils.logger import setup_logger
from transformers import CLIPTextModel, CLIPTokenizer

def test_training_step():
    """测试单个训练步骤"""
    print("="*80)
    print("测试完整训练流程")
    print("="*80)
    
    # 加载配置
    config = load_config("configs/train_config.yaml")
    logger = setup_logger()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    
    # 创建小数据集用于测试
    print("\n1. 创建测试数据集...")
    dataset = TextImageDataset(
        dataset_name="custom",
        dataset_path="./data/test_data",
        image_size=config.data.get("image_size", 256),
        num_samples=100,
        is_train=True,
    )
    print(f"   ✓ 数据集创建成功，共 {len(dataset)} 个样本")
    
    # 创建数据加载器
    print("\n2. 创建数据加载器...")
    dataloader = DataLoader(
        dataset,
        batch_size=min(config.training.get("batch_size", 56), 8),  # 测试用小批次
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    print(f"   ✓ 数据加载器创建成功，批次大小: {dataloader.batch_size}")
    
    # 加载文本编码器
    print("\n3. 加载文本编码器...")
    text_encoder_name = config.text_encoder.get("pretrained_model_name", "openai/clip-vit-base-patch32")
    text_encoder = CLIPTextModel.from_pretrained(text_encoder_name)
    text_encoder = text_encoder.to(device)
    text_encoder.eval()
    print("   ✓ 文本编码器加载成功")
    
    # 加载 VAE
    print("\n4. 加载 VAE 编码器...")
    vae_encoder = VAEEncoder(
        pretrained_model_name=config.vae.get("pretrained_model_name", "runwayml/stable-diffusion-v1-5"),
        use_slicing=config.vae.get("use_slicing", False),
    )
    vae_encoder = vae_encoder.to(device)
    print("   ✓ VAE 编码器加载成功")
    
    # 创建模型
    print("\n5. 创建 DiT 模型...")
    model = DiTModel(
        hidden_size=config.model.get("hidden_size", 768),
        num_layers=config.model.get("num_layers", 16),
        num_heads=config.model.get("num_heads", 12),
        patch_size=config.model.get("patch_size", 2),
        in_channels=config.model.get("in_channels", 4),
        out_channels=config.model.get("out_channels", 4),
        attention_head_dim=config.model.get("attention_head_dim", 64),
        mlp_ratio=config.model.get("mlp_ratio", 4.0),
        dropout=config.model.get("dropout", 0.1),
        input_size=config.data.get("image_size", 256) // 8,
    )
    model = model.to(device)
    print(f"   ✓ 模型创建成功，参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    
    # 测试编译（如果启用）
    if config.training.get("compile_model", False) and hasattr(torch, "compile"):
        print("\n6. 编译模型...")
        try:
            model = torch.compile(model, mode="default")
            print("   ✓ 模型编译成功")
        except Exception as e:
            print(f"   ⚠ 模型编译失败: {e}")
    
    # 测试训练步骤
    print("\n7. 测试训练步骤...")
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    from diffusers import DDPMScheduler
    scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        prediction_type="epsilon",
    )
    
    tokenizer = CLIPTokenizer.from_pretrained(text_encoder_name)
    
    import time
    times = []
    
    for i, batch in enumerate(dataloader):
        if i >= 5:  # 只测试5个批次
            break
        
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        
        start = time.time()
        
        # VAE 编码
        with torch.no_grad():
            latents = vae_encoder.encode(pixel_values)
        
        # 文本编码
        with torch.no_grad():
            text_outputs = text_encoder(input_ids)
            if hasattr(text_outputs, 'pooler_output') and text_outputs.pooler_output is not None:
                text_embeddings = text_outputs.pooler_output
            else:
                text_embeddings = text_outputs.last_hidden_state.mean(dim=1)
            text_embeddings = text_embeddings.contiguous()
        
        # 采样时间步
        timesteps = torch.randint(0, 1000, (latents.shape[0],), device=device, dtype=torch.long)
        
        # 添加噪声
        noise = torch.randn_like(latents)
        noisy_latents = scheduler.add_noise(latents, noise, timesteps)
        
        # 前向传播
        with torch.amp.autocast(device_type="cuda"):
            pred_noise = model(noisy_latents, timesteps, text_embeddings)
            loss = torch.nn.functional.mse_loss(pred_noise, noise)
        
        # 反向传播
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        torch.cuda.synchronize()
        elapsed = time.time() - start
        times.append(elapsed)
        
        print(f"   批次 {i+1}: 损失={loss.item():.4f}, 时间={elapsed:.3f}s")
    
    avg_time = sum(times) / len(times)
    print(f"\n   ✓ 平均训练时间: {avg_time:.3f} 秒/批次")
    
    # 检查显存
    allocated = torch.cuda.memory_allocated(0) / 1024**3
    reserved = torch.cuda.memory_reserved(0) / 1024**3
    print(f"   显存使用: {reserved:.2f} GB / 24 GB ({reserved/24*100:.1f}%)")
    
    print("\n" + "="*80)
    print("✅ 完整训练流程测试通过！")
    print("="*80)
    
    return True

if __name__ == "__main__":
    try:
        test_training_step()
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

