"""检查 GPU 显存使用情况"""
import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"总显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # 测试不同批次大小
    from src.models.dit_model import DiTModel
    from src.utils.config import load_config
    
    config = load_config("configs/train_config.yaml")
    
    # 创建模型
    model = DiTModel(
        hidden_size=config.model.get("hidden_size", 512),
        num_layers=config.model.get("num_layers", 12),
        num_heads=config.model.get("num_heads", 8),
        input_size=config.data.get("image_size", 256) // 8,
    ).to(device)
    
    # 测试不同批次大小
    image_size = config.data.get("image_size", 256)
    latent_size = image_size // 8
    
    for batch_size in [8, 16, 24, 32]:
        try:
            # 模拟输入
            latents = torch.randn(batch_size, 4, latent_size, latent_size).to(device)
            timesteps = torch.randint(0, 1000, (batch_size,)).to(device)
            text_emb = torch.randn(batch_size, 512).to(device)
            
            # 前向传播
            with torch.cuda.amp.autocast():
                _ = model(latents, timesteps, text_emb)
            
            # 检查显存
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            reserved = torch.cuda.memory_reserved(0) / 1024**3
            
            print(f"批次大小 {batch_size:2d}: 已分配 {allocated:.2f} GB, 已保留 {reserved:.2f} GB")
            
            # 清理
            torch.cuda.empty_cache()
        except RuntimeError as e:
            print(f"批次大小 {batch_size:2d}: OOM - {str(e)[:50]}")
            torch.cuda.empty_cache()
            break
else:
    print("CUDA 不可用")
