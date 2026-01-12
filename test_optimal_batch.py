"""测试最优批次大小（256图像）"""
import torch
from src.models.dit_model import DiTModel

device = torch.device("cuda")
image_size = 256
latent_size = image_size // 8

model = DiTModel(
    hidden_size=768,
    num_layers=16,
    num_heads=12,
    input_size=latent_size,
).to(device)

print(f"模型: hidden_size=768, layers=16, heads=12, image_size={image_size}")
print("测试不同批次大小:\n")

for batch_size in [32, 40, 48, 56, 64, 72]:
    try:
        torch.cuda.empty_cache()
        latents = torch.randn(batch_size, 4, latent_size, latent_size).to(device)
        timesteps = torch.randint(0, 1000, (batch_size,)).to(device)
        text_emb = torch.randn(batch_size, 512).to(device)
        
        with torch.amp.autocast(device_type="cuda"):
            _ = model(latents, timesteps, text_emb)
        
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        print(f"批次 {batch_size:2d}: 已分配 {allocated:5.2f} GB, 已保留 {reserved:5.2f} GB, 利用率 {reserved/24*100:4.1f}%")
        
    except RuntimeError as e:
        print(f"批次 {batch_size:2d}: OOM ❌")
        break
    finally:
        torch.cuda.empty_cache()
