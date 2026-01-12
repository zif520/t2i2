"""最终性能基准测试"""
import torch
import time
from src.models.dit_model import DiTModel

device = torch.device("cuda")
image_size = 256
latent_size = image_size // 8
batch_size = 56

print("="*80)
print("最终性能基准测试（最优配置）")
print("="*80)
print(f"配置: 批次={batch_size}, 隐藏=768, 层=16, 头=12, 图像={image_size}\n")

model = DiTModel(
    hidden_size=768,
    num_layers=16,
    num_heads=12,
    input_size=latent_size,
).to(device)

# 编译模型
if hasattr(torch, "compile"):
    model = torch.compile(model, mode="default")
    print("✓ 模型编译成功\n")

# 预热
latents = torch.randn(batch_size, 4, latent_size, latent_size).to(device)
timesteps = torch.randint(0, 1000, (batch_size,), device=device, dtype=torch.long)
text_emb = torch.randn(batch_size, 512).to(device)

with torch.amp.autocast(device_type="cuda"):
    _ = model(latents, timesteps, text_emb)
torch.cuda.synchronize()

# 性能测试
print("性能测试（20次迭代）...")
times = []
for i in range(20):
    torch.cuda.synchronize()
    start = time.time()
    
    with torch.amp.autocast(device_type="cuda"):
        _ = model(latents, timesteps, text_emb)
    
    torch.cuda.synchronize()
    elapsed = time.time() - start
    times.append(elapsed)
    if i < 5 or i % 5 == 0:
        print(f"  迭代 {i+1:2d}: {elapsed*1000:.2f} ms")

avg_time = sum(times[5:]) / len(times[5:])  # 排除前5次
std_time = (sum((t - avg_time)**2 for t in times[5:]) / len(times[5:]))**0.5

allocated = torch.cuda.memory_allocated(0) / 1024**3
reserved = torch.cuda.memory_reserved(0) / 1024**3
utilization = reserved / 24 * 100
throughput = batch_size / avg_time

print(f"\n结果:")
print(f"  平均时间: {avg_time*1000:.2f} ms/批次 (std: {std_time*1000:.2f} ms)")
print(f"  吞吐量: {throughput:.1f} 样本/秒")
print(f"  显存使用: {reserved:.2f} GB / 24 GB ({utilization:.1f}%)")
print(f"  显存分配: {allocated:.2f} GB")
print("\n" + "="*80)
print("✅ 性能测试完成！")
print("="*80)
