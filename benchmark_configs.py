"""测试不同配置组合，找到最优性能配置"""

import torch
import time
from src.models.dit_model import DiTModel
from src.utils.config import load_config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"总显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB\n")

# 测试配置
configs = [
    {"image_size": 256, "hidden_size": 512, "num_layers": 12, "num_heads": 8, "batch_size": 32},
    {"image_size": 256, "hidden_size": 768, "num_layers": 16, "num_heads": 12, "batch_size": 48},
    {"image_size": 256, "hidden_size": 768, "num_layers": 16, "num_heads": 12, "batch_size": 56},
    {"image_size": 256, "hidden_size": 768, "num_layers": 20, "num_heads": 16, "batch_size": 40},
    {"image_size": 256, "hidden_size": 1024, "num_layers": 16, "num_heads": 16, "batch_size": 32},
]

results = []

for i, cfg in enumerate(configs):
    print(f"测试配置 {i+1}/{len(configs)}:")
    print(f"  图像: {cfg['image_size']}, 隐藏层: {cfg['hidden_size']}, 层数: {cfg['num_layers']}, 头数: {cfg['num_heads']}, 批次: {cfg['batch_size']}")
    
    try:
        torch.cuda.empty_cache()
        
        # 创建模型
        latent_size = cfg['image_size'] // 8
        model = DiTModel(
            hidden_size=cfg['hidden_size'],
            num_layers=cfg['num_layers'],
            num_heads=cfg['num_heads'],
            input_size=latent_size,
        ).to(device)
        
        # 测试前向传播
        latents = torch.randn(cfg['batch_size'], 4, latent_size, latent_size).to(device)
        timesteps = torch.randint(0, 1000, (cfg['batch_size'],)).to(device)
        text_emb = torch.randn(cfg['batch_size'], 512).to(device)
        
        # 预热
        with torch.amp.autocast(device_type="cuda"):
            _ = model(latents, timesteps, text_emb)
        
        torch.cuda.synchronize()
        
        # 测试速度
        start_time = time.time()
        for _ in range(10):
            with torch.amp.autocast(device_type="cuda"):
                _ = model(latents, timesteps, text_emb)
        torch.cuda.synchronize()
        elapsed = time.time() - start_time
        
        # 检查显存
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        utilization = reserved / 24 * 100
        
        avg_time = elapsed / 10
        throughput = cfg['batch_size'] / avg_time
        
        results.append({
            **cfg,
            "allocated_gb": allocated,
            "reserved_gb": reserved,
            "utilization": utilization,
            "time_per_batch": avg_time,
            "throughput": throughput,
            "status": "OK"
        })
        
        print(f"  ✓ 显存: {reserved:.2f} GB ({utilization:.1f}%), 速度: {avg_time:.3f}s/批次, 吞吐: {throughput:.1f} 样本/s\n")
        
    except RuntimeError as e:
        results.append({
            **cfg,
            "status": "OOM",
            "error": str(e)[:50]
        })
        print(f"  ✗ OOM\n")
        torch.cuda.empty_cache()

# 找到最优配置
print("\n" + "="*80)
print("最优配置推荐:")
print("="*80)

# 按利用率排序，找到利用率高且速度快的
valid_results = [r for r in results if r.get("status") == "OK"]
if valid_results:
    # 综合评分：利用率 * 吞吐量
    for r in valid_results:
        r["score"] = r["utilization"] * r["throughput"]
    
    best = max(valid_results, key=lambda x: x["score"])
    
    print(f"\n🏆 推荐配置（综合评分最高）:")
    print(f"  图像尺寸: {best['image_size']}")
    print(f"  隐藏层: {best['hidden_size']}")
    print(f"  层数: {best['num_layers']}")
    print(f"  注意力头: {best['num_heads']}")
    print(f"  批次大小: {best['batch_size']}")
    print(f"  显存利用率: {best['utilization']:.1f}% ({best['reserved_gb']:.2f} GB)")
    print(f"  训练速度: {best['time_per_batch']:.3f} 秒/批次")
    print(f"  吞吐量: {best['throughput']:.1f} 样本/秒")
    print(f"  综合评分: {best['score']:.0f}")
    
    print(f"\n📊 所有有效配置:")
    for r in sorted(valid_results, key=lambda x: x["score"], reverse=True):
        print(f"  批次{r['batch_size']:2d} | 隐藏{r['hidden_size']:4d} | 层{r['num_layers']:2d} | 头{r['num_heads']:2d} | "
              f"显存{r['reserved_gb']:5.2f}GB ({r['utilization']:5.1f}%) | "
              f"速度{r['time_per_batch']:.3f}s | 吞吐{r['throughput']:5.1f}样本/s")

