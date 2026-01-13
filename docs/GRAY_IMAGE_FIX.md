# 灰色图像问题修复

## 🔍 问题分析

### 症状

生成的图像都是灰色，没有任何内容：
- 值范围: [108, 155] (非常窄)
- 均值: ~120 (接近128，灰色)
- 标准差: ~5 (非常小，几乎没有变化)
- 通道差异: <10 (RGB三个通道值几乎相同)

### 根本原因

**潜在表示初始化时没有按 `scaling_factor` 缩放！**

#### 训练时的处理

```python
# 训练时
latents = vae_encoder.encode(images)  # 自动应用 scaling_factor
# 内部: latent = latent * scaling_factor
```

#### 推理时的处理（修复前）

```python
# 推理时（错误）
latents = torch.randn(...)  # 标准正态分布，没有缩放！
# 解码时
latents = latents / scaling_factor  # 只在这里缩放
```

**问题**: 初始化的潜在表示范围与训练时不一致，导致模型无法正确去噪。

## ✅ 修复方案

### 修复代码

```python
# 初始化潜在表示（随机噪声）
latents = torch.randn(
    (1, 4, latent_height, latent_width),
    device=self.device,
)
# 应用 scaling_factor（与训练时编码的缩放一致）
scaling_factor = self.vae_decoder.vae.config.scaling_factor
latents = latents * scaling_factor
```

### 修复后的流程

1. **初始化**: `latents = torch.randn(...)` - 标准正态分布
2. **缩放**: `latents = latents * scaling_factor` - 与训练时一致
3. **扩散采样**: 模型去噪
4. **解码**: `latents = latents / scaling_factor` - 解码前还原
5. **VAE解码**: 生成图像

## 📊 预期效果

修复后：
- ✅ 潜在表示范围与训练时一致
- ✅ 模型可以正确去噪
- ✅ 生成的图像应该有颜色和内容
- ✅ 不再出现灰色图像

## 🧪 验证

重新运行推理：

```bash
./run_inference.sh \
    --checkpoint ./outputs/checkpoint-epoch-200 \
    --prompt "a cat" \
    --num_inference_steps 100
```

检查生成的图像：
- 应该有颜色（RGB通道差异 > 20）
- 应该有内容（不是纯灰色）
- 值范围应该更广（不是集中在 [108, 155]）

## 🔧 技术细节

### VAE Scaling Factor

Stable Diffusion VAE 使用 `scaling_factor` 来缩放潜在表示：
- **默认值**: 通常为 `0.18215`
- **作用**: 将潜在表示缩放到合适的范围
- **训练时**: 编码时放大，解码时缩小
- **推理时**: 初始化时也需要放大，解码时缩小

### 为什么需要缩放？

1. **数值稳定性**: 避免潜在表示过大或过小
2. **训练一致性**: 确保训练和推理使用相同的范围
3. **模型性能**: 模型在特定范围内训练，需要在该范围内推理

## ⚠️ 注意事项

1. **确保 scaling_factor 正确**: 从 VAE 配置中读取，不要硬编码
2. **训练和推理一致**: 必须使用相同的缩放方式
3. **检查 VAE 模型**: 确保使用的是正确的 VAE 模型

---

**修复状态**: ✅ 已修复

**文件**: `src/inference/generator.py`

