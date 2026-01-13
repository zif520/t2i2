# 快速开始指南

## 🚀 5 分钟快速上手

### 步骤 1: 安装环境

```bash
# 创建虚拟环境
conda create -n dit python=3.10
conda activate dit

# 安装依赖
pip install -r requirements.txt
```

### 步骤 2: 准备测试数据

```bash
# 准备 100 个测试样本
python src/scripts/prepare_data.py \
    --type coco \
    --output ./data/test_data \
    --num_samples 100
```

### 步骤 3: 开始训练

```bash
# 使用优化配置训练
python src/scripts/train.py --config configs/train_config.yaml
```

### 步骤 4: 生成图像

```bash
# 使用训练好的模型生成图像
python src/scripts/inference.py \
    --checkpoint ./outputs/checkpoint-5000 \
    --prompt "a cat sitting on a chair"
```

---

## 📋 完整训练流程

### 1. 环境检查

```bash
# 检查 GPU
nvidia-smi

# 检查 PyTorch
python -c "import torch; print(torch.cuda.is_available())"
```

### 2. 准备数据

```bash
# 准备训练数据（5000 样本）
python src/scripts/prepare_data.py \
    --type coco \
    --output ./data/coco_subset \
    --num_samples 5000
```

### 3. 配置检查

检查 `configs/train_config.yaml`：

- ✅ 批次大小: 96（显存利用率 83.6%）
- ✅ 模型: 768/16层/12头
- ✅ 混合精度: bf16
- ✅ Epochs: 200（推荐）

### 4. 开始训练

```bash
python src/scripts/train.py --config configs/train_config.yaml
```

**训练时间估算：**
- 5000 样本，批次 96
- 200 epochs ≈ 12 小时

### 5. 监控训练

```bash
# 查看训练日志
tail -f outputs/train.log

# 监控 GPU
watch -n 1 nvidia-smi
```

### 6. 生成图像

```bash
# 基本推理
python src/scripts/inference.py \
    --checkpoint ./outputs/checkpoint-5000 \
    --prompt "a beautiful landscape"

# 高质量生成
python src/scripts/inference.py \
    --checkpoint ./outputs/checkpoint-5000 \
    --prompt "a cat sitting on a chair" \
    --num_inference_steps 100
```

---

## 🎯 关键配置说明

### 训练配置（configs/train_config.yaml）

```yaml
# 模型配置（最优）
model:
  hidden_size: 768
  num_layers: 16
  num_heads: 12

# 训练配置（充分利用显存）
training:
  batch_size: 96        # 显存利用率 83.6%
  num_epochs: 200       # 推荐值
  mixed_precision: "bf16"  # BF16 更稳定
  learning_rate: 0.0001
```

### 性能指标

- **显存利用率**: 83.6% (20.1 GB / 24 GB)
- **训练速度**: ~42 ms/批次
- **吞吐量**: 1327 样本/秒

---

## 💡 常用命令

### 训练相关

```bash
# 开始训练
python src/scripts/train.py --config configs/train_config.yaml

# 恢复训练
python src/scripts/train.py \
    --config configs/train_config.yaml \
    --resume ./outputs/checkpoint-5000
```

### 推理相关

```bash
# 基本推理
python src/scripts/inference.py \
    --checkpoint ./outputs/checkpoint-5000 \
    --prompt "your prompt here"

# 快速测试（20步）
python src/scripts/inference.py \
    --checkpoint ./outputs/checkpoint-5000 \
    --prompt "a cat" \
    --num_inference_steps 20

# 高质量生成（100步）
python src/scripts/inference.py \
    --checkpoint ./outputs/checkpoint-5000 \
    --prompt "a cat sitting on a chair" \
    --num_inference_steps 100
```

### 数据准备

```bash
# COCO 子集
python src/scripts/prepare_data.py \
    --type coco \
    --output ./data/coco_subset \
    --num_samples 5000

# 自定义数据
python src/scripts/prepare_data.py \
    --type custom \
    --input ./custom_data \
    --output ./data/custom
```

---

## ⚠️ 常见问题快速解决

### 显存不足

```yaml
# 减小批次大小
training:
  batch_size: 64  # 从 96 减小到 64

# 或启用 VAE 切片
vae:
  use_slicing: true
```

### 训练速度慢

```yaml
# 检查配置
training:
  num_workers: 8        # 确保足够
  compile_model: true   # 启用编译
  mixed_precision: "bf16"  # 使用 BF16
```

### 生成质量差

- 增加推理步数: `--num_inference_steps 100`
- 改进提示词: 更具体、更详细
- 检查模型训练是否充分

---

## 📚 详细文档

- [完整教程](./00-完整教程.md) - 完整学习路径
- [入门指南](./01-入门指南.md) - 基础概念
- [环境配置](./02-环境配置.md) - 环境搭建
- [数据准备](./03-数据准备.md) - 数据准备
- [训练流程](./05-训练流程.md) - 训练详解
- [推理使用](./06-推理使用.md) - 推理详解
- [常见问题](./07-常见问题.md) - FAQ
- [进阶学习](./08-进阶学习.md) - 进阶内容

---

**开始你的 DiT 文生图之旅吧！** 🎨

