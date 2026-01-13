# 快速改进步骤

## 🎯 最简单的改进方法（推荐先做）

### 步骤1: 继续训练到 400 epochs

```bash
# 1. 修改配置文件
# 编辑 configs/train_config.yaml
# 将 num_epochs: 200 改为 num_epochs: 400

# 2. 继续训练（断点续传）
python src/scripts/train.py \
    --config configs/train_config.yaml \
    --resume ./outputs/checkpoint-epoch-200
```

**时间**: 约 10-15 小时  
**预期**: 图像质量提升 30-50%

---

## 🚀 最佳改进方法（推荐）

### 步骤1: 使用 COCO 数据集

```bash
# 1. 修改配置文件
# 编辑 configs/train_config.yaml
data:
  dataset_name: "coco"
  dataset_path: "./data"
  num_samples: 50000  # 使用 5 万张图像（或 null 使用全部）
  image_size: 256

training:
  num_epochs: 300  # 真实数据通常需要更少 epochs
```

### 步骤2: 重新训练

```bash
# 从头开始训练（使用真实数据）
python src/scripts/train.py \
    --config configs/train_config.yaml
```

**时间**: 约 15-20 小时（数据下载 + 训练）  
**预期**: 图像质量提升 50-100%

---

## 📊 Epoch 数量参考

| 数据量 | 推荐 Epochs |
|--------|------------|
| 5,000 样本 | 500-800 |
| 10,000 样本 | 300-500 |
| 50,000 样本 | 200-300 |
| 100,000+ 样本 | 150-250 |

**当前**: 5,000 样本，200 epochs → **建议**: 继续训练到 400-500 epochs

---

## ⚡ 立即开始

**最快的方法**（现在就做）:
```bash
# 1. 修改配置
sed -i 's/num_epochs: 200/num_epochs: 400/' configs/train_config.yaml

# 2. 继续训练
python src/scripts/train.py \
    --config configs/train_config.yaml \
    --resume ./outputs/checkpoint-epoch-200
```

**最佳的方法**（需要准备数据）:
1. 确保可以下载 COCO 数据集（或手动下载）
2. 修改配置使用真实数据
3. 重新训练

---

详细说明请查看: `docs/IMPROVEMENT_GUIDE.md`

