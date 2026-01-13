# 断点续传快速开始

## 🚀 快速使用

### 从最新检查点恢复训练

```bash
# 方法1: 使用启动脚本
./run_train.sh \
    --config configs/train_config.yaml \
    --resume ./outputs/checkpoint-epoch-118

# 方法2: 直接运行
python src/scripts/train.py \
    --config configs/train_config.yaml \
    --resume ./outputs/checkpoint-epoch-118
```

### 查找最新检查点

```bash
# 查看所有检查点
ls -1td outputs/checkpoint-* | head -5

# 自动使用最新检查点
LATEST=$(ls -1td outputs/checkpoint-* | head -1)
./run_train.sh --config configs/train_config.yaml --resume "$LATEST"
```

## 📋 功能说明

断点续传会自动恢复：
- ✅ 模型权重
- ✅ 优化器状态
- ✅ 学习率调度器状态
- ✅ 训练进度（epoch、step）

## 💡 使用场景

### 场景1: 训练中断后恢复

```bash
# 训练中断后，从最后一个检查点恢复
./run_train.sh \
    --config configs/train_config.yaml \
    --resume ./outputs/checkpoint-epoch-118
```

### 场景2: 继续未完成的训练

```bash
# 继续训练到 200 epochs（当前已完成 118）
./run_train.sh \
    --config configs/train_config.yaml \
    --resume ./outputs/checkpoint-epoch-118
```

## 📝 详细文档

完整说明请查看：[断点续传指南](./docs/RESUME_TRAINING.md)

