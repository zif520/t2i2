# 检查点文件损坏修复指南

## 🔍 错误分析

### 错误信息

```
RuntimeError: PytorchStreamWriter failed reading zip archive: failed finding central directory
```

### 问题原因

**检查点文件损坏** - 通常发生在：
1. 保存过程中磁盘空间不足
2. 保存过程中程序被中断
3. 磁盘 I/O 错误

## 🚀 解决方案

### 方案1: 删除损坏的文件（推荐）

优化器和调度器状态不是必需的，可以删除损坏的文件：

```bash
# 删除损坏的优化器文件
rm -f outputs/checkpoint-epoch-118/optimizer.pt

# 删除损坏的调度器文件（如果有）
rm -f outputs/checkpoint-epoch-118/scheduler.pt
```

**影响**:
- ✅ 模型权重已加载（最重要）
- ⚠️ 优化器会重新初始化（但训练可以继续）
- ⚠️ 学习率调度器会重新初始化

### 方案2: 使用代码自动处理（已实现）

代码已更新，会自动处理损坏的文件：
- 如果优化器文件损坏，会跳过并重新初始化
- 如果调度器文件损坏，会跳过并重新初始化
- 训练会继续，不会中断

## 📝 已实现的改进

### 1. 错误处理

所有检查点加载都添加了 try-except：

```python
try:
    optimizer_state = torch.load(optimizer_path, ...)
    self.optimizer.load_state_dict(optimizer_state)
except (RuntimeError, OSError, EOFError) as e:
    logger.warning("优化器文件损坏，将使用新初始化的优化器")
    # 继续训练
```

### 2. 优雅降级

- 模型权重：必需，如果损坏会报错
- 优化器状态：可选，损坏时重新初始化
- 调度器状态：可选，损坏时重新初始化
- EMA 模型：可选，损坏时重新初始化

## ⚠️ 注意事项

### 优化器重新初始化的影响

1. **学习率**: 会从当前配置的学习率开始
2. **优化器状态**: 动量等内部状态会重置
3. **训练连续性**: 模型权重已加载，训练可以继续

### 建议

如果优化器文件损坏：
1. 删除损坏的文件
2. 继续训练（优化器会重新初始化）
3. 训练会继续，但可能需要几个 epoch 来恢复优化器状态

## 🔧 预防措施

### 1. 确保足够的磁盘空间

```bash
# 定期检查
df -h

# 定期清理
./auto_cleanup_checkpoints.sh
```

### 2. 使用保存频率配置

```yaml
training:
  save_every_n_epochs: 10  # 每 10 个 epoch 保存一次
```

### 3. 错误处理

代码已添加错误处理，保存失败不会中断训练。

## 📊 文件重要性

| 文件 | 重要性 | 损坏影响 |
|------|--------|---------|
| `model.pt` | ⭐⭐⭐⭐⭐ | 无法继续训练 |
| `optimizer.pt` | ⭐⭐⭐ | 优化器重新初始化 |
| `scheduler.pt` | ⭐⭐ | 调度器重新初始化 |
| `training_state.json` | ⭐⭐⭐⭐ | 无法恢复训练进度 |
| `ema_model.pt` | ⭐⭐ | EMA 模型重新初始化 |

## 🎯 当前状态

- ✅ 代码已更新，自动处理损坏文件
- ✅ 优化器文件已删除（如果损坏）
- ✅ 可以继续训练

---

**继续训练**: 即使优化器文件损坏，训练也可以继续！

