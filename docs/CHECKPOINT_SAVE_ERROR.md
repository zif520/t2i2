# 检查点保存错误解决方案

## 🔍 错误分析

### 错误信息

```
RuntimeError: [enforce fail at inline_container.cc:858] . PytorchStreamWriter failed writing file data/37: file write failed
RuntimeError: [enforce fail at inline_container.cc:664] . unexpected pos 56833600 vs 56833492
```

### 问题原因

**磁盘空间不足** - 这是最常见的原因

- 检查点文件很大（每个约 450MB）
- 保存时磁盘空间不足
- 写入过程中失败

## 🚀 解决方案

### 方案1: 清理旧的检查点（立即）

```bash
# 只保留最新的 5 个检查点
cd outputs
ls -1td checkpoint-* | tail -n +6 | xargs rm -rf

# 查看释放的空间
df -h
```

### 方案2: 减少保存频率

编辑 `configs/train_config.yaml`：

```yaml
training:
  save_every_n_epochs: 5  # 每 5 个 epoch 保存一次，而不是每个 epoch
```

### 方案3: 只保存模型权重（节省空间）

修改代码，不保存优化器（优化器文件约 150MB）：

```python
# 在 save_checkpoint 中，可以跳过优化器保存
# 优化器状态对于恢复训练有用，但不是必需的
```

### 方案4: 使用自动清理脚本

创建定期清理脚本：

```bash
# 只保留最新的 5 个检查点
cd outputs
ls -1td checkpoint-* | tail -n +6 | xargs rm -rf
```

## 🔧 已实现的改进

### 1. 错误处理

代码已添加 try-except，保存失败不会中断训练：

```python
try:
    self.save_checkpoint(...)
except Exception as e:
    logger.error(f"保存检查点失败，但继续训练: {e}")
```

### 2. 减少保存频率

添加 `save_every_n_epochs` 配置，可以每 N 个 epoch 保存一次。

### 3. 优雅降级

如果磁盘空间不足，会尝试只保存模型权重，跳过优化器等大文件。

## 📝 最佳实践

### 1. 定期清理

```bash
# 每训练一段时间，清理旧检查点
cd outputs
ls -1td checkpoint-* | tail -n +6 | xargs rm -rf
```

### 2. 配置保存策略

```yaml
training:
  save_every_n_epochs: 5  # 每 5 个 epoch 保存一次
  save_steps: 500  # 每 500 步保存一次（基于步数）
```

### 3. 只保留重要检查点

- 训练中：保留最新的 3-5 个
- 训练完成：只保留最终检查点

## ⚠️ 注意事项

1. **优化器状态**: 如果不保存优化器，恢复训练时优化器会重新初始化（但模型权重会保留）
2. **训练状态**: 训练状态文件很小，应该总是能保存
3. **磁盘监控**: 定期检查磁盘空间

## 🎯 推荐配置

```yaml
training:
  save_every_n_epochs: 5  # 每 5 个 epoch 保存一次
  save_steps: 500  # 每 500 步保存一次
```

这样：
- 每 5 个 epoch 保存一次（约 5 个检查点）
- 每 500 步也保存一次（基于步数）
- 平衡了保存频率和磁盘空间

---

**立即操作**: 清理旧检查点，然后继续训练。

