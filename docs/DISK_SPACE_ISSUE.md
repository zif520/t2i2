# 磁盘空间问题解决方案

## 问题描述

当前系统磁盘空间已满（100%），导致：

1. 无法创建临时目录
2. Python 的 `tempfile` 模块无法工作
3. 无法运行训练和推理脚本

## 错误信息

```
FileNotFoundError: [Errno 2] No usable temporary directory found
```

## 解决方案

### 方案1: 清理旧的检查点（推荐）

检查点文件占用大量空间，可以删除旧的检查点，只保留最新的几个：

```bash
# 查看检查点大小
du -sh outputs/checkpoint-*

# 删除旧的检查点（例如，只保留最后 5 个）
# 注意：请根据实际情况调整
cd outputs
ls -t checkpoint-* | tail -n +6 | xargs rm -rf
```

### 方案2: 清理其他临时文件

```bash
# 清理 Python 缓存
find . -type d -name __pycache__ -exec rm -r {} +
find . -type f -name "*.pyc" -delete

# 清理日志文件（如果很大）
# 注意：请确认日志文件可以删除
```

### 方案3: 使用外部存储

如果可能，将检查点移动到外部存储：

```bash
# 创建外部存储目录（如果有）
mkdir -p /path/to/external/storage

# 移动旧的检查点
mv outputs/checkpoint-epoch-* /path/to/external/storage/
```

### 方案4: 压缩检查点

压缩旧的检查点以节省空间：

```bash
# 压缩检查点
cd outputs
for dir in checkpoint-epoch-*; do
    if [ -d "$dir" ]; then
        tar -czf "${dir}.tar.gz" "$dir"
        rm -rf "$dir"
    fi
done
```

## 预防措施

### 1. 配置检查点保存策略

修改训练配置，只保存最新的几个检查点：

```yaml
training:
  save_steps: 1000  # 每 1000 步保存一次
  save_total_limit: 5  # 只保留最新的 5 个检查点
```

### 2. 定期清理

设置定期清理任务：

```bash
# 创建清理脚本
cat > cleanup_checkpoints.sh << 'EOF'
#!/bin/bash
# 只保留最新的 5 个检查点
cd outputs
ls -t checkpoint-* 2>/dev/null | tail -n +6 | xargs rm -rf
EOF

chmod +x cleanup_checkpoints.sh
```

### 3. 监控磁盘空间

定期检查磁盘使用情况：

```bash
# 查看磁盘使用
df -h

# 查看大文件
du -h outputs/ | sort -h | tail -20
```

## 检查点管理建议

### 保留策略

- **训练中**: 保留最新的 3-5 个检查点
- **训练完成**: 只保留最终检查点和最佳检查点
- **长期保存**: 压缩后移动到外部存储

### 检查点大小

每个检查点大约 1.4GB，100 个检查点约 140GB。

建议：
- 训练过程中：只保留最新 5 个（约 7GB）
- 训练完成后：只保留最终检查点（约 1.4GB）

## 快速清理命令

```bash
# 只保留最新的 5 个检查点
cd /root/zlf/t2i2/outputs
ls -t checkpoint-* 2>/dev/null | tail -n +6 | xargs rm -rf

# 查看释放的空间
df -h
```

## 注意事项

⚠️ **重要**: 删除检查点前，请确认：
1. 已备份重要的检查点
2. 训练已完成或已暂停
3. 不会影响正在进行的训练

## 联系支持

如果问题仍然存在，请：
1. 检查是否有其他大文件占用空间
2. 联系系统管理员增加磁盘空间
3. 考虑使用云存储或外部存储

