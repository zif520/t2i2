# ⚠️ 紧急修复：磁盘空间已满

## 当前问题

**磁盘空间 100% 已满**，导致无法运行任何脚本。

## 立即解决方案

### 方法1: 使用清理脚本（推荐）

```bash
cd /root/zlf/t2i2
./cleanup_disk.sh
```

脚本会：
- 显示当前磁盘使用情况
- 列出要删除的旧检查点
- 询问确认后删除（保留最新的 5 个）

### 方法2: 手动清理

```bash
cd /root/zlf/t2i2/outputs

# 查看所有检查点
ls -1t checkpoint-*

# 删除旧的检查点（保留最新的 5 个）
# 注意：请确认要删除的检查点
ls -1t checkpoint-* | tail -n +6 | xargs rm -rf

# 查看释放的空间
df -h
```

### 方法3: 快速清理（自动，无确认）

```bash
cd /root/zlf/t2i2/outputs
ls -1t checkpoint-* 2>/dev/null | tail -n +6 | xargs rm -rf
df -h
```

## 检查点大小

- 每个检查点约 **1.4GB**
- 100 个检查点约 **140GB**
- 保留 5 个检查点约 **7GB**

## 清理后

清理完成后，可以正常使用：

```bash
# 推理
./run_inference.sh --checkpoint ./outputs/checkpoint-epoch-118 --prompt "a cat sitting on a chair"

# 训练
./run_train.sh --config configs/train_config.yaml
```

## 预防措施

### 1. 修改训练配置，限制检查点数量

编辑 `configs/train_config.yaml`，添加：

```yaml
training:
  save_steps: 1000
  save_total_limit: 5  # 只保留最新的 5 个检查点
```

### 2. 定期清理

```bash
# 添加到 crontab 或定期运行
cd /root/zlf/t2i2/outputs
ls -1t checkpoint-* 2>/dev/null | tail -n +6 | xargs rm -rf
```

## 注意事项

⚠️ **重要**:
- 删除前请确认检查点已备份（如果需要）
- 建议保留最新的几个检查点
- 如果训练还在进行，请谨慎删除

## 如果仍然无法运行

如果清理后仍然无法运行，可能是：
1. 清理的空间不够
2. 其他文件占用空间

检查：

```bash
# 查看大文件
du -h outputs/ | sort -h | tail -20

# 查看磁盘使用
df -h
```

