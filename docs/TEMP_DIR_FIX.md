# 临时目录问题解决方案

## 问题描述

当系统磁盘空间已满或临时目录不可用时，运行脚本会出现以下错误：

```
FileNotFoundError: [Errno 2] No usable temporary directory found
```

## 解决方案

### 方法1: 使用启动脚本（推荐）

项目提供了启动脚本，会自动设置临时目录：

```bash
# 推理
./run_inference.sh --checkpoint ./outputs/checkpoint-epoch-118 --prompt "a cat sitting on a chair"

# 训练
./run_train.sh --config configs/train_config.yaml
```

### 方法2: 手动设置环境变量

在运行脚本前设置环境变量：

```bash
export TMPDIR=$(pwd)/outputs
export TMP=$(pwd)/outputs
export TEMP=$(pwd)/outputs

python src/scripts/inference.py --checkpoint ./outputs/checkpoint-epoch-118 --prompt "a cat sitting on a chair"
```

### 方法3: 清理磁盘空间

如果可能，清理一些空间：

```bash
# 查看磁盘使用情况
df -h

# 清理旧的检查点（保留最新的几个）
# 注意：请谨慎操作，确保不会删除重要文件
```

## 技术说明

Python 的 `tempfile` 模块在导入时会检查临时目录是否可用。当系统临时目录（`/tmp`, `/var/tmp` 等）不可用时，需要手动指定一个可用的目录。

项目脚本已经包含了自动设置临时目录的代码，但如果磁盘空间已满，可能仍然无法创建新目录。此时需要使用已存在的目录（如 `outputs` 目录）作为临时目录。

## 注意事项

1. **磁盘空间**: 确保有足够的磁盘空间用于训练和推理
2. **权限**: 确保对临时目录有读写权限
3. **清理**: 定期清理临时文件和旧的检查点

