# 快速修复：Hugging Face 下载超时

## 🚀 立即解决方案

### 方法1: 使用镜像源（最快）⭐

```bash
# 设置镜像源
export HF_ENDPOINT=https://hf-mirror.com

# 重新运行
./run_inference.sh --checkpoint ./outputs/checkpoint-epoch-118 --prompt "a cat"
```

### 方法2: 使用设置脚本

```bash
# 运行设置脚本
source setup_hf_mirror.sh

# 然后运行推理
./run_inference.sh --checkpoint ./outputs/checkpoint-epoch-118 --prompt "a cat"
```

### 方法3: 永久设置（推荐）

```bash
# 添加到 ~/.bashrc
echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc
source ~/.bashrc
```

## ✅ 已修复的代码

代码已更新，通过环境变量 `HF_HUB_DOWNLOAD_TIMEOUT` 设置超时时间：

- ✅ `src/scripts/inference.py` - 设置超时环境变量
- ✅ `src/scripts/train.py` - 设置超时环境变量
- ✅ `src/models/vae_model.py` - 设置超时环境变量

**注意**：`timeout` 参数不能直接传递给 `from_pretrained`，需要通过环境变量设置。

## 📝 详细说明

完整解决方案请查看：[Hugging Face 超时修复指南](./docs/HUGGINGFACE_TIMEOUT_FIX.md)

