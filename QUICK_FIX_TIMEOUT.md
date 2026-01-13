# 快速修复 Hugging Face 下载超时

## 🔍 问题

训练时遇到 Hugging Face 下载超时：
```
ReadTimeoutError: HTTPSConnectionPool(host='huggingface.co', port=443): Read timed out. (read timeout=10)
```

## ✅ 解决方案

### 方法1: 使用镜像（推荐，最快）

```bash
# 设置镜像和超时
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DOWNLOAD_TIMEOUT=300

# 然后运行训练
python src/scripts/train.py --config configs/train_config.yaml
```

或者使用脚本：
```bash
source setup_hf_mirror.sh
python src/scripts/train.py --config configs/train_config.yaml
```

### 方法2: 只增加超时时间

```bash
export HF_HUB_DOWNLOAD_TIMEOUT=300
python src/scripts/train.py --config configs/train_config.yaml
```

### 方法3: 在训练脚本中设置（已修复）

代码已更新，会自动设置超时为 300 秒。如果仍然超时，使用方法1（镜像）。

## 📝 已修复的文件

- ✅ `src/scripts/train.py` - 超时增加到 300 秒
- ✅ `src/models/vae_model.py` - 超时增加到 300 秒
- ✅ `setup_hf_mirror.sh` - 超时增加到 300 秒

## 🚀 立即使用

```bash
# 使用镜像（推荐）
source setup_hf_mirror.sh
python src/scripts/train.py --config configs/train_config.yaml
```

---

**状态**: ✅ 已修复  
**建议**: 使用镜像可以显著提升下载速度
