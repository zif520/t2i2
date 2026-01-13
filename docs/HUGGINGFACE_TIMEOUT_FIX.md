# Hugging Face 下载超时问题解决方案

## 🔍 问题描述

从 Hugging Face 下载模型时出现超时错误：

```
ReadTimeoutError: HTTPSConnectionPool(host='huggingface.co', port=443): Read timed out. (read timeout=10)
```

## 🚀 解决方案

### 方案1: 增加超时时间（推荐）

修改代码，增加 `from_pretrained` 的超时参数：

```python
# 原来
model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")

# 修改为
model = CLIPTextModel.from_pretrained(
    "openai/clip-vit-base-patch32",
    timeout=60  # 增加到 60 秒
)
```

### 方案2: 使用镜像源

设置环境变量使用 Hugging Face 镜像：

```bash
# 使用国内镜像
export HF_ENDPOINT=https://hf-mirror.com

# 或使用其他镜像
export HF_ENDPOINT=https://huggingface.co
```

### 方案3: 配置代理

如果网络不稳定，可以配置代理：

```bash
export HTTP_PROXY=http://your-proxy:port
export HTTPS_PROXY=http://your-proxy:port
```

### 方案4: 离线使用已下载的模型

如果模型已经下载过，可以指定本地路径：

```python
# 使用本地缓存
model = CLIPTextModel.from_pretrained(
    "openai/clip-vit-base-patch32",
    local_files_only=True  # 只使用本地文件
)
```

### 方案5: 手动下载模型

如果网络问题持续，可以手动下载模型到本地：

```bash
# 使用 huggingface-cli
huggingface-cli download openai/clip-vit-base-patch32 --local-dir ./models/clip-vit-base-patch32

# 然后使用本地路径
model = CLIPTextModel.from_pretrained("./models/clip-vit-base-patch32")
```

## 🔧 代码修改

### 修改所有 from_pretrained 调用

需要在以下文件中添加 `timeout` 参数：

1. `src/scripts/train.py` - 文本编码器和 VAE
2. `src/scripts/inference.py` - 文本编码器和 VAE
3. `src/models/vae_model.py` - VAE 模型
4. 其他使用 `from_pretrained` 的地方

### 示例修改

```python
# 文本编码器
text_encoder = CLIPTextModel.from_pretrained(
    text_encoder_name,
    timeout=60,  # 增加超时时间
)

# VAE
vae = AutoencoderKL.from_pretrained(
    pretrained_model_name,
    timeout=60,  # 增加超时时间
)
```

## 📝 环境变量配置

### 永久设置（推荐）

添加到 `~/.bashrc` 或 `~/.zshrc`：

```bash
# Hugging Face 配置
export HF_ENDPOINT=https://hf-mirror.com  # 使用镜像
export HF_HOME=~/.cache/huggingface  # 缓存目录
export TRANSFORMERS_CACHE=~/.cache/huggingface/transformers
```

### 临时设置

```bash
# 当前会话有效
export HF_ENDPOINT=https://hf-mirror.com
```

## 🎯 快速修复

### 方法1: 设置环境变量（最快）

```bash
# 使用镜像源
export HF_ENDPOINT=https://hf-mirror.com

# 重新运行
./run_inference.sh --checkpoint ./outputs/checkpoint-epoch-118 --prompt "a cat"
```

### 方法2: 修改代码增加超时

修改 `src/scripts/inference.py` 和 `src/scripts/train.py`，在所有 `from_pretrained` 调用中添加 `timeout=60`。

## ⚠️ 注意事项

1. **超时时间**：根据网络情况调整，建议 60-120 秒
2. **镜像源**：某些镜像可能不是最新的
3. **缓存**：模型下载后会缓存，后续使用会更快
4. **网络稳定性**：如果网络不稳定，建议使用代理或镜像

## 🔍 检查模型是否已下载

```bash
# 查看 Hugging Face 缓存
ls -lh ~/.cache/huggingface/hub/

# 查看特定模型
ls -lh ~/.cache/huggingface/hub/models--openai--clip-vit-base-patch32/
```

## 💡 最佳实践

1. **首次运行前**：设置镜像源或代理
2. **网络不稳定时**：增加超时时间
3. **离线环境**：手动下载模型到本地
4. **生产环境**：使用本地模型路径

---

**推荐操作**：设置镜像源环境变量，这是最快的解决方案。

