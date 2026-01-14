# 快速设置国内代理

## 🚀 快速开始

### 方法1: 直接设置代理（推荐）

```bash
# 设置代理（根据你的实际代理地址修改）
export HTTP_PROXY=http://127.0.0.1:7890
export HTTPS_PROXY=http://127.0.0.1:7890

# 设置超时
export HF_HUB_DOWNLOAD_TIMEOUT=600
export HF_ENDPOINT=https://hf-mirror.com

# 运行训练
python src/scripts/train.py --config configs/train_config.yaml
```

### 方法2: 使用脚本

```bash
# 1. 编辑 setup_proxy.sh，设置你的代理地址
# 2. 运行
source setup_proxy.sh
python src/scripts/train.py --config configs/train_config.yaml
```

## 🔧 常见代理端口

- **Clash**: `7890` (HTTP) 或 `7891` (SOCKS5)
- **V2Ray**: `1080` (SOCKS5) 或 `10808` (HTTP)
- **Shadowsocks**: `1080` (SOCKS5)

## 📝 示例

```bash
# Clash 代理示例
export HTTP_PROXY=http://127.0.0.1:7890
export HTTPS_PROXY=http://127.0.0.1:7890

# V2Ray SOCKS5 代理示例
export HTTP_PROXY=socks5://127.0.0.1:1080
export HTTPS_PROXY=socks5://127.0.0.1:1080
```

## ✅ 验证

```bash
# 检查代理是否工作
curl -I --proxy http://127.0.0.1:7890 https://huggingface.co
```

---

**注意**: 请将代理地址替换为你实际的代理地址和端口！

