# Hugging Face 代理设置指南

## 🔍 问题

从 Hugging Face 下载模型时出现超时错误，即使使用镜像也可能超时。

## ✅ 解决方案

### 方法1: 使用 HTTP/HTTPS 代理（推荐）

如果你有可用的代理（如 Clash、V2Ray 等），可以设置环境变量：

```bash
# 设置代理（根据你的实际代理地址修改）
export HTTP_PROXY=http://127.0.0.1:7890
export HTTPS_PROXY=http://127.0.0.1:7890

# 或者使用 socks5 代理
export HTTP_PROXY=socks5://127.0.0.1:1080
export HTTPS_PROXY=socks5://127.0.0.1:1080

# 然后运行训练
python src/scripts/train.py --config configs/train_config.yaml
```

### 方法2: 使用脚本设置

```bash
# 编辑 setup_proxy.sh，设置你的代理地址
# 然后运行
source setup_proxy.sh
python src/scripts/train.py --config configs/train_config.yaml
```

### 方法3: 永久设置（推荐）

将代理设置添加到 `~/.bashrc` 或 `~/.zshrc`：

```bash
# 添加到 ~/.bashrc
echo 'export HTTP_PROXY=http://127.0.0.1:7890' >> ~/.bashrc
echo 'export HTTPS_PROXY=http://127.0.0.1:7890' >> ~/.bashrc
echo 'export HF_HUB_DOWNLOAD_TIMEOUT=600' >> ~/.bashrc
echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc

# 重新加载配置
source ~/.bashrc
```

## 🔧 常见代理端口

- **Clash**: 通常是 `7890` (HTTP) 或 `7891` (SOCKS5)
- **V2Ray**: 通常是 `1080` (SOCKS5) 或 `10808` (HTTP)
- **Shadowsocks**: 通常是 `1080` (SOCKS5)

## 📝 验证代理

```bash
# 检查代理是否工作
curl -I --proxy http://127.0.0.1:7890 https://huggingface.co

# 或者在 Python 中测试
python3 << 'EOF'
import os
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
import requests
response = requests.get('https://huggingface.co', timeout=10)
print(f"状态码: {response.status_code}")
EOF
```

## ⚠️ 注意事项

1. **代理地址**: 请根据你的实际代理软件和配置修改端口
2. **代理类型**: HTTP 代理和 SOCKS5 代理的格式不同
3. **超时设置**: 已增加到 600 秒，如果仍然超时，可以进一步增加

## 🚀 快速开始

```bash
# 1. 设置代理（替换为你的实际代理地址）
export HTTP_PROXY=http://127.0.0.1:7890
export HTTPS_PROXY=http://127.0.0.1:7890

# 2. 运行训练
python src/scripts/train.py --config configs/train_config.yaml
```

---

**状态**: ✅ 已更新代码支持代理设置

