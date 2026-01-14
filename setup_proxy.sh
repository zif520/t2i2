#!/bin/bash
# 设置 Hugging Face 国内代理和超时配置

echo "=== 设置 Hugging Face 代理配置 ===\n"

# 方法1: 使用 Hugging Face 镜像（如果可用）
export HF_ENDPOINT=https://hf-mirror.com

# 方法2: 设置 HTTP/HTTPS 代理（根据你的实际代理地址修改）
# 示例代理地址（请替换为你的实际代理）
# export HTTP_PROXY=http://127.0.0.1:7890
# export HTTPS_PROXY=http://127.0.0.1:7890

# 或者使用环境变量中的代理（如果已设置）
if [ -z "$HTTP_PROXY" ]; then
    echo "⚠️  HTTP_PROXY 未设置，如果需要使用代理，请设置："
    echo "   export HTTP_PROXY=http://your-proxy:port"
    echo "   export HTTPS_PROXY=http://your-proxy:port"
else
    echo "✓ 使用已设置的代理: $HTTP_PROXY"
fi

# 设置下载超时（增加到 600 秒）
export HF_HUB_DOWNLOAD_TIMEOUT=600

# 设置 requests 库的超时
export REQUESTS_TIMEOUT=600

echo "✓ 已设置 HF_ENDPOINT=$HF_ENDPOINT"
echo "✓ 已设置 HF_HUB_DOWNLOAD_TIMEOUT=$HF_HUB_DOWNLOAD_TIMEOUT"
echo "✓ 已设置 REQUESTS_TIMEOUT=$REQUESTS_TIMEOUT"

echo "\n使用方法："
echo "  source setup_proxy.sh"
echo "  python src/scripts/train.py --config configs/train_config.yaml"

