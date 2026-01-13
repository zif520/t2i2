#!/bin/bash
# 设置 Hugging Face 镜像源和超时（解决下载超时问题）

echo "=== 设置 Hugging Face 配置 ==="
echo ""

# 使用国内镜像（推荐）
export HF_ENDPOINT=https://hf-mirror.com

# 设置下载超时（60秒）
export HF_HUB_DOWNLOAD_TIMEOUT=60

echo "✓ 已设置 HF_ENDPOINT=$HF_ENDPOINT"
echo "✓ 已设置 HF_HUB_DOWNLOAD_TIMEOUT=$HF_HUB_DOWNLOAD_TIMEOUT"
echo ""
echo "使用方法："
echo "  1. 运行此脚本: source setup_hf_mirror.sh"
echo "  2. 或添加到 ~/.bashrc:"
echo "     echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc"
echo "     echo 'export HF_HUB_DOWNLOAD_TIMEOUT=60' >> ~/.bashrc"
echo ""
echo "永久设置（推荐）："
echo "  echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc"
echo "  echo 'export HF_HUB_DOWNLOAD_TIMEOUT=60' >> ~/.bashrc"
echo "  source ~/.bashrc"

