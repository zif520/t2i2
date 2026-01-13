#!/bin/bash
# 推理启动脚本 - 解决临时目录问题

# 设置临时目录到 outputs 目录
export TMPDIR="$(cd "$(dirname "$0")" && pwd)/outputs"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"

# 确保目录存在
mkdir -p "$TMPDIR"

# 运行推理脚本
python src/scripts/inference.py "$@"

