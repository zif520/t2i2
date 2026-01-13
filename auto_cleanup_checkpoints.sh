#!/bin/bash
# 自动清理旧检查点脚本

KEEP=5  # 保留最新的 5 个检查点

cd "$(dirname "$0")/outputs" 2>/dev/null || exit 1

total=$(ls -1td checkpoint-* 2>/dev/null | wc -l)
to_delete=$((total - KEEP))

if [ $to_delete -gt 0 ]; then
    echo "清理旧检查点（保留最新的 $KEEP 个）..."
    ls -1td checkpoint-* 2>/dev/null | tail -n +$((KEEP + 1)) | xargs rm -rf
    echo "✓ 已删除 $to_delete 个旧检查点"
else
    echo "检查点数量 <= $KEEP，无需清理"
fi

echo ""
echo "当前磁盘使用:"
df -h / | tail -1
