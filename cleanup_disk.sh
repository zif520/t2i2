#!/bin/bash
# 清理磁盘空间脚本

echo "=== 当前磁盘使用情况 ==="
df -h / | tail -1

echo -e "\n=== 检查点文件统计 ==="
cd "$(dirname "$0")/outputs" 2>/dev/null || exit 1
total_checkpoints=$(ls -1 checkpoint-* 2>/dev/null | wc -l)
echo "总检查点数: $total_checkpoints"

if [ "$total_checkpoints" -gt 5 ]; then
    echo -e "\n=== 清理旧的检查点（保留最新的 5 个）==="
    to_delete=$(ls -1t checkpoint-* 2>/dev/null | tail -n +6)
    if [ -n "$to_delete" ]; then
        echo "将删除以下检查点:"
        echo "$to_delete" | head -10
        if [ $(echo "$to_delete" | wc -l) -gt 10 ]; then
            echo "... 还有更多"
        fi
        echo ""
        read -p "确认删除? (y/N): " confirm
        if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
            echo "$to_delete" | xargs rm -rf
            echo "✓ 清理完成"
        else
            echo "取消清理"
        fi
    fi
else
    echo "检查点数量 <= 5，无需清理"
fi

echo -e "\n=== 清理后的磁盘使用情况 ==="
df -h / | tail -1

