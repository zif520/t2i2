#!/bin/bash
# 测试断点续传功能

echo "=== 断点续传测试 ==="
echo ""
echo "可用的检查点："
ls -1td outputs/checkpoint-* 2>/dev/null | head -5
echo ""
echo "最新检查点："
LATEST=$(ls -1td outputs/checkpoint-* 2>/dev/null | head -1)
if [ -n "$LATEST" ]; then
    echo "  $LATEST"
    echo ""
    echo "训练状态："
    if [ -f "$LATEST/training_state.json" ]; then
        cat "$LATEST/training_state.json"
    else
        echo "  训练状态文件不存在（将从检查点名称推断）"
    fi
    echo ""
    echo "恢复训练命令："
    echo "  ./run_train.sh --config configs/train_config.yaml --resume $LATEST"
else
    echo "  没有找到检查点"
fi
