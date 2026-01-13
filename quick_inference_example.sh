#!/bin/bash
# 快速推理示例脚本

# 检查点路径（请根据实际情况修改）
CHECKPOINT="./outputs/checkpoint-5000"

# 如果检查点不存在，尝试其他路径
if [ ! -d "$CHECKPOINT" ] && [ ! -f "$CHECKPOINT" ]; then
    # 尝试查找最新的检查点
    LATEST_CHECKPOINT=$(ls -td outputs/checkpoint-* 2>/dev/null | head -1)
    if [ -n "$LATEST_CHECKPOINT" ]; then
        CHECKPOINT="$LATEST_CHECKPOINT"
        echo "使用最新检查点: $CHECKPOINT"
    else
        echo "错误: 找不到检查点，请先训练模型"
        exit 1
    fi
fi

# 推理命令
python src/scripts/inference.py \
    --config configs/train_config.yaml \
    --checkpoint "$CHECKPOINT" \
    --prompt "a cat sitting on a chair" \
    --output ./outputs/generated \
    --num_inference_steps 50

echo ""
echo "✅ 推理完成！图像保存在: ./outputs/generated/"
