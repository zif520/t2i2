# 推理快速开始

## 🚀 基本命令

```bash
python src/scripts/inference.py \
    --checkpoint ./outputs/checkpoint-5000 \
    --prompt "a cat sitting on a chair"
```

## 📋 完整参数示例

```bash
python src/scripts/inference.py \
    --config configs/train_config.yaml \
    --checkpoint ./outputs/checkpoint-5000 \
    --prompt "a beautiful landscape with mountains" \
    --output ./outputs/generated \
    --num_inference_steps 50 \
    --height 256 \
    --width 256 \
    --seed 42
```

## 💡 常用示例

### 快速测试（20步）
```bash
python src/scripts/inference.py \
    --checkpoint ./outputs/checkpoint-5000 \
    --prompt "a cat" \
    --num_inference_steps 20
```

### 高质量生成（100步）
```bash
python src/scripts/inference.py \
    --checkpoint ./outputs/checkpoint-5000 \
    --prompt "a cat sitting on a chair" \
    --num_inference_steps 100
```

### 使用最新检查点
```bash
# 查找最新检查点
LATEST=$(ls -td outputs/checkpoint-* 2>/dev/null | head -1)

python src/scripts/inference.py \
    --checkpoint "$LATEST" \
    --prompt "a beautiful landscape"
```

## 📖 详细文档

查看 `docs/INFERENCE_GUIDE.md` 获取完整说明。
