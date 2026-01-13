# 推理使用指南

## 🚀 快速开始

### 基本推理命令

```bash
python src/scripts/inference.py \
    --config configs/train_config.yaml \
    --checkpoint ./outputs/checkpoint-5000 \
    --prompt "a cat sitting on a chair" \
    --output ./outputs/generated
```

## 📋 参数说明

### 必需参数

- `--checkpoint`: 模型检查点路径（目录或模型文件）
  - 示例: `./outputs/checkpoint-5000` 或 `./outputs/checkpoint-5000/model.pt`

- `--prompt`: 文本提示（要生成的图像描述）
  - 示例: `"a cat sitting on a chair"`

### 可选参数

- `--config`: 配置文件路径（默认: `configs/train_config.yaml`）
- `--output`: 输出目录（默认: `./outputs/generated`）
- `--num_inference_steps`: 推理步数（默认: 50）
  - 更多步数 = 更好质量，但更慢
  - 推荐: 20-100
- `--height`: 图像高度（默认: 256）
- `--width`: 图像宽度（默认: 256）
- `--seed`: 随机种子（可选，用于可重复结果）

## 💡 使用示例

### 示例 1: 基本推理

```bash
python src/scripts/inference.py \
    --checkpoint ./outputs/checkpoint-5000 \
    --prompt "a beautiful landscape with mountains"
```

### 示例 2: 高质量推理（更多步数）

```bash
python src/scripts/inference.py \
    --checkpoint ./outputs/checkpoint-5000 \
    --prompt "a cat sitting on a chair" \
    --num_inference_steps 100 \
    --output ./outputs/generated
```

### 示例 3: 快速推理（更少步数）

```bash
python src/scripts/inference.py \
    --checkpoint ./outputs/checkpoint-5000 \
    --prompt "a dog playing in the park" \
    --num_inference_steps 20 \
    --output ./outputs/quick_test
```

### 示例 4: 指定随机种子（可重复结果）

```bash
python src/scripts/inference.py \
    --checkpoint ./outputs/checkpoint-5000 \
    --prompt "a red car on the street" \
    --seed 42 \
    --output ./outputs/generated
```

### 示例 5: 批量生成（使用循环）

```bash
# 生成多个提示词
for prompt in "a cat" "a dog" "a bird"; do
    python src/scripts/inference.py \
        --checkpoint ./outputs/checkpoint-5000 \
        --prompt "$prompt" \
        --output ./outputs/generated
done
```

## 📝 提示词技巧

### 好的提示词

- **具体描述**: "a red cat sitting on a wooden chair"
- **包含细节**: "a cat with green eyes, sitting on a chair, indoor lighting"
- **指定风格**: "a cat sitting on a chair, oil painting style"

### 提示词模板

```
[主体] + [动作/状态] + [环境] + [风格/质量]
```

示例：
- "a cat sitting on a chair, indoor, photorealistic"
- "a landscape with mountains, sunset, oil painting"

## ⚙️ 推理参数调优

### 推理步数 (num_inference_steps)

- **20-30 步**: 快速测试，质量一般
- **50 步**（默认）: 平衡质量和速度
- **100 步**: 高质量，但较慢

### 图像尺寸

- 必须与训练时的尺寸匹配（或按比例缩放）
- 当前配置: 256x256

### 随机种子

- 不指定: 每次生成不同结果
- 指定种子: 可重复的结果（用于对比）

## 🔍 检查点路径

检查点通常保存在：

```
outputs/
├── checkpoint-500/
│   ├── model.pt
│   ├── optimizer.pt
│   └── training_state.json
├── checkpoint-1000/
│   └── ...
└── checkpoint-epoch-1/
    └── ...
```

使用检查点时，可以指定：
- 目录: `--checkpoint ./outputs/checkpoint-5000`
- 模型文件: `--checkpoint ./outputs/checkpoint-5000/model.pt`

## 🐛 常见问题

### 1. 检查点不存在

**错误**: `FileNotFoundError: 模型检查点不存在`

**解决**: 检查路径是否正确，确保模型已训练并保存

### 2. 维度不匹配

**错误**: 模型维度与配置不匹配

**解决**: 确保使用与训练时相同的配置文件

### 3. 显存不足

**解决**: 
- 减小图像尺寸
- 使用更少的推理步数
- 关闭其他程序释放显存

## 📊 推理性能

- **推理时间**: 约 2-5 秒/图像（取决于步数）
- **显存使用**: 约 2-4 GB
- **输出格式**: PNG 图像

## 🎨 生成结果

生成的图像保存在指定的输出目录：

```
outputs/generated/
├── a_cat_sitting_on_a_chair.png
├── a_beautiful_landscape.png
└── ...
```

## 💻 使用代码推理（高级）

如果需要批量推理或自定义流程，可以使用代码：

```python
from src.inference.generator import ImageGenerator
from src.models.dit_model import DiTModel
from src.models.vae_model import VAEDecoder
from transformers import CLIPTextModel, CLIPTokenizer
import torch

# 加载模型
device = torch.device("cuda")
model = DiTModel(...).to(device)
model.load_state_dict(torch.load("checkpoint/model.pt"))

# 创建生成器
generator = ImageGenerator(
    model=model,
    vae_decoder=vae_decoder,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    device=device,
)

# 生成图像
image = generator.generate(
    prompt="a cat sitting on a chair",
    num_inference_steps=50,
)
image.save("output.png")
```

---

**开始生成你的第一张图像吧！** 🎨

