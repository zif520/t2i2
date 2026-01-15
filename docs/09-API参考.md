# 09. API 参考

本指南提供项目所有主要类和函数的 API 文档，基于实际代码。

## 📚 模块索引

- [模型模块](#模型模块)
- [数据处理模块](#数据处理模块)
- [训练模块](#训练模块)
- [推理模块](#推理模块)
- [工具模块](#工具模块)

## 🧠 模型模块

### `DiTModel`

**位置**: `src/models/dit_model.py`

**类定义**:
```python
class DiTModel(nn.Module):
    def __init__(
        self,
        hidden_size: int = 384,
        num_layers: int = 8,
        num_heads: int = 6,
        patch_size: int = 2,
        in_channels: int = 4,
        out_channels: int = 4,
        attention_head_dim: int = 64,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        input_size: int = 32,
    )
```

**方法**:

#### `forward(x, t, y)`

前向传播

**参数**:
- `x` (torch.Tensor): 输入潜在表示，形状 `(B, in_channels, H, W)`
- `t` (torch.Tensor): 时间步，形状 `(B,)`
- `y` (torch.Tensor): 文本条件嵌入，形状 `(B, text_dim)` 或 `(B, seq_len, text_dim)`

**返回**:
- `torch.Tensor`: 预测的噪声，形状 `(B, out_channels, H, W)`

### `VAEEncoder`

**位置**: `src/models/vae_model.py`

**类定义**:
```python
class VAEEncoder:
    def __init__(
        self,
        pretrained_model_name: str = "runwayml/stable-diffusion-v1-5",
        use_slicing: bool = False,
    )
```

**方法**:

#### `encode(images)`

将图像编码到潜在空间

**参数**:
- `images` (torch.Tensor): 图像张量，形状 `(B, 3, H, W)`，值范围 `[-1, 1]`

**返回**:
- `torch.Tensor`: 潜在表示，形状 `(B, 4, H//8, W//8)`

### `VAEDecoder`

**位置**: `src/models/vae_model.py`

**类定义**:
```python
class VAEDecoder:
    def __init__(
        self,
        pretrained_model_name: str = "runwayml/stable-diffusion-v1-5",
        use_slicing: bool = False,
    )
```

**方法**:

#### `decode(latents)`

将潜在表示解码为图像

**参数**:
- `latents` (torch.Tensor): 潜在表示，形状 `(B, 4, H, W)`

**返回**:
- `torch.Tensor`: 图像张量，形状 `(B, 3, H*8, W*8)`，值范围 `[-1, 1]`

## 📊 数据处理模块

### `TextImageDataset`

**位置**: `src/data/dataset.py`

**类定义**:
```python
class TextImageDataset(Dataset):
    def __init__(
        self,
        dataset_name: str = "coco",
        dataset_path: Optional[str] = None,
        image_size: int = 256,
        tokenizer_name: str = "openai/clip-vit-base-patch32",
        max_length: int = 77,
        num_samples: Optional[int] = None,
        is_train: bool = True,
    )
```

**支持的数据集**:
- `"coco"`: COCO 数据集
- `"cub"`: CUB-200-2011 数据集
- `"custom"`: 自定义数据集

**方法**:

#### `__getitem__(idx)`

获取数据项

**参数**:
- `idx` (int): 索引

**返回**:
- `Dict[str, torch.Tensor]`: 包含 `pixel_values` 和 `input_ids` 的字典

### `get_image_transforms`

**位置**: `src/data/transforms.py`

**函数定义**:
```python
def get_image_transforms(
    image_size: int = 256,
    is_train: bool = True,
) -> transforms.Compose
```

**返回**: 图像变换组合

**变换内容**:
- 调整大小到 `image_size`
- 中心裁剪
- 随机水平翻转（训练时）
- 转换为张量
- 归一化到 `[-1, 1]`

## 🏋️ 训练模块

### `Trainer`

**位置**: `src/training/trainer.py`

**类定义**:
```python
class Trainer:
    def __init__(
        self,
        model: DiTModel,
        vae_encoder: VAEEncoder,
        text_encoder: Any,
        train_dataloader: DataLoader,
        config: Config,
        accelerator: Optional[Accelerator] = None,
    )
```

**方法**:

#### `train()`

执行训练循环

**功能**:
- 训练循环
- 自动保存检查点
- 日志记录

#### `train_step(batch)`

执行一个训练步骤

**参数**:
- `batch` (Dict[str, torch.Tensor]): 批次数据

**返回**:
- `Dict[str, float]`: 损失字典

#### `save_checkpoint(checkpoint_dir)`

保存检查点

**参数**:
- `checkpoint_dir` (Path): 检查点目录

**保存内容**:
- `model.pt`: 模型权重
- `optimizer.pt`: 优化器状态
- `scheduler.pt`: 学习率调度器状态
- `training_state.json`: 训练状态

#### `load_checkpoint(checkpoint_dir)`

加载检查点

**参数**:
- `checkpoint_dir` (Path): 检查点目录

**功能**:
- 加载模型权重
- 恢复优化器状态
- 恢复训练进度

### `get_scheduler`

**位置**: `src/training/scheduler.py`

**函数定义**:
```python
def get_scheduler(
    scheduler_type: str = "ddpm",
    num_train_timesteps: int = 1000,
    beta_start: float = 0.00085,
    beta_end: float = 0.012,
    beta_schedule: str = "scaled_linear",
    prediction_type: str = "epsilon",
) -> DDPMScheduler | DDIMScheduler
```

**返回**: 扩散调度器对象

### `DiffusionLoss`

**位置**: `src/training/loss.py`

**类定义**:
```python
class DiffusionLoss(nn.Module):
    def __init__(self, loss_type: str = "mse")
    def forward(
        self,
        pred_noise: torch.Tensor,
        target_noise: torch.Tensor,
    ) -> torch.Tensor
```

**支持的损失类型**:
- `"mse"`: 均方误差（默认）
- `"l1"`: L1 损失

## 🎨 推理模块

### `ImageGenerator`

**位置**: `src/inference/generator.py`

**类定义**:
```python
class ImageGenerator:
    def __init__(
        self,
        model: DiTModel,
        vae_decoder: VAEDecoder,
        text_encoder: Any,
        tokenizer: Any,
        scheduler_type: str = "ddpm",
        device: Optional[torch.device] = None,
    )
```

**方法**:

#### `generate(prompt, num_inference_steps, guidance_scale, height, width, seed)`

生成图像

**参数**:
- `prompt` (str): 文本提示
- `num_inference_steps` (int): 推理步数，默认 50
- `guidance_scale` (float): 引导强度，默认 7.5
- `height` (int): 图像高度，默认 256
- `width` (int): 图像宽度，默认 256
- `seed` (Optional[int]): 随机种子

**返回**:
- `Image.Image`: 生成的图像

#### `generate_batch(prompts, ...)`

批量生成图像

**参数**:
- `prompts` (List[str]): 文本提示列表
- 其他参数同 `generate`

**返回**:
- `List[Image.Image]`: 生成的图像列表

## 🛠️ 工具模块

### `load_config`

**位置**: `src/utils/config.py`

**函数定义**:
```python
def load_config(config_path: str) -> Config
```

**参数**:
- `config_path` (str): 配置文件路径

**返回**:
- `Config`: 配置对象

### `Config`

**位置**: `src/utils/config.py`

**类定义**:
```python
@dataclass
class Config:
    data: Dict[str, Any]
    model: Dict[str, Any]
    training: Dict[str, Any]
    scheduler: Dict[str, Any]
    text_encoder: Dict[str, Any]
    optimizer: Dict[str, Any]
    lr_scheduler: Dict[str, Any]
    vae: Optional[Dict[str, Any]] = None
```

**方法**:

#### `to_dict()`

转换为字典

**返回**:
- `Dict[str, Any]`: 配置字典

### `setup_logger`

**位置**: `src/utils/logger.py`

**函数定义**:
```python
def setup_logger(
    name: str = "dit_tutorial",
    log_file: Optional[Path] = None,
) -> logging.Logger
```

**返回**: 日志记录器对象

### `tensor_to_pil_image`

**位置**: `src/utils/visualization.py`

**函数定义**:
```python
def tensor_to_pil_image(tensor: torch.Tensor) -> Image.Image
```

**参数**:
- `tensor` (torch.Tensor): 图像张量，形状 `(C, H, W)` 或 `(1, C, H, W)`，值范围 `[0, 1]`

**返回**:
- `Image.Image`: PIL 图像对象

## 📝 脚本 API

### `train.py`

**位置**: `src/scripts/train.py`

**命令行参数**:
- `--config`: 配置文件路径（默认: `configs/train_config.yaml`）
- `--resume`: 恢复训练的检查点路径（可选）

### `inference.py`

**位置**: `src/scripts/inference.py`

**命令行参数**:
- `--checkpoint`: 模型检查点路径（必需）
- `--prompt`: 文本提示（必需）
- `--config`: 配置文件路径（默认: `configs/train_config.yaml`）
- `--output`: 输出目录（默认: `./outputs/generated`）
- `--num_inference_steps`: 推理步数（默认: 50）
- `--height`: 图像高度（默认: 256）
- `--width`: 图像宽度（默认: 256）
- `--seed`: 随机种子（可选）

### `prepare_data.py`

**位置**: `src/scripts/prepare_data.py`

**命令行参数**:
- `--type`: 数据集类型（`coco` 或 `custom`）
- `--output`: 输出目录
- `--num_samples`: 样本数量

### `prepare_coco_from_download.py`

**位置**: `src/scripts/prepare_coco_from_download.py`

**命令行参数**:
- `--images_dir`: COCO 图像目录
- `--annotations_file`: COCO 标注文件
- `--output`: 输出目录
- `--num_samples`: 样本数量

### `prepare_cub_from_kaggle.py`

**位置**: `src/scripts/prepare_cub_from_kaggle.py`

**命令行参数**:
- `--kaggle_dir`: Kaggle 下载的 CUB 数据目录
- `--output`: 输出目录
- `--num_samples`: 样本数量
- `--use_train`: 使用训练集（默认）
- `--use_val`: 使用验证集

## 📝 下一步

- 📖 [01. 快速开始](./01-快速开始.md) - 使用这些 API
- 📖 [05. 训练指南](./05-训练指南.md) - 训练流程
- 📖 [06. 推理使用](./06-推理使用.md) - 推理流程

---

**API 查询**: 查看源码 `src/` 目录获取更详细的实现细节！

