# DiT 文生图教程项目

基于 Hugging Face 的 DiT (Diffusion Transformer) 文生图学习教程，包含完整的文档和可运行代码。

## ✨ 项目特点

- ✅ **基于 Hugging Face** - 使用标准化的 Hugging Face 生态系统
- ✅ **小模型配置** - 适配 RTX 4090，使用小模型和小数据集
- ✅ **完整可运行** - 确保能够成功训练和生成图像
- ✅ **详细文档** - 8 个详细教程文档，从入门到进阶
- ✅ **易于学习** - 代码注释清晰，结构模块化

## 📚 文档

**完整的系统化文档位于 `docs/` 目录，与代码完全同步！**

### 🎯 快速开始

- **[文档索引](./docs/README.md)** ⭐ - 文档总导航，从这里开始
- **[快速开始](./docs/01-快速开始.md)** ⭐ - 5分钟快速上手

### 📖 核心教程（系统化，与代码完全匹配）

1. **[01. 快速开始](./docs/01-快速开始.md)** - 5分钟运行第一个示例
2. **[02. 安装配置](./docs/02-安装配置.md)** - 环境搭建和依赖安装
3. **[03. 数据准备](./docs/03-数据准备.md)** - 准备训练数据（COCO、CUB、自定义）
4. **[04. 模型架构](./docs/04-模型架构.md)** - 深入理解 DiT 模型结构
5. **[05. 训练指南](./docs/05-训练指南.md)** - 完整的训练流程和技巧
6. **[06. 推理使用](./docs/06-推理使用.md)** - 使用训练好的模型生成图像

### 📋 参考文档

7. **[07. 配置参考](./docs/07-配置参考.md)** - 配置文件详解
8. **[08. 故障排除](./docs/08-故障排除.md)** - 常见问题和解决方案
9. **[09. API 参考](./docs/09-API参考.md)** - 代码 API 文档

**所有文档都与代码完全匹配，确保示例和说明的准确性！**

## 🚀 快速开始

### 5 分钟快速上手

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 准备数据（以 CUB 为例）
python src/scripts/prepare_cub_from_kaggle.py \
    --kaggle_dir ./data/cub_raw/cub2002011 \
    --output ./data/cub_subset \
    --num_samples 5000

# 3. 开始训练
python src/scripts/train.py --config configs/train_config.yaml

# 4. 生成图像
python src/scripts/inference.py \
    --checkpoint ./outputs/checkpoint-epoch-10 \
    --prompt "a bird with colorful feathers"
```

**详细教程：**
- 📖 [文档索引](./docs/README.md) - 完整文档导航
- ⚡ [快速开始](./docs/01-快速开始.md) - 5分钟快速上手
- 📚 [所有教程](./docs/) - 系统化教程体系

## 📁 项目结构

```
t2i2/
├── docs/                    # 文档目录
│   ├── README.md           # 文档首页
│   ├── 01-入门指南.md
│   ├── 02-环境配置.md
│   ├── 03-数据准备.md
│   ├── 04-模型架构.md
│   ├── 05-训练流程.md
│   ├── 06-推理使用.md
│   ├── 07-常见问题.md
│   └── 08-进阶学习.md
│
├── src/                    # 源代码目录
│   ├── models/             # 模型定义
│   │   ├── dit_model.py    # DiT 模型
│   │   └── vae_model.py    # VAE 编码器/解码器
│   ├── data/               # 数据处理
│   │   ├── dataset.py      # 数据集类
│   │   └── transforms.py   # 数据预处理
│   ├── training/           # 训练相关
│   │   ├── trainer.py      # 训练器
│   │   ├── scheduler.py    # 扩散调度器
│   │   └── loss.py         # 损失函数
│   ├── inference/          # 推理相关
│   │   └── generator.py    # 图像生成器
│   ├── utils/              # 工具函数
│   │   ├── config.py       # 配置管理
│   │   ├── logger.py       # 日志工具
│   │   └── visualization.py # 可视化工具
│   └── scripts/            # 可执行脚本
│       ├── train.py         # 训练脚本
│       ├── inference.py     # 推理脚本
│       └── prepare_data.py  # 数据准备脚本
│
├── configs/                # 配置文件
│   ├── train_config.yaml   # 训练配置
│   └── model_config.yaml    # 模型配置
│
├── requirements.txt        # Python 依赖
├── setup.py               # 安装脚本
└── README.md              # 本文件
```

## 🛠️ 技术栈

- **PyTorch** - 深度学习框架
- **Hugging Face Diffusers** - 扩散模型库
- **Hugging Face Transformers** - Transformer 模型库
- **Hugging Face Accelerate** - 训练加速
- **Hugging Face Datasets** - 数据集管理

## ⚙️ 配置说明

### 小模型配置（适配 RTX 4090）

项目使用小模型配置，适合学习和实验：

- **隐藏层维度**：384
- **Transformer 层数**：8
- **注意力头数**：6
- **图像尺寸**：256x256
- **批次大小**：4（配合梯度累积）

详细配置请查看 `configs/train_config.yaml`。

## 📋 系统要求

### 硬件要求

- **GPU**：NVIDIA GPU（推荐 RTX 4090 或更高）
  - 显存：至少 24GB
  - CUDA 计算能力：7.5 或更高
- **内存**：至少 16GB RAM
- **存储**：至少 50GB 可用空间

### 软件要求

- **Python**：3.8 或更高版本（推荐 3.9 或 3.10）
- **CUDA**：11.8 或更高版本
- **操作系统**：Linux（推荐）或 Windows

## 📖 学习路径

### 初学者

1. 阅读 [入门指南](./docs/01-入门指南.md) 了解基本概念
2. 按照 [环境配置](./docs/02-环境配置.md) 搭建环境
3. 使用示例数据运行训练
4. 尝试生成图像

### 进阶用户

1. 深入理解 [模型架构](./docs/04-模型架构.md)
2. 优化训练参数
3. 探索 [进阶学习](./docs/08-进阶学习.md) 内容

## 🐛 常见问题

遇到问题？请查看 **[故障排除文档](./docs/08-故障排除.md)**，包含所有已知问题和解决方案。

常见问题包括：
- 网络问题（下载超时）→ [故障排除 - 网络问题](./docs/08-故障排除.md#网络问题)
- 显存问题（OOM）→ [故障排除 - 显存问题](./docs/08-故障排除.md#显存问题)
- 数据问题 → [故障排除 - 数据问题](./docs/08-故障排除.md#数据问题)
- 训练问题 → [故障排除 - 训练问题](./docs/08-故障排除.md#训练问题)
- 推理问题 → [故障排除 - 推理问题](./docs/08-故障排除.md#推理问题)

## 📝 许可证

本项目采用 Apache 2.0 许可证。详见 [LICENSE](./LICENSE) 文件。

## 🙏 致谢

本项目基于以下开源项目和技术：

- [Hugging Face Diffusers](https://github.com/huggingface/diffusers)
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [DiT (Facebook Research)](https://github.com/facebookresearch/DiT)

## 📧 贡献

欢迎提交 Issue 和 Pull Request！

## 🎯 下一步

1. 阅读 [文档索引](./docs/README.md) 了解文档结构
2. 按照 [快速开始](./docs/01-快速开始.md) 运行第一个示例
3. 系统学习 [核心教程](./docs/) 内容

---

**祝你学习愉快！** 🎉
