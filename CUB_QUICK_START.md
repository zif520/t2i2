# CUB-200-2011 数据集快速开始

## 📥 下载数据集

### 步骤1: 手动下载

由于自动下载链接失效，需要手动下载：

1. **访问下载页面**:
   - https://data.caltech.edu/records/65de6-4bqg6
   - 或搜索 "CUB-200-2011 dataset download"

2. **下载文件**:
   - 文件名: `CUB_200_2011.tgz`
   - 大小: 约 1.1 GB
   - 保存到: `./data/cub_raw/CUB_200_2011.tgz`

### 步骤2: 解压

```bash
cd data/cub_raw
tar -xzf CUB_200_2011.tgz
```

解压后会得到 `CUB_200_2011/` 目录。

## 🔧 处理数据集

```bash
python src/scripts/download_cub.py \
    --skip_download \
    --cub_dir ./data/cub_raw/CUB_200_2011 \
    --output ./data/cub_subset \
    --num_samples 5000
```

这会：
- 读取 CUB 数据集的图像和类别信息
- 生成文本描述（基于类别名称）
- 调整图像大小到 256x256
- 保存为项目需要的格式

## 🚀 使用 CUB 数据集训练

### 修改配置

编辑 `configs/train_config.yaml`:

```yaml
data:
  dataset_name: "cub"  # 改为 cub
  dataset_path: "./data/cub_subset"
  num_samples: 5000
  image_size: 256
```

### 开始训练

```bash
python src/scripts/train.py --config configs/train_config.yaml
```

## ✅ 验证

处理完成后，检查数据：

```bash
# 检查元数据
cat data/cub_subset/metadata.json | head -20

# 检查图像
ls data/cub_subset/images/ | head -10
```

## 📊 数据集信息

- **类别数**: 200 种鸟类
- **图像数**: 11,788 张（全部）或 5,000 张（子集）
- **文本描述**: 基于类别名称生成（如 "a photo of a Black_footed_Albatross"）

---

**注意**: 如果已下载数据集，直接运行处理脚本即可！

