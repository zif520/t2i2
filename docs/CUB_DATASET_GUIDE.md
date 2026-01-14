# CUB-200-2011 数据集下载和适配指南

## 🔍 数据集信息

CUB-200-2011 (Caltech-UCSD Birds-200-2011) 数据集：
- **200 种鸟类**
- **11,788 张图像**
- **每张图像有类别标签和文本描述**

## 📥 下载方法

### 方法1: 手动下载（推荐）

由于官方链接可能失效，建议手动下载：

1. **访问 Caltech 数据仓库**:
   - https://data.caltech.edu/records/65de6-4bqg6
   - 或搜索 "CUB-200-2011 Caltech"

2. **下载文件**:
   - 文件名: `CUB_200_2011.tgz`
   - 大小: 约 1.1 GB
   - 保存到: `./data/cub_raw/CUB_200_2011.tgz`

3. **解压**:
   ```bash
   cd data/cub_raw
   tar -xzf CUB_200_2011.tgz
   ```

### 方法2: 使用脚本（如果链接可用）

```bash
python src/scripts/download_cub.py --download --num_samples 5000
```

## 🔧 处理数据集

### 如果已下载并解压

```bash
python src/scripts/download_cub.py \
    --skip_download \
    --cub_dir ./data/cub_raw/CUB_200_2011 \
    --output ./data/cub_subset \
    --num_samples 5000
```

### 如果只有压缩文件

```bash
# 先解压
cd data/cub_raw
tar -xzf CUB_200_2011.tgz

# 然后处理
python src/scripts/download_cub.py \
    --skip_download \
    --cub_dir ./data/cub_raw/CUB_200_2011 \
    --output ./data/cub_subset \
    --num_samples 5000
```

## 📊 数据集结构

CUB 数据集解压后的结构：

```
CUB_200_2011/
├── images/
│   ├── 001.Black_footed_Albatross/
│   ├── 002.Laysan_Albatross/
│   └── ...
├── images.txt
├── image_class_labels.txt
├── classes.txt
├── train_test_split.txt
└── ...
```

## 🎯 适配后的结构

处理后会生成：

```
data/cub_subset/
├── images/
│   ├── image_000000.jpg
│   ├── image_000001.jpg
│   └── ...
└── metadata.json
```

## 🚀 使用 CUB 数据集训练

修改配置文件：

```yaml
# configs/train_config.yaml
data:
  dataset_name: "cub"
  dataset_path: "./data/cub_subset"
  num_samples: 5000
  image_size: 256
```

然后训练：

```bash
python src/scripts/train.py --config configs/train_config.yaml
```

## ⚠️ 注意事项

1. **文本描述**: CUB 数据集没有现成的文本描述，脚本会基于类别名称生成描述
2. **数据量**: 建议使用 5000-10000 个样本进行训练
3. **类别**: 200 种鸟类，适合细粒度图像生成任务

---

**状态**: ✅ 脚本已创建，支持手动下载后处理

