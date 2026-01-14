"""处理从 Kaggle 下载的 CUB-200-2011 数据集"""

import argparse
import json
import sys
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import random

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def prepare_cub_from_kaggle(
    kaggle_dir: str,
    output_dir: str = "./data/cub_subset",
    num_samples: int = None,
    use_train_split: bool = True,
):
    """处理从 Kaggle 下载的 CUB 数据集"""
    kaggle_dir = Path(kaggle_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)
    
    print(f"\n=== 处理 CUB 数据集（Kaggle 版本）===\n")
    print(f"源目录: {kaggle_dir}")
    print(f"输出目录: {output_dir}\n")
    
    # 确定数据路径
    base_path = kaggle_dir / "versions/7"
    images_path = base_path / "CUB_200_2011/images"
    text_path = base_path / "cvpr2016_cub/text_c10"
    
    if not images_path.exists():
        raise ValueError(f"找不到图像目录: {images_path}")
    if not text_path.exists():
        raise ValueError(f"找不到文本描述目录: {text_path}")
    
    # 不使用划分文件，直接处理所有图像（更简单）
    print("ℹ️  将处理所有图像（不使用划分文件）")
    
    # 收集所有图像和文本描述
    metadata = []
    count = 0
    
    print("收集图像和文本描述...")
    
    # 遍历所有类别目录
    for class_dir in tqdm(sorted(images_path.iterdir()), desc="处理类别"):
        if not class_dir.is_dir():
            continue
        
        class_name = class_dir.name
        # 提取类别名称（去掉编号，如 "136.Barn_Swallow" -> "Barn Swallow"）
        if '.' in class_name:
            class_display_name = class_name.split('.', 1)[1].replace('_', ' ')
        else:
            class_display_name = class_name.replace('_', ' ')
        
        # 对应的文本描述目录
        text_class_dir = text_path / class_name
        
        # 遍历该类别的所有图像
        for image_file in sorted(class_dir.glob("*.jpg")):
            image_id = image_file.stem  # 例如: "Barn_Swallow_0027_129978"
            
            # 读取文本描述
            text_descriptions = []
            if text_class_dir.exists():
                # 查找对应的文本文件（可能有多个）
                text_files = list(text_class_dir.glob(f"{image_id}*.txt"))
                for text_file in text_files:
                    try:
                        with open(text_file, 'r', encoding='utf-8') as f:
                            # 读取所有行，每行是一个描述
                            lines = f.readlines()
                            for line in lines:
                                line = line.strip()
                                if line:
                                    text_descriptions.append(line)
                    except Exception as e:
                        print(f"读取文本文件失败 {text_file}: {e}")
                        continue
            
            # 如果没有文本描述，使用类别名称生成
            if not text_descriptions:
                text_descriptions = [f"a photo of a {class_display_name.lower()}"]
            
            # 随机选择一个文本描述（只使用一个描述，不是多行）
            text = random.choice(text_descriptions)
            
            try:
                # 加载并调整图像大小
                image = Image.open(image_file).convert("RGB")
                image = image.resize((256, 256), Image.Resampling.LANCZOS)
                
                # 保存图像
                output_image_name = f"image_{count:06d}.jpg"
                output_image_path = images_dir / output_image_name
                image.save(output_image_path, quality=95)
                
                # 保存元数据
                metadata.append({
                    "image": f"images/{output_image_name}",
                    "text": text,
                })
                
                count += 1
                
                # 如果指定了样本数，达到后停止
                if num_samples and count >= num_samples:
                    break
                    
            except Exception as e:
                print(f"处理图像 {image_file} 时出错: {e}")
                continue
        
        # 如果已达到样本数，停止
        if num_samples and count >= num_samples:
            break
    
    # 保存元数据
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 数据准备完成！")
    print(f"输出目录: {output_dir}")
    print(f"元数据文件: {metadata_path}")
    print(f"图像目录: {images_dir}")
    print(f"共处理 {len(metadata)} 个样本")
    
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="处理从 Kaggle 下载的 CUB-200-2011 数据集")
    parser.add_argument(
        "--kaggle_dir",
        type=str,
        default="./data/cub_raw/cub2002011",
        help="Kaggle 下载的 CUB 数据集目录",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./data/cub_subset",
        help="输出目录",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="使用的样本数量（None 表示使用全部）",
    )
    parser.add_argument(
        "--use_train",
        action="store_true",
        default=True,
        help="使用训练集（默认）",
    )
    parser.add_argument(
        "--use_val",
        action="store_true",
        help="使用验证集",
    )
    
    args = parser.parse_args()
    
    use_train = args.use_train and not args.use_val
    
    prepare_cub_from_kaggle(
        kaggle_dir=args.kaggle_dir,
        output_dir=args.output,
        num_samples=args.num_samples,
        use_train_split=use_train,
    )


if __name__ == "__main__":
    main()

