"""数据准备脚本"""

import argparse
import os
import sys
import json
from pathlib import Path

# 在导入其他库之前设置临时目录（如果系统临时目录不可用）
project_root = Path(__file__).parent.parent.parent
if not os.environ.get("TMPDIR"):
    # 使用 outputs 目录下的 tmp 子目录
    tmp_dir = project_root / "outputs" / "tmp"
    try:
        tmp_dir.mkdir(parents=True, exist_ok=True)
        os.environ["TMPDIR"] = str(tmp_dir)
        os.environ["TMP"] = str(tmp_dir)
        os.environ["TEMP"] = str(tmp_dir)
    except (OSError, PermissionError):
        # 如果无法创建，尝试使用 outputs 目录本身
        os.environ["TMPDIR"] = str(project_root / "outputs")
        os.environ["TMP"] = str(project_root / "outputs")
        os.environ["TEMP"] = str(project_root / "outputs")

from PIL import Image
from tqdm import tqdm

# 添加项目根目录到路径，支持直接运行脚本
sys.path.insert(0, str(project_root))


def prepare_coco_subset(output_dir: str, num_samples: int = 5000):
    """
    准备 COCO 数据集子集
    
    注意: 此函数尝试从 Hugging Face 下载真实的 COCO 数据。
    如果下载失败，会提示用户手动准备数据。
    
    Args:
        output_dir: 输出目录
        num_samples: 样本数量
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)
    
    metadata = []
    
    print(f"准备 {num_samples} 个样本...")
    print("尝试从 Hugging Face 下载真实的 COCO 数据集...")
    
    # 尝试从 Hugging Face 加载真实的 COCO 数据
    try:
        from datasets import load_dataset
        import os
        os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
        
        print("正在下载 COCO 数据集（这可能需要一些时间）...")
        # 尝试多个 COCO 数据集源
        coco_sources = [
            "detection-datasets/coco_2017_val_panoptic",
            "HuggingFaceM4/COCO",
            "coco",
        ]
        
        dataset = None
        for source in coco_sources:
            try:
                print(f"尝试从 {source} 加载...")
                dataset = load_dataset(source, split="train", streaming=False)
                print(f"✅ 成功从 {source} 加载数据集")
                break
            except Exception as e:
                print(f"  ❌ 失败: {e}")
                continue
        
        if dataset is None:
            raise Exception("无法从任何源加载 COCO 数据集")
        
        print(f"数据集包含 {len(dataset)} 个样本")
        
        # 加载真实图像和文本
        count = 0
        for i, item in enumerate(tqdm(dataset, desc="处理图像")):
            if count >= num_samples:
                break
            
            try:
                # 尝试不同的字段名
                image = None
                caption = None
                
                if "image" in item:
                    image = item["image"]
                elif "img" in item:
                    image = item["img"]
                
                if "caption" in item:
                    caption = item["caption"]
                elif "captions" in item:
                    captions = item["captions"]
                    caption = captions[0] if isinstance(captions, list) else captions
                elif "text" in item:
                    caption = item["text"]
                
                if image is None or caption is None:
                    continue
                
                # 确保图像是 PIL Image
                if not isinstance(image, Image.Image):
                    if hasattr(image, 'convert'):
                        image = image.convert("RGB")
                    else:
                        continue
                
                # 调整图像大小
                image = image.resize((256, 256), Image.Resampling.LANCZOS)
                
                # 保存图像
                image_path = images_dir / f"image_{count:06d}.jpg"
                image.save(image_path, quality=95)
                
                # 保存元数据
                metadata.append({
                    "image": f"images/image_{count:06d}.jpg",
                    "text": str(caption),
                })
                
                count += 1
                
            except Exception as e:
                print(f"处理样本 {i} 时出错: {e}")
                continue
        
        if count == 0:
            raise Exception("无法从数据集中提取任何有效样本")
        
        print(f"成功处理 {count} 个真实 COCO 样本")
        
    except Exception as e:
        print(f"\n❌ 无法下载真实的 COCO 数据集: {e}")
        print("\n" + "="*60)
        print("请使用以下方法之一获取真实数据:")
        print("="*60)
        print("\n方法1: 手动下载 COCO 数据集")
        print("  1. 访问 https://cocodataset.org/")
        print("  2. 下载 COCO 2017 训练集图像和标注")
        print("  3. 使用 prepare_custom_data() 函数处理")
        print("\n方法2: 使用其他数据源")
        print("  1. 准备图像和对应的文本描述")
        print("  2. 使用 --type custom 选项")
        print("\n方法3: 使用现有的真实数据集")
        print("  如果有其他图像-文本对数据集，可以使用")
        print("="*60)
        raise
    
    # 保存元数据
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 数据准备完成！")
    print(f"输出目录: {output_dir}")
    print(f"元数据文件: {metadata_path}")
    print(f"图像目录: {images_dir}")
    print(f"共 {len(metadata)} 个真实样本")


def prepare_custom_data(
    input_dir: str,
    output_dir: str,
    image_extensions: tuple = (".jpg", ".jpeg", ".png"),
):
    """
    准备自定义数据集
    
    Args:
        input_dir: 输入目录（包含图像和文本文件）
        output_dir: 输出目录
        image_extensions: 图像文件扩展名
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)
    
    metadata = []
    
    # 查找所有图像文件
    image_files = []
    for ext in image_extensions:
        image_files.extend(input_dir.glob(f"*{ext}"))
        image_files.extend(input_dir.glob(f"**/*{ext}"))
    
    print(f"找到 {len(image_files)} 个图像文件")
    
    for image_file in tqdm(image_files):
        # 查找对应的文本文件
        text_file = image_file.with_suffix(".txt")
        if not text_file.exists():
            # 尝试其他位置
            text_file = input_dir / f"{image_file.stem}.txt"
        
        if text_file.exists():
            with open(text_file, "r", encoding="utf-8") as f:
                text = f.read().strip()
        else:
            # 如果没有文本文件，使用文件名作为提示
            text = image_file.stem.replace("_", " ")
        
        # 复制图像
        output_image_path = images_dir / image_file.name
        image = Image.open(image_file)
        image.save(output_image_path)
        
        metadata.append({
            "image": f"images/{image_file.name}",
            "text": text,
        })
    
    # 保存元数据
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"数据准备完成！")
    print(f"输出目录: {output_dir}")
    print(f"元数据文件: {metadata_path}")
    print(f"共 {len(metadata)} 个样本")


def main():
    parser = argparse.ArgumentParser(description="准备训练数据")
    parser.add_argument(
        "--type",
        type=str,
        choices=["coco", "custom"],
        default="coco",
        help="数据集类型",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./data",
        help="输出目录",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="输入目录（自定义数据集时使用）",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5000,
        help="样本数量（COCO 子集时使用）",
    )
    args = parser.parse_args()
    
    if args.type == "coco":
        prepare_coco_subset(args.output, args.num_samples)
    elif args.type == "custom":
        if args.input is None:
            raise ValueError("自定义数据集需要指定 --input 参数")
        prepare_custom_data(args.input, args.output)


if __name__ == "__main__":
    main()

