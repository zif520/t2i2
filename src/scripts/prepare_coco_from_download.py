"""从已下载的COCO数据集准备训练数据"""

import argparse
import json
import sys
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def prepare_coco_from_download(
    images_dir: str,
    annotations_file: str,
    output_dir: str,
    num_samples: int = 5000,
):
    """
    从已下载的COCO数据集准备训练数据
    
    Args:
        images_dir: COCO图像目录（如 train2017/）
        annotations_file: COCO标注文件路径（captions_train2017.json）
        output_dir: 输出目录
        num_samples: 使用的样本数量
    """
    images_dir = Path(images_dir)
    annotations_file = Path(annotations_file)
    output_dir = Path(output_dir)
    
    # 检查输入
    if not images_dir.exists():
        raise ValueError(f"图像目录不存在: {images_dir}")
    if not annotations_file.exists():
        raise ValueError(f"标注文件不存在: {annotations_file}")
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    images_output_dir = output_dir / "images"
    images_output_dir.mkdir(exist_ok=True)
    
    print(f"加载COCO标注文件: {annotations_file}")
    with open(annotations_file, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)
    
    # 构建图像ID到文件名的映射
    image_id_to_filename = {}
    for img_info in coco_data.get('images', []):
        image_id_to_filename[img_info['id']] = img_info['file_name']
    
    # 构建图像ID到标注的映射（每个图像可能有多个标注，取第一个）
    image_id_to_caption = {}
    for ann in coco_data.get('annotations', []):
        image_id = ann['image_id']
        if image_id not in image_id_to_caption:
            image_id_to_caption[image_id] = ann['caption']
    
    print(f"找到 {len(image_id_to_filename)} 张图像")
    print(f"找到 {len(image_id_to_caption)} 个标注")
    
    # 处理图像
    metadata = []
    count = 0
    
    # 获取所有有标注的图像ID
    valid_image_ids = list(set(image_id_to_filename.keys()) & set(image_id_to_caption.keys()))
    print(f"有标注的图像: {len(valid_image_ids)} 张")
    
    if num_samples:
        valid_image_ids = valid_image_ids[:num_samples]
    
    print(f"\n处理 {len(valid_image_ids)} 个样本...")
    
    for image_id in tqdm(valid_image_ids, desc="处理图像"):
        try:
            # 获取文件名和标注
            filename = image_id_to_filename[image_id]
            caption = image_id_to_caption[image_id]
            
            # 源图像路径
            source_image_path = images_dir / filename
            
            if not source_image_path.exists():
                print(f"警告: 图像不存在: {source_image_path}")
                continue
            
            # 加载并处理图像
            try:
                image = Image.open(source_image_path).convert("RGB")
            except Exception as e:
                print(f"警告: 无法加载图像 {source_image_path}: {e}")
                continue
            
            # 调整图像大小到256x256
            image = image.resize((256, 256), Image.Resampling.LANCZOS)
            
            # 保存图像
            output_image_name = f"image_{count:06d}.jpg"
            output_image_path = images_output_dir / output_image_name
            image.save(output_image_path, quality=95)
            
            # 保存元数据
            metadata.append({
                "image": f"images/{output_image_name}",
                "text": caption,
            })
            
            count += 1
            
        except Exception as e:
            print(f"处理图像 {image_id} 时出错: {e}")
            continue
    
    # 保存元数据
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 数据准备完成！")
    print(f"输出目录: {output_dir}")
    print(f"元数据文件: {metadata_path}")
    print(f"图像目录: {images_output_dir}")
    print(f"共处理 {len(metadata)} 个样本")
    
    # 验证数据
    print(f"\n验证数据...")
    if len(metadata) > 0:
        sample = metadata[0]
        sample_image_path = output_dir / sample["image"]
        if sample_image_path.exists():
            img = Image.open(sample_image_path)
            print(f"✅ 样本图像: {sample_image_path.name}, 尺寸: {img.size}, 文本: {sample['text'][:50]}...")
        else:
            print(f"❌ 样本图像不存在: {sample_image_path}")


def main():
    parser = argparse.ArgumentParser(description="从已下载的COCO数据集准备训练数据")
    parser.add_argument(
        "--images_dir",
        type=str,
        required=True,
        help="COCO图像目录（如 train2017/）",
    )
    parser.add_argument(
        "--annotations_file",
        type=str,
        required=True,
        help="COCO标注文件路径（captions_train2017.json）",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="输出目录",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5000,
        help="使用的样本数量（默认: 5000）",
    )
    
    args = parser.parse_args()
    
    prepare_coco_from_download(
        images_dir=args.images_dir,
        annotations_file=args.annotations_file,
        output_dir=args.output,
        num_samples=args.num_samples,
    )


if __name__ == "__main__":
    main()

