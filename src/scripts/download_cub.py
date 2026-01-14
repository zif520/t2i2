"""下载并准备 CUB-200-2011 数据集"""

import argparse
import json
import sys
import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import requests
import zipfile
import shutil

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def download_file(url: str, output_path: Path, chunk_size: int = 8192, timeout: int = 600):
    """下载文件并显示进度"""
    # 设置更长的超时时间
    response = requests.get(url, stream=True, timeout=timeout)
    response.raise_for_status()  # 检查HTTP错误
    
    total_size = int(response.headers.get('content-length', 0))
    
    # 检查Content-Type，确保不是HTML错误页面
    content_type = response.headers.get('content-type', '')
    if 'text/html' in content_type:
        raise ValueError(f"下载失败: 服务器返回HTML页面而不是文件。可能是链接失效或需要认证。")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    downloaded = 0
    with open(output_path, 'wb') as f, tqdm(
        desc=f"下载 {output_path.name}",
        total=total_size if total_size > 0 else None,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                bar.update(len(chunk))
    
    # 验证文件大小（CUB数据集应该大于100MB）
    file_size = output_path.stat().st_size
    if file_size < 100 * 1024 * 1024:  # 小于100MB
        raise ValueError(f"下载的文件太小 ({file_size / 1024 / 1024:.2f} MB)，可能下载失败。预期约1.1GB。")
    
    print(f"\n✓ 下载完成，文件大小: {file_size / 1024 / 1024:.2f} MB")


def download_cub_dataset_from_hf(output_dir: str = "./data/cub_raw"):
    """从 Hugging Face 下载 CUB 数据集"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=== 从 Hugging Face 下载 CUB 数据集 ===\n")
    
    # 尝试不同的数据集名称（优先使用 cassiekang/cub200_dataset）
    cub_sources = [
        ("cassiekang/cub200_dataset", "train"),  # 优先使用这个
        ("caltech/cub200", "train"),
        ("cub200", "train"),
    ]
    
    for source, split in cub_sources:
        try:
            from datasets import load_dataset
            import os
            # 设置更长的超时时间
            os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")
            os.environ.setdefault("HF_ENDPOINT", os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"))
            
            print(f"尝试从 {source} 下载...")
            print(f"超时设置: {os.environ.get('HF_HUB_DOWNLOAD_TIMEOUT')} 秒")
            print(f"镜像: {os.environ.get('HF_ENDPOINT')}")
            dataset = load_dataset(source, split=split)
            print(f"✓ 成功加载，包含 {len(dataset)} 个样本")
            
            # 直接处理并保存为项目格式（不保存原始 CUB 格式）
            # 直接输出到 cub_subset 格式
            output_subset_dir = Path("./data/cub_subset")
            output_subset_dir.mkdir(parents=True, exist_ok=True)
            images_dir = output_subset_dir / "images"
            images_dir.mkdir(exist_ok=True)
            
            metadata = []
            
            print("处理图像和文本描述...")
            for idx, item in enumerate(tqdm(dataset, desc="处理数据")):
                image = item["image"]
                
                # 获取文本描述（cassiekang/cub200_dataset 有 text 字段）
                if "text" in item:
                    text = item["text"]
                else:
                    # 如果没有文本，尝试从 label 获取
                    label = item.get("label", idx % 200)
                    if hasattr(dataset, "features") and "label" in dataset.features:
                        if hasattr(dataset.features["label"], "names"):
                            label_names = dataset.features["label"].names
                            class_name = label_names[label] if label < len(label_names) else f"class_{label}"
                        else:
                            class_name = f"class_{label}"
                    else:
                        class_name = f"class_{label}"
                    text = f"a photo of a {class_name}"
                
                # 调整图像大小到 256x256
                image = image.convert("RGB")
                image = image.resize((256, 256), Image.Resampling.LANCZOS)
                
                # 保存图像
                image_filename = f"image_{idx:06d}.jpg"
                image_path = images_dir / image_filename
                image.save(image_path, quality=95)
                
                # 保存元数据
                metadata.append({
                    "image": f"images/{image_filename}",
                    "text": text,
                })
            
            # 保存元数据
            metadata_file = output_subset_dir / "metadata.json"
            import json
            with open(metadata_file, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            print(f"\n✅ 数据已处理并保存到: {output_subset_dir}")
            print(f"   - 图像: {images_dir}")
            print(f"   - 元数据: {metadata_file}")
            print(f"   - 样本数: {len(metadata)}")
            
            return output_subset_dir
            
        except Exception as e:
            print(f"  ❌ 失败: {e}")
            continue
    
    print("\n❌ 从 Hugging Face 下载失败，尝试使用官方链接下载...")
    return download_cub_dataset_official(output_dir)


def download_cub_dataset_official(output_dir: str = "./data/cub_raw"):
    """从官方链接下载 CUB-200-2011 数据集"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 尝试多个可能的下载链接
    cub_urls = [
        "https://data.caltech.edu/records/65de6-4bqg6/files/CUB_200_2011.tgz",
        "http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz",
        "https://www.vision.caltech.edu/datasets/cub_200_2011/CUB_200_2011.tgz",
    ]
    
    tgz_path = output_dir / "CUB_200_2011.tgz"
    
    # 检查是否已下载
    if tgz_path.exists() and tgz_path.stat().st_size > 100 * 1024 * 1024:
        print(f"✓ 文件已存在: {tgz_path}")
        return tgz_path
    
    print("=== 从官方链接下载 CUB-200-2011 数据集 ===\n")
    print(f"输出目录: {output_dir}\n")
    
    for cub_url in cub_urls:
        try:
            print(f"尝试链接: {cub_url}")
            # 先检查链接是否可用
            response = requests.head(cub_url, timeout=10, allow_redirects=True)
            if response.status_code == 200:
                print(f"✓ 链接可用，开始下载（文件约 1.1GB）...")
                download_file(cub_url, tgz_path, timeout=600)
                print(f"\n✓ 下载完成: {tgz_path}")
                return tgz_path
            else:
                print(f"  ❌ 状态码: {response.status_code}")
        except Exception as e:
            print(f"  ❌ 失败: {e}")
            continue
    
    # 所有链接都失败
    print("\n❌ 所有下载链接都失败")
    print("\n请手动下载 CUB-200-2011 数据集:")
    print("  1. 访问: https://www.vision.caltech.edu/datasets/cub_200_2011/")
    print("  2. 或访问: https://data.caltech.edu/records/65de6-4bqg6")
    print(f"  3. 下载到: {tgz_path}")
    print("  4. 然后运行: python src/scripts/download_cub.py --skip_download --cub_dir ./data/cub_raw/CUB_200_2011")
    raise ValueError("无法下载 CUB 数据集，请手动下载")


def download_cub_dataset(output_dir: str = "./data/cub_raw"):
    """下载 CUB-200-2011 数据集（优先使用 torchvision，然后 Hugging Face，最后官方链接）"""
    output_dir = Path(output_dir)
    
    # 方法1: 尝试使用 torchvision
    try:
        print("=== 方法1: 使用 torchvision 下载 ===\n")
        import torchvision.datasets as datasets
        cub_dir = output_dir / "CUB_200_2011"
        
        print("正在下载 CUB-200-2011 数据集（使用 torchvision）...")
        dataset = datasets.CUB200(
            root=str(output_dir),
            train=True,
            download=True,
            transform=None
        )
        print(f"✅ torchvision 下载成功，包含 {len(dataset)} 个样本")
        
        # torchvision 下载的数据结构可能不同，需要适配
        # 通常下载到 output_dir/cub200/ 或类似目录
        possible_dirs = [
            output_dir / "cub200",
            output_dir / "CUB_200_2011",
            output_dir / "CUB200",
        ]
        
        for dir_path in possible_dirs:
            if dir_path.exists():
                print(f"✓ 找到数据集目录: {dir_path}")
                return dir_path
        
        # 如果找不到，返回包含数据的目录
        return output_dir
        
    except Exception as e:
        print(f"❌ torchvision 下载失败: {e}")
        print("尝试其他方法...\n")
    
    # 方法2: 尝试 Hugging Face
    try:
        return download_cub_dataset_from_hf(output_dir)
    except Exception as e:
        print(f"从 Hugging Face 下载失败: {e}")
        print("尝试官方链接...\n")
    
    # 方法3: 官方链接
    try:
        tgz_path = download_cub_dataset_official(output_dir)
        return tgz_path
    except Exception as e:
        print(f"官方链接下载失败: {e}")
        raise


def extract_cub_dataset(tgz_path: Path, output_dir: Path):
    """解压 CUB 数据集"""
    print(f"\n=== 解压数据集 ===\n")
    print(f"源文件: {tgz_path}")
    print(f"输出目录: {output_dir}")
    
    extract_dir = output_dir / "CUB_200_2011"
    
    if extract_dir.exists():
        print(f"✓ 已解压到: {extract_dir}")
        return extract_dir
    
    print("正在解压...")
    try:
        import tarfile
        # 尝试不同的压缩格式
        for mode in ['r:gz', 'r', 'r:*']:
            try:
                with tarfile.open(tgz_path, mode) as tar:
                    tar.extractall(output_dir)
                print(f"✓ 解压完成: {extract_dir}")
                return extract_dir
            except Exception:
                continue
        raise Exception("无法解压文件，可能文件损坏或格式不正确")
    except Exception as e:
        print(f"❌ 解压失败: {e}")
        print("\n请尝试手动解压:")
        print(f"  tar -xzf {tgz_path} -C {output_dir}")
        raise


def prepare_cub_dataset(
    cub_dir: Path,
    output_dir: str = "./data/cub_subset",
    num_samples: int = None,
):
    """准备 CUB 数据集为训练格式"""
    cub_dir = Path(cub_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)
    
    print(f"\n=== 准备 CUB 数据集 ===\n")
    print(f"源目录: {cub_dir}")
    print(f"输出目录: {output_dir}\n")
    
    # 读取图像列表和标注
    images_file = cub_dir / "images.txt"
    image_class_labels_file = cub_dir / "image_class_labels.txt"
    classes_file = cub_dir / "classes.txt"
    images_path = cub_dir / "images"
    
    if not images_file.exists():
        raise ValueError(f"找不到 images.txt: {images_file}")
    
    # 读取类别信息
    class_id_to_name = {}
    if classes_file.exists():
        with open(classes_file, 'r') as f:
            for line in f:
                class_id, class_name = line.strip().split(' ', 1)
                class_id_to_name[int(class_id)] = class_name.replace('_', ' ')
    
    # 读取图像列表
    image_id_to_path = {}
    with open(images_file, 'r') as f:
        for line in f:
            image_id, image_path = line.strip().split(' ', 1)
            image_id_to_path[int(image_id)] = image_path
    
    # 读取类别标签
    image_id_to_class = {}
    if image_class_labels_file.exists():
        with open(image_class_labels_file, 'r') as f:
            for line in f:
                image_id, class_id = line.strip().split(' ', 1)
                image_id_to_class[int(image_id)] = int(class_id)
    
    # 处理图像
    metadata = []
    count = 0
    
    print("处理图像...")
    for image_id, image_path in tqdm(sorted(image_id_to_path.items()), desc="处理图像"):
        if num_samples and count >= num_samples:
            break
        
        source_image_path = images_path / image_path
        
        if not source_image_path.exists():
            continue
        
        try:
            # 加载图像
            image = Image.open(source_image_path).convert("RGB")
            
            # 调整大小到 256x256
            image = image.resize((256, 256), Image.Resampling.LANCZOS)
            
            # 生成文本描述
            # 使用类别名称作为基础描述
            class_id = image_id_to_class.get(image_id, None)
            if class_id and class_id in class_id_to_name:
                bird_name = class_id_to_name[class_id]
                # 生成多个描述变体
                captions = [
                    f"a photo of a {bird_name}",
                    f"a {bird_name} bird",
                    f"a picture of a {bird_name}",
                    f"an image of a {bird_name}",
                ]
                caption = captions[count % len(captions)]  # 轮换使用不同描述
            else:
                caption = "a bird"
            
            # 保存图像
            output_image_name = f"image_{count:06d}.jpg"
            output_image_path = images_dir / output_image_name
            image.save(output_image_path, quality=95)
            
            # 保存元数据
            metadata.append({
                "image": f"images/{output_image_name}",
                "text": caption,
            })
            
            count += 1
            
        except Exception as e:
            print(f"处理图像 {image_path} 时出错: {e}")
            continue
    
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
    parser = argparse.ArgumentParser(description="下载并准备 CUB-200-2011 数据集")
    parser.add_argument(
        "--download",
        action="store_true",
        help="下载数据集",
    )
    parser.add_argument(
        "--cub_dir",
        type=str,
        default="./data/cub_raw/CUB_200_2011",
        help="CUB 数据集目录（如果已下载）",
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
        "--skip_download",
        action="store_true",
        help="跳过下载，直接处理已有数据",
    )
    
    args = parser.parse_args()
    
    # 下载数据集
    if args.download and not args.skip_download:
        result = download_cub_dataset("./data/cub_raw")
        # 如果返回的是目录（从 HF 下载），直接使用
        if isinstance(result, Path) and result.is_dir():
            cub_dir = result
        else:
            # 如果返回的是压缩文件，需要解压
            tgz_path = result
            cub_dir = extract_cub_dataset(tgz_path, Path("./data/cub_raw"))
        args.cub_dir = str(cub_dir)
    
    # 准备数据集
    cub_dir = Path(args.cub_dir)
    if not cub_dir.exists():
        print(f"❌ CUB 数据集目录不存在: {cub_dir}")
        print("请先下载数据集: python src/scripts/download_cub.py --download")
        return
    
    prepare_cub_dataset(
        cub_dir=cub_dir,
        output_dir=args.output,
        num_samples=args.num_samples,
    )


if __name__ == "__main__":
    main()

