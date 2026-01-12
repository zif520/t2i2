"""快速验证项目设置"""

import sys
from pathlib import Path

def check_imports():
    """检查关键模块是否可以导入"""
    print("检查模块导入...")
    
    try:
        from src.models.dit_model import DiTModel
        print("✓ DiTModel 导入成功")
    except Exception as e:
        print(f"✗ DiTModel 导入失败: {e}")
        return False
    
    try:
        from src.models.vae_model import VAEEncoder, VAEDecoder
        print("✓ VAE 模型导入成功")
    except Exception as e:
        print(f"✗ VAE 模型导入失败: {e}")
        return False
    
    try:
        from src.data.dataset import TextImageDataset
        print("✓ 数据集类导入成功")
    except Exception as e:
        print(f"✗ 数据集类导入失败: {e}")
        return False
    
    try:
        from src.training.trainer import Trainer
        print("✓ 训练器导入成功")
    except Exception as e:
        print(f"✗ 训练器导入失败: {e}")
        return False
    
    try:
        from src.inference.generator import ImageGenerator
        print("✓ 生成器导入成功")
    except Exception as e:
        print(f"✗ 生成器导入失败: {e}")
        return False
    
    try:
        from src.utils.config import load_config
        print("✓ 配置工具导入成功")
    except Exception as e:
        print(f"✗ 配置工具导入失败: {e}")
        return False
    
    return True

def check_files():
    """检查关键文件是否存在"""
    print("\n检查文件结构...")
    
    required_files = [
        "configs/train_config.yaml",
        "configs/model_config.yaml",
        "requirements.txt",
        "setup.py",
        "README.md",
        "docs/README.md",
    ]
    
    all_exist = True
    for file_path in required_files:
        path = Path(file_path)
        if path.exists():
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} 不存在")
            all_exist = False
    
    return all_exist

def check_docs():
    """检查文档文件"""
    print("\n检查文档...")
    
    doc_files = [
        "docs/01-入门指南.md",
        "docs/02-环境配置.md",
        "docs/03-数据准备.md",
        "docs/04-模型架构.md",
        "docs/05-训练流程.md",
        "docs/06-推理使用.md",
        "docs/07-常见问题.md",
        "docs/08-进阶学习.md",
    ]
    
    all_exist = True
    for doc_file in doc_files:
        path = Path(doc_file)
        if path.exists():
            print(f"✓ {doc_file}")
        else:
            print(f"✗ {doc_file} 不存在")
            all_exist = False
    
    return all_exist

def main():
    """主函数"""
    print("=" * 50)
    print("DiT 文生图教程项目 - 设置验证")
    print("=" * 50)
    
    # 检查文件
    files_ok = check_files()
    
    # 检查文档
    docs_ok = check_docs()
    
    # 检查导入（可能会失败，因为可能没有安装依赖）
    print("\n注意: 导入检查需要先安装依赖 (pip install -r requirements.txt)")
    imports_ok = check_imports()
    
    print("\n" + "=" * 50)
    if files_ok and docs_ok:
        print("✓ 项目结构完整！")
        if imports_ok:
            print("✓ 所有模块可以正常导入！")
        else:
            print("⚠ 部分模块导入失败，请检查依赖是否安装")
        print("\n下一步:")
        print("1. 安装依赖: pip install -r requirements.txt")
        print("2. 阅读文档: docs/README.md")
        print("3. 开始训练: python src/scripts/train.py --config configs/train_config.yaml")
    else:
        print("✗ 项目结构不完整，请检查缺失的文件")
    print("=" * 50)

if __name__ == "__main__":
    main()



