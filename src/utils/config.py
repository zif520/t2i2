"""配置管理模块"""

import yaml
from dataclasses import dataclass
from typing import Optional, Dict, Any
from pathlib import Path


@dataclass
class Config:
    """配置数据类"""
    data: Dict[str, Any]
    model: Dict[str, Any]
    training: Dict[str, Any]
    scheduler: Dict[str, Any]
    text_encoder: Dict[str, Any]
    optimizer: Dict[str, Any]
    lr_scheduler: Dict[str, Any]
    vae: Optional[Dict[str, Any]] = None

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        """从字典创建配置对象"""
        return cls(**config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "data": self.data,
            "model": self.model,
            "training": self.training,
            "scheduler": self.scheduler,
            "text_encoder": self.text_encoder,
            "optimizer": self.optimizer,
            "lr_scheduler": self.lr_scheduler,
            "vae": self.vae,
        }


def load_config(config_path: str) -> Config:
    """
    加载 YAML 配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        Config 对象
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)
    
    return Config.from_dict(config_dict)



