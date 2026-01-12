"""DiT (Diffusion Transformer) 模型实现"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class TimestepEmbedder(nn.Module):
    """时间步嵌入模块"""
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.SiLU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.hidden_size = hidden_size
    
    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        生成时间步嵌入
        
        Args:
            timesteps: 形状为 (B,) 的时间步张量
            
        Returns:
            时间步嵌入，形状为 (B, hidden_size)
        """
        # 使用正弦位置编码
        half_dim = self.hidden_size // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        
        if self.hidden_size % 2 == 1:
            emb = F.pad(emb, (0, 1))
        
        return self.mlp(emb)


class LabelEmbedder(nn.Module):
    """标签嵌入模块（用于条件信息）"""
    
    def __init__(self, num_classes: int, hidden_size: int):
        super().__init__()
        self.embedding_table = nn.Embedding(num_classes, hidden_size)
        self.num_classes = num_classes
    
    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        """
        生成标签嵌入
        
        Args:
            labels: 形状为 (B,) 的标签张量
            
        Returns:
            标签嵌入，形状为 (B, hidden_size)
        """
        return self.embedding_table(labels)


class Attention(nn.Module):
    """多头自注意力模块"""
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        head_dim: Optional[int] = None,
        qkv_bias: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = head_dim or dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        自注意力前向传播
        
        Args:
            x: 输入张量，形状为 (B, N, C)
            
        Returns:
            输出张量，形状为 (B, N, C)
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class DiTBlock(nn.Module):
    """DiT Transformer 块"""
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads)
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6)
        
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, hidden_size),
            nn.Dropout(dropout),
        )
    
    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入特征，形状为 (B, N, C)
            c: 条件嵌入（时间步+文本），形状为 (B, C)
            
        Returns:
            输出特征，形状为 (B, N, C)
        """
        # 自注意力 + 条件注入
        x = x + self.attn(self.norm1(x) + c.unsqueeze(1))
        # MLP
        x = x + self.mlp(self.norm2(x) + c.unsqueeze(1))
        return x


class FinalLayer(nn.Module):
    """最终输出层"""
    
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size),
        )
    
    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入特征，形状为 (B, N, C)
            c: 条件嵌入，形状为 (B, C)
            
        Returns:
            输出，形状为 (B, N, patch_size*patch_size*out_channels)
        """
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = self.norm_final(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        x = self.linear(x)
        return x


class DiTModel(nn.Module):
    """DiT (Diffusion Transformer) 模型"""
    
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
        input_size: int = 32,  # 潜在空间尺寸（256/8 = 32）
    ):
        """
        初始化 DiT 模型
        
        Args:
            hidden_size: 隐藏层维度
            num_layers: Transformer 层数
            num_heads: 注意力头数
            patch_size: Patch 大小
            in_channels: 输入通道数（VAE 潜在空间）
            out_channels: 输出通道数
            attention_head_dim: 注意力头维度
            mlp_ratio: MLP 扩展比例
            dropout: Dropout 率
            input_size: 输入特征图尺寸（潜在空间）
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.num_patches = (input_size // patch_size) ** 2
        
        # Patch 嵌入
        self.x_embedder = nn.Conv2d(
            in_channels,
            hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True,
        )
        
        # 位置嵌入
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, hidden_size)
        )
        
        # 时间步嵌入
        self.t_embedder = TimestepEmbedder(hidden_size)
        
        # 文本条件嵌入（通过线性层投影）
        # CLIP 输出维度通常是 512 (base) 或 768 (large)
        # 默认使用 512，如果实际维度不同会在第一次前向传播时调整
        self.text_embed_dim = 512  # 默认 CLIP base 的维度
        self.y_embedder = nn.Linear(self.text_embed_dim, hidden_size)
        
        # Transformer 块
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        
        # 最终层
        self.final_layer = FinalLayer(hidden_size, patch_size, out_channels)
        
        # 初始化权重
        self.initialize_weights()
    
    def initialize_weights(self):
        """初始化模型权重"""
        # 位置嵌入
        nn.init.normal_(self.pos_embed, std=0.02)
        
        # Patch 嵌入
        nn.init.xavier_uniform_(self.x_embedder.weight)
        nn.init.constant_(self.x_embedder.bias, 0)
        
        # 最终层
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)
    
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入潜在表示，形状为 (B, in_channels, H, W)
            t: 时间步，形状为 (B,)
            y: 文本条件嵌入，形状为 (B, 768)
            
        Returns:
            预测的噪声，形状为 (B, out_channels, H, W)
        """
        # Patch 嵌入
        x = self.x_embedder(x)  # (B, hidden_size, H//patch_size, W//patch_size)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, N, hidden_size)
        
        # 添加位置嵌入
        x = x + self.pos_embed
        
        # 时间步嵌入
        t = self.t_embedder(t)  # (B, hidden_size)
        
        # 文本条件嵌入
        # 如果 y 是 3D (B, seq_len, dim)，需要池化
        if y.dim() == 3:
            # (B, seq_len, dim) -> (B, dim) 平均池化
            y = y.mean(dim=1)
        elif y.dim() == 2:
            # 已经是 (B, dim) 格式
            pass
        else:
            raise ValueError(f"不支持的文本嵌入维度: {y.dim()}")
        
        # 检查维度是否匹配，如果不匹配则重新创建 y_embedder（应该很少见）
        actual_dim = y.shape[-1]
        if actual_dim != self.text_embed_dim:
            # 维度不匹配，需要重新创建
            self.text_embed_dim = actual_dim
            self.y_embedder = nn.Linear(actual_dim, self.hidden_size).to(y.device)
        
        # 确保张量是连续的（避免编译模式下的问题）
        y = y.contiguous()
        y = self.y_embedder(y)  # (B, hidden_size)
        
        # 合并条件
        c = t + y  # (B, hidden_size)
        
        # Transformer 块
        for block in self.blocks:
            x = block(x, c)
        
        # 最终层
        x = self.final_layer(x, c)  # (B, N, patch_size*patch_size*out_channels)
        
        # 重塑为图像格式
        x = x.reshape(B, H, W, self.patch_size, self.patch_size, -1)
        x = x.permute(0, 5, 1, 3, 2, 4)  # (B, out_channels, H, patch_size, W, patch_size)
        x = x.reshape(B, -1, H * self.patch_size, W * self.patch_size)
        
        return x

