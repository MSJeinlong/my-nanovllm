from functools import lru_cache
import torch
from torch import nn


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """应用旋转位置编码
    
    Args:
        x: 输入张量
        cos: 余弦值张量
        sin: 正弦值张量
        
    Returns:
        应用旋转位置编码后的张量
    """
    # 将输入张量在最后一个维度上分成两部分
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    # 计算旋转后的结果
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    # 拼接结果并转换回原始数据类型
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


class RotaryEmbedding(nn.Module):
    """旋转位置编码模块
    
    实现了旋转位置编码（Rotary Position Embedding, RoPE），
    用于为Transformer模型中的查询和键添加位置信息。
    """

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
    ) -> None:
        """初始化RotaryEmbedding对象
        
        Args:
            head_size: 头大小
            rotary_dim: 旋转维度
            max_position_embeddings: 最大位置嵌入
            base: 基础值
        """
        super().__init__()
        self.head_size = head_size
        # 验证旋转维度是否等于头大小
        assert rotary_dim == head_size
        # 计算逆频率
        inv_freq = 1.0 / (base**(torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        # 生成位置张量
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        # 计算频率
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        # 计算余弦和正弦值
        cos = freqs.cos()
        sin = freqs.sin()
        # 创建缓存
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)
        # 注册缓冲区
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    @torch.compile
    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """前向传播
        
        Args:
            positions: 位置张量
            query: 查询张量
            key: 键张量
            
        Returns:
            应用旋转位置编码后的查询和键张量
        """
        # 获取对应位置的余弦和正弦值
        cos_sin = self.cos_sin_cache[positions]
        # 分离余弦和正弦值
        cos, sin = cos_sin.chunk(2, dim=-1)
        # 应用旋转位置编码到查询和键
        query = apply_rotary_emb(query, cos, sin)
        key = apply_rotary_emb(key, cos, sin)
        return query, key


@lru_cache(1)
def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: dict | None = None,
):
    """获取旋转位置编码实例
    
    Args:
        head_size: 头大小
        rotary_dim: 旋转维度
        max_position: 最大位置
        base: 基础值
        rope_scaling: 旋转编码缩放，目前不支持
        
    Returns:
        旋转位置编码实例
    """
    assert rope_scaling is None
    rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base)
    return rotary_emb
