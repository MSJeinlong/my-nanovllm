import torch
from torch import nn
import torch.nn.functional as F


class SiluAndMul(nn.Module):
    """Silu激活函数和乘法的组合模块
    
    这个模块将输入张量在最后一个维度上分成两部分，
    对第一部分应用Silu激活函数，然后与第二部分相乘。
    """

    def __init__(self):
        """初始化SiluAndMul对象"""
        super().__init__()

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            x: 输入张量
            
        Returns:
            输出张量
        """
        # 在最后一个维度上分成两部分
        x, y = x.chunk(2, -1)
        # 对第一部分应用Silu激活函数，然后与第二部分相乘
        return F.silu(x) * y
