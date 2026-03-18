import torch
from torch import nn


class RMSNorm(nn.Module):
    """RMS归一化模块
    
    实现了RMS（均方根）归一化，是LayerNorm的一种变体，计算效率更高。
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        """初始化RMSNorm对象
        
        Args:
            hidden_size: 隐藏层大小
            eps: 小常数，用于防止除零
        """
        super().__init__()
        self.eps = eps
        # 权重参数
        self.weight = nn.Parameter(torch.ones(hidden_size))

    @torch.compile
    def rms_forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """RMS归一化前向传播
        
        Args:
            x: 输入张量
            
        Returns:
            归一化后的张量
        """
        # 保存原始数据类型
        orig_dtype = x.dtype
        # 转换为float类型进行计算
        x = x.float()
        # 计算方差
        var = x.pow(2).mean(dim=-1, keepdim=True)
        # 计算RMS归一化
        x.mul_(torch.rsqrt(var + self.eps))
        # 转换回原始数据类型并应用权重
        x = x.to(orig_dtype).mul_(self.weight)
        return x

    @torch.compile
    def add_rms_forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """带残差连接的RMS归一化前向传播
        
        Args:
            x: 输入张量
            residual: 残差张量
            
        Returns:
            归一化后的张量和残差张量
        """
        # 保存原始数据类型
        orig_dtype = x.dtype
        # 转换为float类型并添加残差
        x = x.float().add_(residual.float())
        # 保存残差
        residual = x.to(orig_dtype)
        # 计算方差
        var = x.pow(2).mean(dim=-1, keepdim=True)
        # 计算RMS归一化
        x.mul_(torch.rsqrt(var + self.eps))
        # 转换回原始数据类型并应用权重
        x = x.to(orig_dtype).mul_(self.weight)
        return x, residual

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """前向传播
        
        Args:
            x: 输入张量
            residual: 残差张量，可选
            
        Returns:
            归一化后的张量，或归一化后的张量和残差张量
        """
        if residual is None:
            # 无残差连接
            return self.rms_forward(x)
        else:
            # 带残差连接
            return self.add_rms_forward(x, residual)
