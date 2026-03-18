import torch
from torch import nn


class Sampler(nn.Module):
    """采样器模块
    
    实现了基于温度的采样方法，用于从模型的输出logits中采样token。
    """

    def __init__(self):
        """初始化Sampler对象"""
        super().__init__()

    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        """前向传播
        
        Args:
            logits: 模型的输出logits
            temperatures: 温度参数列表
            
        Returns:
            采样的token ID
        """
        # 应用温度缩放
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))
        # 计算概率分布
        probs = torch.softmax(logits, dim=-1)
        # 使用Gumbel-Softmax采样方法
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        return sample_tokens
