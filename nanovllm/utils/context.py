from dataclasses import dataclass
import torch


@dataclass
class Context:
    """上下文类，用于存储和管理模型运行时的上下文信息"""
    # 是否为prefill阶段
    is_prefill: bool = False
    # 查询序列的累积长度
    cu_seqlens_q: torch.Tensor | None = None
    # 键序列的累积长度
    cu_seqlens_k: torch.Tensor | None = None
    # 查询序列的最大长度
    max_seqlen_q: int = 0
    # 键序列的最大长度
    max_seqlen_k: int = 0
    # 槽映射
    slot_mapping: torch.Tensor | None = None
    # 上下文长度
    context_lens: torch.Tensor | None = None
    # 块表
    block_tables: torch.Tensor | None = None

# 全局上下文对象
_CONTEXT = Context()

def get_context():
    """获取全局上下文对象
    
    Returns:
        全局上下文对象
    """
    return _CONTEXT

def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None, context_lens=None, block_tables=None):
    """设置全局上下文对象
    
    Args:
        is_prefill: 是否为prefill阶段
        cu_seqlens_q: 查询序列的累积长度
        cu_seqlens_k: 键序列的累积长度
        max_seqlen_q: 查询序列的最大长度
        max_seqlen_k: 键序列的最大长度
        slot_mapping: 槽映射
        context_lens: 上下文长度
        block_tables: 块表
    """
    global _CONTEXT
    _CONTEXT = Context(is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, context_lens, block_tables)

def reset_context():
    """重置全局上下文对象"""
    global _CONTEXT
    _CONTEXT = Context()
