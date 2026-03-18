import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.utils.context import get_context


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    """存储KV缓存的Triton内核函数
    
    Args:
        key_ptr: key张量的指针
        key_stride: key张量的步长
        value_ptr: value张量的指针
        value_stride: value张量的步长
        k_cache_ptr: key缓存的指针
        v_cache_ptr: value缓存的指针
        slot_mapping_ptr: 槽映射的指针
        D: 维度
    """
    # 获取程序ID
    idx = tl.program_id(0)
    # 加载槽映射
    slot = tl.load(slot_mapping_ptr + idx)
    # 如果槽为-1，则返回
    if slot == -1: return
    # 计算key和value的偏移量
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    # 加载key和value
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    # 计算缓存的偏移量
    cache_offsets = slot * D + tl.arange(0, D)
    # 存储到缓存
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    """存储KV缓存
    
    Args:
        key: key张量
        value: value张量
        k_cache: key缓存
        v_cache: value缓存
        slot_mapping: 槽映射
    """
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    # 验证张量的步长
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    # 调用Triton内核
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


class Attention(nn.Module):
    """注意力模块
    
    实现了带有KV缓存的注意力机制，支持prefill和decode阶段。
    """

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        """初始化Attention对象
        
        Args:
            num_heads: 注意力头数
            head_dim: 头维度
            scale: 缩放因子
            num_kv_heads: KV头数
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        # 初始化KV缓存
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        """前向传播
        
        Args:
            q: 查询张量
            k: 键张量
            v: 值张量
            
        Returns:
            输出张量
        """
        # 获取上下文
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        # 如果存在KV缓存，则存储
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        # 如果是prefill阶段
        if context.is_prefill:
            # 如果存在块表（前缀缓存）
            if context.block_tables is not None:    # prefix cache
                k, v = k_cache, v_cache
            # 使用变长注意力函数
            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        else:    # decode阶段
            # 使用KV缓存的注意力函数
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                        cache_seqlens=context.context_lens, block_table=context.block_tables, 
                                        softmax_scale=self.scale, causal=True)
        return o
