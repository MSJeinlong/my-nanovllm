import torch
from typing import List, Tuple, Dict, Optional
import numpy as np

from nanovllm.utils.monitor import monitor


class CacheEngine:
    """缓存引擎，管理GPU和CPU上的KV Cache物理存储
    
    负责分配、管理和传输KV Cache张量，是CPU swap机制的核心组件。
    """
    
    def __init__(
        self,
        num_gpu_blocks: int,
        num_cpu_blocks: int,
        block_size: int,
        num_layers: int,
        num_heads: int,
        head_size: int,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda"
    ):
        """初始化CacheEngine
        
        Args:
            num_gpu_blocks: GPU上的块数量
            num_cpu_blocks: CPU上的块数量（用于swap）
            block_size: 每个块的大小（token数量）
            num_layers: 模型层数
            num_heads: 注意力头数
            head_size: 每个头的大小
            dtype: 数据类型
            device: GPU设备
        """
        self.block_size = block_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_size = head_size
        self.dtype = dtype
        self.device = device

        if num_cpu_blocks <= 0:
            num_cpu_blocks = int(num_gpu_blocks / 2)
        
        monitor.info(f"初始化CacheEngine，GPU块数: {num_gpu_blocks}, CPU块数: {num_cpu_blocks}, 块大小: {block_size}, 层数: {num_layers}")
        
        # 分配GPU缓存 [num_layers, 2 (K/V), num_gpu_blocks, num_heads, head_size, block_size]
        self.gpu_cache = self._allocate_kv_cache(num_gpu_blocks, device)
        # 分配CPU缓存 [num_layers, 2 (K/V), num_cpu_blocks, num_heads, head_size, block_size]
        self.cpu_cache = self._allocate_kv_cache(num_cpu_blocks, "cpu")
        
        self.num_gpu_blocks = num_gpu_blocks
        self.num_cpu_blocks = num_cpu_blocks
        
        monitor.info("CacheEngine初始化完成")
        
    def _allocate_kv_cache(self, num_blocks: int, device: str) -> List[torch.Tensor]:
        """分配KV缓存张量
        
        Args:
            num_blocks: 块数量
            device: 设备（cuda或cpu）
            
        Returns:
            KV缓存列表，每层包含Key和Value两个张量
            形状: [num_blocks, num_heads, head_size, block_size]
        """
        # 确保num_blocks为正数
        num_blocks = max(1, num_blocks)
        
        monitor.debug(f"在 {device} 上分配KV缓存，块数: {num_blocks}, 层数: {self.num_layers}, num_heads: {self.num_heads}, head_size: {self.head_size}")
        
        kv_cache = []
        for i in range(self.num_layers):
            # Key缓存
            key_cache = torch.zeros(
                (num_blocks, self.num_heads, self.head_size, self.block_size),
                dtype=self.dtype,
                device=device
            )
            # Value缓存  
            value_cache = torch.zeros(
                (num_blocks, self.num_heads, self.head_size, self.block_size),
                dtype=self.dtype,
                device=device
            )
            kv_cache.append((key_cache, value_cache))
            
            # 打印第一层的缓存信息
            if i == 0:
                monitor.info(f"{device} KV缓存形状: {key_cache.shape}")
                monitor.info(f"{device} KV缓存步长: {key_cache.stride()}")
                D = self.num_heads * self.head_size
                monitor.info(f"{device} KV缓存D值: {D}")
                monitor.info(f"{device} KV缓存stride(1): {key_cache.stride(1)}")
        
        monitor.debug(f"{device} KV缓存分配完成")
        return kv_cache
    
    def swap_blocks(
        self, 
        src_cache: List[Tuple[torch.Tensor, torch.Tensor]], 
        dst_cache: List[Tuple[torch.Tensor, torch.Tensor]], 
        src_to_dst: torch.Tensor
    ):
        """执行块交换（核心swap操作）
        
        Args:
            src_cache: 源缓存（GPU或CPU）
            dst_cache: 目标缓存（GPU或CPU）
            src_to_dst: 映射张量 [num_mappings, 2]，每行[src_block_id, dst_block_id]
        """
        if src_to_dst.numel() == 0:
            return
        
        src_device = src_cache[0][0].device
        dst_device = dst_cache[0][0].device
        num_blocks = src_to_dst.size(0)
        
        monitor.debug(f"执行块交换，从 {src_device} 到 {dst_device}，块数: {num_blocks}")
            
        # 对每一层执行swap
        for layer_idx in range(self.num_layers):
            src_key, src_value = src_cache[layer_idx]
            dst_key, dst_value = dst_cache[layer_idx]
            
            # 使用index_copy或scatter进行块拷贝
            # src_to_dst[:, 0] 是源块ID, src_to_dst[:, 1] 是目标块ID
            src_indices = src_to_dst[:, 0].long()
            dst_indices = src_to_dst[:, 1].long()
            
            # 拷贝Key缓存
            dst_key[dst_indices] = src_key[src_indices].to(dst_key.device)
            # 拷贝Value缓存
            dst_value[dst_indices] = src_value[src_indices].to(dst_value.device)
        
        monitor.debug(f"块交换完成，从 {src_device} 到 {dst_device}")
    
    def swap_out(self, src_to_dst: torch.Tensor):
        """从GPU swap到CPU
        
        Args:
            src_to_dst: 映射张量，GPU块ID -> CPU块ID
        """
        if src_to_dst.numel() > 0:
            monitor.info(f"执行swap out操作，GPU块数: {src_to_dst.size(0)}")
        self.swap_blocks(self.gpu_cache, self.cpu_cache, src_to_dst)
    
    def swap_in(self, src_to_dst: torch.Tensor):
        """从CPU swap到GPU
        
        Args:
            src_to_dst: 映射张量，CPU块ID -> GPU块ID
        """
        if src_to_dst.numel() > 0:
            monitor.info(f"执行swap in操作，CPU块数: {src_to_dst.size(0)}")
        self.swap_blocks(self.cpu_cache, self.gpu_cache, src_to_dst)
    
    def copy_blocks(
        self, 
        kv_cache: List[Tuple[torch.Tensor, torch.Tensor]], 
        block_mapping: torch.Tensor
    ):
        """在同一个设备内拷贝块（用于fork/decode时的块复制）
        
        Args:
            kv_cache: 缓存（GPU或CPU）
            block_mapping: 映射张量 [num_pairs, 2]，[src_block, dst_block]
        """
        if block_mapping.numel() == 0:
            return
            
        for layer_idx in range(self.num_layers):
            key_cache, value_cache = kv_cache[layer_idx]
            src_indices = block_mapping[:, 0].long()
            dst_indices = block_mapping[:, 1].long()
            
            key_cache[dst_indices] = key_cache[src_indices]
            value_cache[dst_indices] = value_cache[src_indices]
    
    def get_cache_block(self, block_id: int, layer_idx: int, is_key: bool, device: str = "cuda"):
        """获取指定块的缓存视图（用于attention计算）
        
        Args:
            block_id: 块ID
            layer_idx: 层索引
            is_key: 是否获取Key（否则获取Value）
            device: 设备类型
            
        Returns:
            缓存张量视图
        """
        cache = self.gpu_cache if device == "cuda" else self.cpu_cache
        cache_tensor = cache[layer_idx][0 if is_key else 1]
        return cache_tensor[block_id]
    
    def zero_cache(self, block_ids: List[int], device: str = "cuda"):
        """清零指定块的缓存
        
        Args:
            block_ids: 块ID列表
            device: 设备类型
        """
        if not block_ids:
            return
            
        cache = self.gpu_cache if device == "cuda" else self.cpu_cache
        indices = torch.tensor(block_ids, dtype=torch.long, device=self.device if device == "cuda" else "cpu")
        
        for layer_idx in range(self.num_layers):
            key_cache, value_cache = cache[layer_idx]
            key_cache[indices] = 0
            value_cache[indices] = 0