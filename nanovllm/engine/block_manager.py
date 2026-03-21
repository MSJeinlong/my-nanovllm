from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


class Block:
    """缓存块类，用于存储token序列和相关信息
    
    每个Block对象代表一个固定大小的缓存块，用于存储token序列。
    它包含块的唯一标识符、引用计数、哈希值和token ID列表。
    """

    def __init__(self, block_id):
        """初始化Block对象
        
        Args:
            block_id: 块的唯一标识符
        """
        # 块的唯一标识符
        self.block_id = block_id
        # 引用计数，用于跟踪有多少序列使用此块
        # 当引用计数为0时，块可以被释放和重用
        self.ref_count = 0
        # 块的哈希值，用于快速查找相同内容的块
        self.hash = -1
        # 存储的token ID列表
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        """更新块的哈希值和token ID列表
        
        Args:
            hash: 新的哈希值
            token_ids: 新的token ID列表
        """
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        """重置块的状态，将引用计数设为1，清空哈希值和token ID列表
        
        当块被重新分配时调用此方法。
        """
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:
    """块管理器类，用于管理和分配缓存块
    
    负责缓存块的分配、释放和管理，支持通过哈希值快速查找和重用相同内容的块。
    核心功能包括：
    1. 块的分配和释放
    2. 块的哈希计算和查找
    3. 块表（block_table）的管理
    4. 支持序列的token追加操作
    """

    def __init__(self, num_blocks: int, block_size: int):
        """初始化BlockManager对象
        
        Args:
            num_blocks: 总块数
            block_size: 每个块的大小（token数量）
        """
        # 每个块的大小
        self.block_size = block_size
        # 块列表，存储所有的Block对象
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        # 哈希值到块ID的映射，用于快速查找相同内容的块
        self.hash_to_block_id: dict[int, int] = dict()
        # 空闲块ID队列，用于快速获取可用块
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        # 已使用块ID集合，用于快速判断块是否被使用
        self.used_block_ids: set[int] = set()

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        """计算token ID列表的哈希值
        
        哈希值用于唯一标识一个token序列，支持通过前缀哈希值构建链式哈希。
        
        Args:
            token_ids: token ID列表
            prefix: 前缀哈希值，默认为-1
            
        Returns:
            计算得到的哈希值
        """
        h = xxhash.xxh64()
        # 如果提供了前缀哈希值，则将其作为哈希的一部分
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        # 将token ID列表转换为字节并更新哈希
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self, block_id: int) -> Block:
        """分配一个块
        
        从空闲块列表中移除块，将其添加到已使用块集合，并重置块的状态。
        
        Args:
            block_id: 要分配的块ID
            
        Returns:
            分配的Block对象
        """
        block = self.blocks[block_id]
        # 确保块当前未被使用
        assert block.ref_count == 0
        # 重置块的状态
        block.reset()
        # 从空闲块列表中移除
        self.free_block_ids.remove(block_id)
        # 添加到已使用块集合
        self.used_block_ids.add(block_id)
        return self.blocks[block_id]

    def _deallocate_block(self, block_id: int):
        """释放一个块
        
        将块从已使用块集合中移除，添加到空闲块列表。
        
        Args:
            block_id: 要释放的块ID
        """
        # 确保块的引用计数为0
        assert self.blocks[block_id].ref_count == 0
        # 从已使用块集合中移除
        self.used_block_ids.remove(block_id)
        # 添加到空闲块列表
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> bool:
        """检查是否有足够的空闲块来分配给序列
        
        Args:
            seq: 要分配块的序列
            
        Returns:
            是否可以分配
        """
        return len(self.free_block_ids) >= seq.num_blocks

    def allocate(self, seq: Sequence):
        """为序列分配块
        
        为序列分配块并构建块表（block_table），支持缓存命中和缓存未命中两种情况。
        
        Args:
            seq: 要分配块的序列
        """
        # 确保序列当前没有块表
        assert not seq.block_table
        # 哈希值，用于链式哈希计算
        h = -1
        # 缓存未命中标志
        cache_miss = False
        
        # 遍历序列的每个块
        for i in range(seq.num_blocks):
            # 获取当前块的token ID列表
            token_ids = seq.block(i)
            # 计算哈希值（如果是完整块）
            h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1
            # 尝试通过哈希值查找块
            block_id = self.hash_to_block_id.get(h, -1)
            
            # 检查是否缓存未命中
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                cache_miss = True
            
            # 处理缓存未命中
            if cache_miss:
                # 获取第一个空闲块
                block_id = self.free_block_ids[0]
                # 分配块
                block = self._allocate_block(block_id)
            else:
                # 缓存命中，增加缓存token计数
                seq.num_cached_tokens += self.block_size
                # 如果块已被使用，增加引用计数
                if block_id in self.used_block_ids:
                    block = self.blocks[block_id]
                    block.ref_count += 1
                else:
                    # 块未被使用，分配块
                    block = self._allocate_block(block_id)
            
            # 更新块的哈希值和token ID列表（如果是完整块）
            if h != -1:
                block.update(h, token_ids)
                # 更新哈希到块ID的映射
                self.hash_to_block_id[h] = block_id
            
            # 将块ID添加到序列的块表
            seq.block_table.append(block_id)

    def deallocate(self, seq: Sequence):
        """释放序列使用的块
        
        遍历序列的块表，减少每个块的引用计数，当引用计数为0时释放块。
        
        Args:
            seq: 要释放块的序列
        """
        # 反向遍历块表，确保正确处理引用计数
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            # 减少引用计数
            block.ref_count -= 1
            # 如果引用计数为0，释放块
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        # 重置序列的缓存token计数
        seq.num_cached_tokens = 0
        # 清空序列的块表
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        """检查是否可以为序列追加一个token。
        为什么需要追加一个token？原因如下：
        - 模型接收输入序列（提示词）
        - 每次生成一个新token
        - 将新token追加到输入序列末尾
        - 用更新后的序列作为下一次生成的输入
        - 重复此过程直到生成结束符或达到最大长度
        - 因此，序列长度会逐个token递增。

        当序列长度模块大小等于1时，需要分配新块。
        
        Args:
            seq: 要追加token的序列
            
        Returns:
            是否可以追加
        """
        # 当需要分配新块时（len(seq) % self.block_size == 1），需要至少1个空闲块
        # 其他情况不需要新块，直接返回True
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        """处理序列的token追加
        
        根据序列长度与块大小的关系，处理token追加操作：
        1. 当长度模块大小为1时，分配新块
        2. 当长度模块大小为0时，更新最后一个块的哈希值
        
        Args:
            seq: 要追加token的序列
        """
        block_table = seq.block_table
        # 获取最后一个块
        last_block = self.blocks[block_table[-1]]
        
        # 当序列长度模块大小为1时，需要分配新块
        if len(seq) % self.block_size == 1:
            # 确保最后一个块已经有哈希值
            assert last_block.hash != -1
            # 获取第一个空闲块
            block_id = self.free_block_ids[0]
            # 分配块
            self._allocate_block(block_id)
            # 将块ID添加到块表
            block_table.append(block_id)
        # 当序列长度模块大小为0时，更新最后一个块的哈希值
        elif len(seq) % self.block_size == 0:
            # 确保最后一个块还没有哈希值
            assert last_block.hash == -1
            # 获取最后一个块的token ID列表
            token_ids = seq.block(seq.num_blocks-1)
            # 获取前一个块的哈希值作为前缀
            prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
            # 计算哈希值
            h = self.compute_hash(token_ids, prefix)
            # 更新块的哈希值和token ID列表
            last_block.update(h, token_ids)
            # 更新哈希到块ID的映射
            self.hash_to_block_id[h] = last_block.block_id
        else:
            # 其他情况，确保最后一个块还没有哈希值
            assert last_block.hash == -1


"""
后续可优化的地方：
1. 错误处理 ：添加更完善的错误处理机制，如块分配失败时的处理
2. 内存使用监控 ：添加内存使用统计和监控功能
3. 并行优化 ：考虑使用多线程或异步操作提高并发性能
4. 缓存策略优化 ：实现更智能的缓存淘汰策略，如 LRU 机制
5. 内存对齐 ：优化内存分配，提高缓存命中率
"""