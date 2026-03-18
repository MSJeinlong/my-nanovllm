from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:
    """调度器类，负责调度序列的执行"""

    def __init__(self, config: Config):
        """初始化Scheduler对象
        
        Args:
            config: 配置对象
        """
        # 最大序列数
        self.max_num_seqs = config.max_num_seqs
        # 批处理的最大token数
        self.max_num_batched_tokens = config.max_num_batched_tokens
        # 结束标记的token ID
        self.eos = config.eos
        # 块管理器
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        # 等待队列
        self.waiting: deque[Sequence] = deque()
        # 运行队列
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        """检查是否所有请求都已完成
        
        Returns:
            是否所有请求都已完成
        """
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        """添加序列到等待队列
        
        Args:
            seq: 要添加的序列
        """
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        """调度序列
        
        Returns:
            调度的序列列表和是否为prefill阶段
        """
        # prefill阶段
        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0
        # 从等待队列中取出序列进行调度
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]
            # 检查是否超过最大token数或无法分配块
            if num_batched_tokens + len(seq) > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq):
                break
            num_seqs += 1
            # 分配块
            self.block_manager.allocate(seq)
            # 更新批处理token数
            num_batched_tokens += len(seq) - seq.num_cached_tokens
            # 设置序列状态为运行中
            seq.status = SequenceStatus.RUNNING
            # 从等待队列中移除
            self.waiting.popleft()
            # 添加到运行队列
            self.running.append(seq)
            # 添加到调度列表
            scheduled_seqs.append(seq)
        # 如果有调度的序列，则返回
        if scheduled_seqs:
            return scheduled_seqs, True

        # decode阶段
        while self.running and num_seqs < self.max_num_seqs:
            # 从运行队列中取出序列
            seq = self.running.popleft()
            # 检查是否可以追加token
            while not self.block_manager.can_append(seq):
                # 如果运行队列不为空，则抢占最后一个序列
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    # 否则抢占当前序列
                    self.preempt(seq)
                    break
            else:
                # 如果可以追加token，则调度该序列
                num_seqs += 1
                # 处理追加
                self.block_manager.may_append(seq)
                # 添加到调度列表
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        # 将调度的序列重新添加到运行队列
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        """抢占序列
        
        Args:
            seq: 要抢占的序列
        """
        # 设置序列状态为等待
        seq.status = SequenceStatus.WAITING
        # 释放块
        self.block_manager.deallocate(seq)
        # 添加到等待队列的开头
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[bool]:
        """后处理序列
        
        Args:
            seqs: 序列列表
            token_ids: 生成的token ID列表
        """
        for seq, token_id in zip(seqs, token_ids):
            # 追加token
            seq.append_token(token_id)
            # 检查是否完成
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                # 设置序列状态为完成
                seq.status = SequenceStatus.FINISHED
                # 释放块
                self.block_manager.deallocate(seq)
                # 从运行队列中移除
                self.running.remove(seq)
