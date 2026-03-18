from copy import copy
from enum import Enum, auto
from itertools import count

from nanovllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    """序列状态枚举类"""
    # 等待状态
    WAITING = auto()
    # 运行状态
    RUNNING = auto()
    # 完成状态
    FINISHED = auto()


class Sequence:
    """序列类，用于表示和管理生成序列"""
    # 块大小
    block_size = 256
    # 计数器，用于生成序列ID
    counter = count()

    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
        """初始化Sequence对象
        
        Args:
            token_ids: token ID列表
            sampling_params: 采样参数
        """
        # 序列ID
        self.seq_id = next(Sequence.counter)
        # 序列状态
        self.status = SequenceStatus.WAITING
        # token ID列表
        self.token_ids = copy(token_ids)
        # 最后一个token
        self.last_token = token_ids[-1]
        # token数
        self.num_tokens = len(self.token_ids)
        # 提示词token数
        self.num_prompt_tokens = len(token_ids)
        # 缓存的token数
        self.num_cached_tokens = 0
        # 块表
        self.block_table = []
        # 温度参数
        self.temperature = sampling_params.temperature
        # 最大token数
        self.max_tokens = sampling_params.max_tokens
        # 是否忽略结束标记
        self.ignore_eos = sampling_params.ignore_eos

    def __len__(self):
        """返回序列的token数
        
        Returns:
            序列的token数
        """
        return self.num_tokens

    def __getitem__(self, key):
        """获取指定位置的token ID
        
        Args:
            key: 索引或切片
            
        Returns:
            指定位置的token ID或token ID列表
        """
        return self.token_ids[key]

    @property
    def is_finished(self):
        """检查序列是否完成
        
        Returns:
            序列是否完成
        """
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        """获取完成的token数
        
        Returns:
            完成的token数
        """
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        """获取提示词的token ID列表
        
        Returns:
            提示词的token ID列表
        """
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        """获取完成的token ID列表
        
        Returns:
            完成的token ID列表
        """
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_cached_blocks(self):
        """获取缓存的块数
        
        Returns:
            缓存的块数
        """
        return self.num_cached_tokens // self.block_size

    @property
    def num_blocks(self):
        """获取块数
        
        Returns:
            块数
        """
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        """获取最后一个块的token数
        
        Returns:
            最后一个块的token数
        """
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i):
        """获取指定块的token ID列表
        
        Args:
            i: 块索引
            
        Returns:
            指定块的token ID列表
        """
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    def append_token(self, token_id: int):
        """追加token
        
        Args:
            token_id: 要追加的token ID
        """
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    def __getstate__(self):
        """获取对象的状态，用于序列化
        
        Returns:
            对象的状态
        """
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table,
                self.token_ids if self.num_completion_tokens == 0 else self.last_token)

    def __setstate__(self, state):
        """设置对象的状态，用于反序列化
        
        Args:
            state: 对象的状态
        """
        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table = state[:-1]
        if self.num_completion_tokens == 0:
            self.token_ids = state[-1]
        else:
            self.last_token = state[-1]
