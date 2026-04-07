import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass
class Config:
    """模型配置类，用于存储和管理模型的各种配置参数"""
    # 模型路径或名称
    model: str
    # 批处理的最大token数
    max_num_batched_tokens: int = 16384
    # 最大序列数
    max_num_seqs: int = 512
    # 模型的最大长度
    max_model_len: int = 4096
    # GPU内存利用率
    gpu_memory_utilization: float = 0.9
    # 张量并行大小
    tensor_parallel_size: int = 1
    # 是否强制使用eager模式
    enforce_eager: bool = False
    # Hugging Face配置对象
    hf_config: AutoConfig | None = None
    # 结束标记的token ID
    eos: int = -1
    # KV缓存的块大小
    kvcache_block_size: int = 256
    # KV缓存的块数，-1表示自动计算
    num_kvcache_blocks: int = -1
    # 调度策略
    scheduling_policy: str = "fcfs"  # fcfs或priority
    # 抢占模式
    preemption_mode: str = "aggressive"  # aggressive、conservative或last_in
    # Prefill阶段的最大token数
    max_prefill_tokens: int = 4096
    # Decode阶段的最小序列数
    min_decode_seqs: int = 1

    def __post_init__(self):
        """初始化后验证和设置配置"""
        # 验证模型路径是否存在
        assert os.path.isdir(self.model)
        # 验证KV缓存块大小是否为16的倍数
        assert self.kvcache_block_size % 256 == 0
        # 验证张量并行大小是否在有效范围内
        assert 1 <= self.tensor_parallel_size <= 8
        # 从预训练模型加载Hugging Face配置
        self.hf_config = AutoConfig.from_pretrained(self.model)
        # 确保最大模型长度不超过Hugging Face配置中的最大位置嵌入
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        # 验证批处理的最大token数是否不小于最大模型长度
        assert self.max_num_batched_tokens >= self.max_model_len
        # 验证调度策略
        assert self.scheduling_policy in ["fcfs", "priority"], f"无效的调度策略: {self.scheduling_policy}"
        # 验证抢占模式
        assert self.preemption_mode in ["aggressive", "conservative", "last_in"], f"无效的抢占模式: {self.preemption_mode}"
        # 验证Prefill最大token数
        assert self.max_prefill_tokens > 0, "max_prefill_tokens必须大于0"
        # 验证Decode最小序列数
        assert self.min_decode_seqs > 0, "min_decode_seqs必须大于0"
