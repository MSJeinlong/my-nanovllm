import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from nanovllm.utils.context import get_context


class VocabParallelEmbedding(nn.Module):
    """词汇并行嵌入层
    
    支持张量并行的嵌入层，将词汇表均匀分布到多个设备上。
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ):
        """初始化VocabParallelEmbedding对象
        
        Args:
            num_embeddings: 词汇表大小
            embedding_dim: 嵌入维度
        """
        super().__init__()
        # 获取当前进程的排名
        self.tp_rank = dist.get_rank()
        # 获取进程组大小
        self.tp_size = dist.get_world_size()
        # 验证词汇表大小是否能被进程组大小整除
        assert num_embeddings % self.tp_size == 0
        # 词汇表大小
        self.num_embeddings = num_embeddings
        # 每个分区的词汇表大小
        self.num_embeddings_per_partition = self.num_embeddings // self.tp_size
        # 当前分区的词汇表起始索引
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank
        # 当前分区的词汇表结束索引
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition
        # 权重参数
        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim))
        # 设置权重加载器
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """权重加载器
        
        Args:
            param: 参数
            loaded_weight: 加载的权重
        """
        param_data = param.data
        shard_size = param_data.size(0)
        start_idx = self.tp_rank * shard_size
        # 截取当前分区的权重
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        # 复制到参数
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor):
        """前向传播
        
        Args:
            x: 输入张量
            
        Returns:
            输出张量
        """
        if self.tp_size > 1:
            # 创建掩码，只保留当前分区的词汇
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
            # 调整输入索引
            x = mask * (x - self.vocab_start_idx)
        # 进行嵌入
        y = F.embedding(x, self.weight)
        if self.tp_size > 1:
            # 应用掩码
            y = mask.unsqueeze(1) * y
            # 所有进程的结果求和
            dist.all_reduce(y)
        return y


class ParallelLMHead(VocabParallelEmbedding):
    """并行语言模型头
    
    支持张量并行的语言模型头，用于生成词汇表上的概率分布。
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
    ):
        """初始化ParallelLMHead对象
        
        Args:
            num_embeddings: 词汇表大小
            embedding_dim: 嵌入维度
            bias: 是否使用偏置，目前不支持
        """
        assert not bias
        super().__init__(num_embeddings, embedding_dim)

    def forward(self, x: torch.Tensor):
        """前向传播
        
        Args:
            x: 输入张量
            
        Returns:
            输出logits
        """
        # 获取上下文
        context = get_context()
        # 如果是prefill阶段，只取每个序列的最后一个token
        if context.is_prefill:
            last_indices = context.cu_seqlens_q[1:] - 1
            x = x[last_indices].contiguous()
        # 计算logits
        logits = F.linear(x, self.weight)
        # 如果使用张量并行
        if self.tp_size > 1:
            # 收集所有进程的logits
            all_logits = [torch.empty_like(logits) for _ in range(self.tp_size)] if self.tp_rank == 0 else None
            dist.gather(logits, all_logits, 0)
            # 在主进程中拼接logits
            logits = torch.cat(all_logits, -1) if self.tp_rank == 0 else None
        return logits
