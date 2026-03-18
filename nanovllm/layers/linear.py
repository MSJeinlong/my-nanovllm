import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist


def divide(numerator, denominator):
    """安全除法，确保分子能被分母整除
    
    Args:
        numerator: 分子
        denominator: 分母
        
    Returns:
        商
    """
    assert numerator % denominator == 0
    return numerator // denominator


class LinearBase(nn.Module):
    """线性层基类
    
    为不同类型的线性层提供基础功能。
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        tp_dim: int | None = None,
    ):
        """初始化LinearBase对象
        
        Args:
            input_size: 输入大小
            output_size: 输出大小
            bias: 是否使用偏置
            tp_dim: 张量并行维度
        """
        super().__init__()
        self.tp_dim = tp_dim
        # 获取当前进程的排名
        self.tp_rank = dist.get_rank()
        # 获取进程组大小
        self.tp_size = dist.get_world_size()
        # 权重参数
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        # 设置权重加载器
        self.weight.weight_loader = self.weight_loader
        # 如果使用偏置
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            x: 输入张量
            
        Returns:
            输出张量
        """
        raise NotImplementedError


class ReplicatedLinear(LinearBase):
    """复制线性层
    
    在所有进程中复制相同的权重。
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        """初始化ReplicatedLinear对象
        
        Args:
            input_size: 输入大小
            output_size: 输出大小
            bias: 是否使用偏置
        """
        super().__init__(input_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """权重加载器
        
        Args:
            param: 参数
            loaded_weight: 加载的权重
        """
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            x: 输入张量
            
        Returns:
            输出张量
        """
        return F.linear(x, self.weight, self.bias)


class ColumnParallelLinear(LinearBase):
    """列并行线性层
    
    将输出维度在多个进程间分配。
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        """初始化ColumnParallelLinear对象
        
        Args:
            input_size: 输入大小
            output_size: 输出大小
            bias: 是否使用偏置
        """
        tp_size = dist.get_world_size()
        super().__init__(input_size, divide(output_size, tp_size), bias, 0)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """权重加载器
        
        Args:
            param: 参数
            loaded_weight: 加载的权重
        """
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        # 截取当前分区的权重
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            x: 输入张量
            
        Returns:
            输出张量
        """
        return F.linear(x, self.weight, self.bias)


class MergedColumnParallelLinear(ColumnParallelLinear):
    """合并列并行线性层
    
    合并多个输出大小的列并行线性层。
    """

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = False,
    ):
        """初始化MergedColumnParallelLinear对象
        
        Args:
            input_size: 输入大小
            output_sizes: 输出大小列表
            bias: 是否使用偏置
        """
        self.output_sizes = output_sizes
        super().__init__(input_size, sum(output_sizes), bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int):
        """权重加载器
        
        Args:
            param: 参数
            loaded_weight: 加载的权重
            loaded_shard_id: 加载的分片ID
        """
        param_data = param.data
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


class QKVParallelLinear(ColumnParallelLinear):
    """QKV并行线性层
    
    为注意力机制的Q、K、V投影提供并行线性层。
    """

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = False,
    ):
        """初始化QKVParallelLinear对象
        
        Args:
            hidden_size: 隐藏层大小
            head_size: 头大小
            total_num_heads: 总头数
            total_num_kv_heads: 总KV头数
            bias: 是否使用偏置
        """
        tp_size = dist.get_world_size()
        total_num_kv_heads = total_num_kv_heads or total_num_heads
        self.head_size = head_size
        self.num_heads = divide(total_num_heads, tp_size)
        self.num_kv_heads = divide(total_num_kv_heads, tp_size)
        output_size = (total_num_heads + 2 * total_num_kv_heads) * self.head_size
        super().__init__(hidden_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str):
        """权重加载器
        
        Args:
            param: 参数
            loaded_weight: 加载的权重
            loaded_shard_id: 加载的分片ID，可选值为"q"、"k"、"v"
        """
        param_data = param.data
        assert loaded_shard_id in ["q", "k", "v"]
        if loaded_shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
        elif loaded_shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
        else:
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


class RowParallelLinear(LinearBase):
    """行并行线性层
    
    将输入维度在多个进程间分配。
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        """初始化RowParallelLinear对象
        
        Args:
            input_size: 输入大小
            output_size: 输出大小
            bias: 是否使用偏置
        """
        tp_size = dist.get_world_size()
        super().__init__(divide(input_size, tp_size), output_size, bias, 1)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """权重加载器
        
        Args:
            param: 参数
            loaded_weight: 加载的权重
        """
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        # 截取当前分区的权重
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            x: 输入张量
            
        Returns:
            输出张量
        """
        # 只有主进程使用偏置
        y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
        # 如果使用张量并行，所有进程的结果求和
        if self.tp_size > 1:
            dist.all_reduce(y)
        return y
