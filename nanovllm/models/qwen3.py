import torch
from torch import nn
import torch.distributed as dist
from transformers import Qwen3Config

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead


class Qwen3Attention(nn.Module):
    """Qwen3注意力层
    
    实现了Qwen3模型的注意力机制，支持张量并行和KV缓存。
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        rope_theta: float = 10000,
        rope_scaling: tuple | None = None,
    ) -> None:
        """初始化Qwen3Attention对象
        
        Args:
            hidden_size: 隐藏层大小
            num_heads: 注意力头数
            num_kv_heads: KV头数
            max_position: 最大位置
            head_dim: 头维度
            rms_norm_eps: RMS归一化的epsilon值
            qkv_bias: 是否使用QKV偏置
            rope_theta: RoPE的theta值
            rope_scaling: RoPE的缩放参数
        """
        super().__init__()
        tp_size = dist.get_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5
        self.qkv_bias = qkv_bias

        # QKV投影
        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
        )
        # 输出投影
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
        )
        # 旋转位置编码
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta
        )
        # 注意力模块
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )
        # 如果不使用QKV偏置，则添加归一化层
        if not self.qkv_bias:
            self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """前向传播
        
        Args:
            positions: 位置张量
            hidden_states: 隐藏状态张量
            
        Returns:
            输出张量
        """
        # QKV投影
        qkv = self.qkv_proj(hidden_states)
        # 分割Q、K、V
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        # 重塑Q、K、V
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        # 如果不使用QKV偏置，则进行归一化
        if not self.qkv_bias:
            q = self.q_norm(q)
            k = self.k_norm(k)
        # 应用旋转位置编码
        q, k = self.rotary_emb(positions, q, k)
        # 注意力计算
        o = self.attn(q, k, v)
        # 输出投影
        output = self.o_proj(o.flatten(1, -1))
        return output


class Qwen3MLP(nn.Module):
    """Qwen3 MLP层
    
    实现了Qwen3模型的MLP层，使用Silu激活函数。
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ) -> None:
        """初始化Qwen3MLP对象
        
        Args:
            hidden_size: 隐藏层大小
            intermediate_size: 中间层大小
            hidden_act: 激活函数
        """
        super().__init__()
        # 门控和上投影
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
        )
        # 下投影
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
        )
        # 验证激活函数
        assert hidden_act == "silu"
        # 激活函数
        self.act_fn = SiluAndMul()

    def forward(self, x):
        """前向传播
        
        Args:
            x: 输入张量
            
        Returns:
            输出张量
        """
        # 门控和上投影
        gate_up = self.gate_up_proj(x)
        # 激活函数
        x = self.act_fn(gate_up)
        # 下投影
        x = self.down_proj(x)
        return x


class Qwen3DecoderLayer(nn.Module):
    """Qwen3解码器层
    
    实现了Qwen3模型的解码器层，包含注意力层和MLP层。
    """

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        """初始化Qwen3DecoderLayer对象
        
        Args:
            config: Qwen3配置对象
        """
        super().__init__()
        # 自注意力层
        self.self_attn = Qwen3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, 'attention_bias', True),
            head_dim=getattr(config, 'head_dim', None),
            rope_theta=getattr(config, "rope_theta", 1000000),
            rope_scaling=getattr(config, "rope_scaling", None),
        )
        # MLP层
        self.mlp = Qwen3MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        # 输入归一化
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # 注意力后归一化
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """前向传播
        
        Args:
            positions: 位置张量
            hidden_states: 隐藏状态张量
            residual: 残差张量
            
        Returns:
            输出张量和残差张量
        """
        # 输入归一化
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        # 自注意力计算
        hidden_states = self.self_attn(positions, hidden_states)
        # 注意力后归一化
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        # MLP计算
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen3Model(nn.Module):
    """Qwen3模型
    
    实现了Qwen3模型的主体结构，包含嵌入层、解码器层和输出归一化。
    """

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        """初始化Qwen3Model对象
        
        Args:
            config: Qwen3配置对象
        """
        super().__init__()
        # 词嵌入
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        # 解码器层列表
        self.layers = nn.ModuleList([Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        # 输出归一化
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """前向传播
        
        Args:
            input_ids: 输入ID张量
            positions: 位置张量
            
        Returns:
            输出张量
        """
        # 词嵌入
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        # 遍历解码器层
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        # 输出归一化
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen3ForCausalLM(nn.Module):
    """Qwen3因果语言模型
    
    实现了Qwen3模型的因果语言模型版本，包含模型主体和语言模型头。
    """
    # 打包模块映射
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: Qwen3Config
    ) -> None:
        """初始化Qwen3ForCausalLM对象
        
        Args:
            config: Qwen3配置对象
        """
        super().__init__()
        # 模型主体
        self.model = Qwen3Model(config)
        # 语言模型头
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        # 如果绑定词嵌入
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """前向传播
        
        Args:
            input_ids: 输入ID张量
            positions: 位置张量
            
        Returns:
            隐藏状态张量
        """
        return self.model(input_ids, positions)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """计算logits
        
        Args:
            hidden_states: 隐藏状态张量
            
        Returns:
            logits张量
        """
        return self.lm_head(hidden_states)
