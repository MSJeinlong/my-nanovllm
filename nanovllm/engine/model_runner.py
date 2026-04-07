import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model
from nanovllm.utils.monitor import monitor


class ModelRunner:
    """模型运行器类，负责在多进程环境中运行模型"""

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        """初始化ModelRunner对象

        Args:
            config: 配置对象
            rank: 当前进程的排名
            event: 事件对象或事件对象列表，用于进程间通信
        """
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        monitor.info(f"[ModelRunner] 初始化进程 rank={rank}, world_size={self.world_size}")
        monitor.start_timer(f"model_runner_init_rank{rank}")

        # 初始化分布式进程组
        monitor.debug(f"[ModelRunner] rank={rank} 初始化NCCL进程组")
        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)
        monitor.debug(f"[ModelRunner] rank={rank} 设置CUDA设备: {rank}")

        # 保存并设置默认数据类型和设备
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.dtype)
        torch.set_default_device("cuda")

        # 创建模型
        monitor.info(f"[ModelRunner] rank={rank} 创建模型")
        self.model = Qwen3ForCausalLM(hf_config)

        # 加载模型权重
        monitor.info(f"[ModelRunner] rank={rank} 加载模型权重: {config.model}")
        monitor.start_timer(f"load_model_rank{rank}")
        load_model(self.model, config.model)
        load_time = monitor.end_timer(f"load_model_rank{rank}")
        monitor.info(f"[ModelRunner] rank={rank} 模型加载完成，耗时: {load_time:.3f}s")

        # 创建采样器
        self.sampler = Sampler()

        # 预热模型
        monitor.info(f"[ModelRunner] rank={rank} 开始模型预热")
        self.warmup_model()
        monitor.info(f"[ModelRunner] rank={rank} 模型预热完成")

        # 分配KV缓存
        monitor.info(f"[ModelRunner] rank={rank} 分配KV缓存")
        self.allocate_kv_cache()

        # 捕获CUDA图
        if not self.enforce_eager:
            monitor.info(f"[ModelRunner] rank={rank} 捕获CUDA图")
            self.capture_cudagraph()
            monitor.info(f"[ModelRunner] rank={rank} CUDA图捕获完成")

        # 恢复默认设置
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        init_time = monitor.end_timer(f"model_runner_init_rank{rank}")
        monitor.info(f"[ModelRunner] rank={rank} 初始化完成，总耗时: {init_time:.3f}s")
        monitor.record_memory(f"model_runner_init_rank{rank}")

        # 张量并行设置
        if self.world_size > 1:
            if rank == 0:
                monitor.info(f"[ModelRunner] rank=0 创建共享内存")
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                monitor.debug(f"[ModelRunner] rank={rank} 连接到共享内存")
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

    def exit(self):
        """退出函数，用于清理资源"""
        if self.world_size > 1:
            # 关闭共享内存
            self.shm.close()
            # 等待所有进程
            dist.barrier()
            if self.rank == 0:
                # 取消链接共享内存
                self.shm.unlink()
        # 如果不强制使用eager模式，则删除图和图池
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        # 等待CUDA操作完成
        torch.cuda.synchronize()
        # 销毁进程组
        dist.destroy_process_group()

    def loop(self):
        """循环函数，用于处理来自主进程的请求"""
        while True:
            # 从共享内存读取方法名和参数
            method_name, args = self.read_shm()
            # 调用方法
            self.call(method_name, *args)
            # 如果方法名是exit，则退出循环
            if method_name == "exit":
                break

    def read_shm(self):
        """从共享内存读取数据
        
        Returns:
            方法名和参数
        """
        assert self.world_size > 1 and self.rank > 0
        # 等待事件
        self.event.wait()
        # 读取数据长度
        n = int.from_bytes(self.shm.buf[0:4], "little")
        # 读取数据并反序列化
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        # 清除事件
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        """向共享内存写入数据
        
        Args:
            method_name: 方法名
            *args: 参数
        """
        assert self.world_size > 1 and self.rank == 0
        # 序列化数据
        data = pickle.dumps([method_name, *args])
        # 获取数据长度
        n = len(data)
        # 写入数据长度
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        # 写入数据
        self.shm.buf[4:n+4] = data
        # 通知所有进程
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        """调用方法
        
        Args:
            method_name: 方法名
            *args: 参数
            
        Returns:
            方法的返回值
        """
        # 如果使用张量并行且是主进程，则向共享内存写入数据
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        # 获取方法并调用
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        """预热模型，用于初始化模型和缓存"""
        # 清空CUDA缓存
        torch.cuda.empty_cache()
        # 重置峰值内存统计
        torch.cuda.reset_peak_memory_stats()
        # 获取配置参数
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        # 计算序列数
        num_seqs = min(max_num_batched_tokens // max_model_len, self.config.max_num_seqs)
        # 创建序列
        seqs = [Sequence([0] * max_model_len) for _ in range(num_seqs)]
        # 运行模型
        self.run(seqs, True)
        # 清空CUDA缓存
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        """分配KV缓存"""

        config = self.config
        hf_config = config.hf_config

        # 计算KV缓存参数
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.dtype.itemsize

        if config.num_kvcache_blocks > 0:
            monitor.info(f"[ModelRunner] rank={self.rank} KV缓存分配 - "
                        f"blocks={config.num_kvcache_blocks}, block_size={self.block_size}, ")
        else:
            # num_kvcache_blocks=-1，根据GPU内存自动分配
            # 获取CUDA内存信息
            free, total = torch.cuda.mem_get_info()
            used = total - free
            peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
            current = torch.cuda.memory_stats()["allocated_bytes.all.current"]

            # 计算KV缓存块数
            available_memory = int(total * config.gpu_memory_utilization - used - peak + current)
            config.num_kvcache_blocks = available_memory // block_bytes

            monitor.info(f"[ModelRunner] rank={self.rank} KV缓存分配 - "
                        f"blocks={config.num_kvcache_blocks}, block_size={self.block_size}, "
                        f"num_layers={hf_config.num_hidden_layers}, num_kv_heads={num_kv_heads}, head_dim={head_dim}")
            monitor.debug(f"[ModelRunner] rank={self.rank} GPU内存 - total={total/1024**3:.2f}GB, "
                        f"available={available_memory/1024**3:.2f}GB, block_bytes={block_bytes}")

        assert config.num_kvcache_blocks > 0, "KV缓存块数必须大于0"

        # 创建KV缓存
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

        monitor.debug(f"[ModelRunner] rank={self.rank} KV缓存分配完成，共 {layer_id} 层")

    def prepare_block_tables(self, seqs: list[Sequence]):
        """准备块表
        
        Args:
            seqs: 序列列表
            
        Returns:
            块表张量
        """
        # 计算最大块表长度
        max_len = max(len(seq.block_table) for seq in seqs)
        # 填充块表
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        # 转换为张量
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        """准备prefill阶段的数据
        
        Args:
            seqs: 序列列表
            
        Returns:
            input_ids和positions张量
        """
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            seqlen = len(seq)
            # 添加输入ID
            input_ids.extend(seq[seq.num_cached_tokens:])
            # 添加位置
            positions.extend(list(range(seq.num_cached_tokens, seqlen)))
            # 计算查询序列长度
            seqlen_q = seqlen - seq.num_cached_tokens
            # 计算键序列长度
            seqlen_k = seqlen
            # 更新累积序列长度
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            # 更新最大序列长度
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            # 如果没有块表，则跳过（预热阶段）
            if not seq.block_table:    # warmup
                continue
            # 计算槽映射
            for i in range(seq.num_cached_blocks, seq.num_blocks):
                start = seq.block_table[i] * self.block_size
                if i != seq.num_blocks - 1:
                    end = start + self.block_size
                else:
                    end = start + seq.last_block_num_tokens 
                slot_mapping.extend(list(range(start, end)))
        # 如果键序列长度大于查询序列长度（前缀缓存）
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            block_tables = self.prepare_block_tables(seqs)
        # 转换为张量
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        # 设置上下文
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        """准备decode阶段的数据
        
        Args:
            seqs: 序列列表
            
        Returns:
            input_ids和positions张量
        """
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            # 添加最后一个token
            input_ids.append(seq.last_token)
            # 添加位置
            positions.append(len(seq) - 1)
            # 添加上下文长度
            context_lens.append(len(seq))
            # 计算槽映射
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
        # 转换为张量
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        # 准备块表
        block_tables = self.prepare_block_tables(seqs)
        # 设置上下文
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        """准备采样数据
        
        Args:
            seqs: 序列列表
            
        Returns:
            温度张量
        """
        temperatures = []
        for seq in seqs:
            # 添加温度参数
            temperatures.append(seq.temperature)
        # 转换为张量
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        """运行模型
        
        Args:
            input_ids: 输入ID张量
            positions: 位置张量
            is_prefill: 是否为prefill阶段
            
        Returns:
            模型的logits
        """
        # 如果是prefill阶段，或者强制使用eager模式，或者输入大小大于512
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            # 直接运行模型
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            # 获取批量大小
            bs = input_ids.size(0)
            # 获取上下文
            context = get_context()
            # 获取合适的图
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            # 获取图变量
            graph_vars = self.graph_vars
            # 设置输入ID
            graph_vars["input_ids"][:bs] = input_ids
            # 设置位置
            graph_vars["positions"][:bs] = positions
            # 填充槽映射
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            # 填充上下文长度
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            # 填充块表
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            # 重放图
            graph.replay()
            # 计算logits
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        """运行模型并返回生成的token ID

        Args:
            seqs: 序列列表
            is_prefill: 是否为prefill阶段

        Returns:
            生成的token ID列表
        """
        stage = "prefill" if is_prefill else "decode"
        total_tokens = sum(len(seq) for seq in seqs) if is_prefill else len(seqs)

        monitor.start_timer(f"run_{stage}")
        monitor.debug(f"[ModelRunner] rank={self.rank} 开始{stage} - seqs={len(seqs)}, tokens={total_tokens}")

        # 准备数据
        monitor.start_timer(f"prepare_{stage}")
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        prep_time = monitor.end_timer(f"prepare_{stage}")
        monitor.debug(f"[ModelRunner] rank={self.rank} 数据准备完成 - time={prep_time:.4f}s, input_ids_shape={input_ids.shape}")

        # 准备采样数据
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None

        # 运行模型
        monitor.start_timer(f"model_forward_{stage}")
        logits = self.run_model(input_ids, positions, is_prefill)
        forward_time = monitor.end_timer(f"model_forward_{stage}")
        monitor.debug(f"[ModelRunner] rank={self.rank} 模型前向完成 - time={forward_time:.4f}s, logits_shape={logits.shape}")

        # 采样
        if self.rank == 0:
            monitor.start_timer("sampling")
            token_ids = self.sampler(logits, temperatures).tolist()
            sample_time = monitor.end_timer("sampling")
            monitor.debug(f"[ModelRunner] rank=0 采样完成 - time={sample_time:.4f}s, num_tokens={len(token_ids)}")
        else:
            token_ids = None

        # 重置上下文
        reset_context()

        run_time = monitor.end_timer(f"run_{stage}")
        monitor.increment(f"{stage}_tokens", total_tokens)
        monitor.debug(f"[ModelRunner] rank={self.rank} {stage}完成 - total_time={run_time:.4f}s")

        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        """捕获CUDA图，用于加速推理"""
        config = self.config
        hf_config = config.hf_config
        # 计算最大批量大小
        max_bs = min(self.config.max_num_seqs, 512)
        # 计算最大块数
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        # 创建占位张量
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        # 定义批量大小列表
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        # 初始化图字典
        self.graphs = {}
        # 初始化图池
        self.graph_pool = None

        # 为每个批量大小捕获图
        for bs in reversed(self.graph_bs):
            # 创建图
            graph = torch.cuda.CUDAGraph()
            # 设置上下文
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            # 预热
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            # 捕获图
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            # 如果图池为None，则设置图池
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            # 存储图
            self.graphs[bs] = graph
            # 等待CUDA操作完成
            torch.cuda.synchronize()
            # 重置上下文
            reset_context()

        # 存储图变量
        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )



"""
## 代码优化建议
1. 错误处理 ：添加更完善的错误处理机制，如内存分配失败时的处理
2. 性能监控 ：添加性能监控和统计功能，跟踪推理速度和内存使用
3. 可配置性 ：增加更多的配置选项，如 CUDA 图的批量大小设置
4. 内存管理 ：实现更智能的内存管理策略，如动态调整 KV 缓存大小
5. 扩展性 ：支持更多的模型类型和并行策略
6. 代码可读性 ：增加更多的注释和文档，提高代码可读性
"""