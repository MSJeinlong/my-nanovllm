import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.utils.monitor import monitor


class LLMEngine:
    """LLM引擎类，负责管理模型运行、调度和生成文本"""

    def __init__(self, model, **kwargs):
        """初始化LLMEngine对象

        Args:
            model: 模型路径或名称
            **kwargs: 配置参数
        """
        monitor.info(f"[LLMEngine] 开始初始化引擎，模型: {model}")
        monitor.start_timer("engine_init")

        # 提取Config类中的字段
        config_fields = {field.name for field in fields(Config)}
        # 过滤出配置参数
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        # 创建配置对象
        config = Config(model, **config_kwargs)
        monitor.info(f"[LLMEngine] 配置加载完成 - tensor_parallel_size={config.tensor_parallel_size}, max_num_seqs={config.max_num_seqs}")

        # 进程列表
        self.ps = []
        # 事件列表
        self.events = []
        # 获取多进程上下文
        ctx = mp.get_context("spawn")

        # 创建张量并行进程
        monitor.info(f"[LLMEngine] 启动 {config.tensor_parallel_size} 个张量并行进程")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
            monitor.debug(f"[LLMEngine] 进程 {i} 已启动")

        # 创建主进程的模型运行器
        monitor.info("[LLMEngine] 初始化主进程模型运行器")
        self.model_runner = ModelRunner(config, 0, self.events)

        # 加载分词器
        monitor.info(f"[LLMEngine] 加载分词器: {config.model}")
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        monitor.info(f"[LLMEngine] 分词器加载完成，EOS token ID: {config.eos}")

        # 创建调度器
        monitor.info("[LLMEngine] 创建调度器")
        self.scheduler = Scheduler(config)

        # 注册退出函数
        atexit.register(self.exit)

        init_time = monitor.end_timer("engine_init")
        monitor.info(f"[LLMEngine] 引擎初始化完成，耗时: {init_time:.3f}s")
        monitor.record_memory("engine_init")

    def exit(self):
        """退出函数，用于清理资源"""
        # 调用模型运行器的退出方法
        self.model_runner.call("exit")
        # 删除模型运行器
        del self.model_runner
        # 等待所有进程结束
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        """添加生成请求

        Args:
            prompt: 提示文本或token ID列表
            sampling_params: 采样参数
        """
        # 如果提示是字符串，则进行编码
        if isinstance(prompt, str):
            prompt_tokens = self.tokenizer.encode(prompt)
            monitor.debug(f"[LLMEngine] 文本编码完成，token数: {len(prompt_tokens)}")
        else:
            prompt_tokens = prompt

        # 创建序列对象
        seq = Sequence(prompt_tokens, sampling_params)
        monitor.info(f"[LLMEngine] 添加请求 - seq_id={seq.seq_id}, prompt_tokens={len(prompt_tokens)}, max_tokens={sampling_params.max_tokens}")

        # 添加到调度器
        self.scheduler.add(seq)
        monitor.increment("total_requests")

    def step(self):
        """执行一步生成

        Returns:
            outputs: 完成的序列ID和token ID列表
            num_tokens: 处理的token数，正数表示prefill，负数表示decode
        """
        monitor.start_timer("step")

        # 调度序列
        seqs, is_prefill = self.scheduler.schedule()
        stage = "prefill" if is_prefill else "decode"
        monitor.debug(f"[LLMEngine] Step开始 - stage={stage}, num_seqs={len(seqs)}")

        # 运行模型
        monitor.start_timer(f"model_run_{stage}")
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        model_time = monitor.end_timer(f"model_run_{stage}")
        monitor.debug(f"[LLMEngine] 模型推理完成 - stage={stage}, time={model_time:.4f}s")

        # 后处理序列
        self.scheduler.postprocess(seqs, token_ids)

        # 收集完成的输出
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        if outputs:
            monitor.info(f"[LLMEngine] 序列完成 - seq_ids={[seq_id for seq_id, _ in outputs]}")

        # 计算处理的token数
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)

        step_time = monitor.end_timer("step")
        monitor.increment(f"{stage}_steps")
        monitor.debug(f"[LLMEngine] Step完成 - stage={stage}, num_tokens={abs(num_tokens)}, time={step_time:.4f}s")

        return outputs, num_tokens

    def is_finished(self):
        """检查是否所有请求都已完成
        
        Returns:
            是否所有请求都已完成
        """
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        """生成文本

        Args:
            prompts: 提示文本列表或token ID列表的列表
            sampling_params: 采样参数或采样参数列表
            use_tqdm: 是否使用进度条

        Returns:
            生成的文本和token ID列表
        """
        monitor.info(f"[LLMEngine] 开始生成，请求数: {len(prompts)}")
        monitor.start_timer("generate")

        # 创建进度条
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)

        # 如果采样参数不是列表，则复制为与提示数量相同的列表
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)

        # 添加所有请求
        monitor.info("[LLMEngine] 添加所有生成请求")
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)

        # 存储输出
        outputs = {}
        # 初始化吞吐量
        prefill_throughput = decode_throughput = 0.
        step_count = 0

        # 循环直到所有请求完成
        monitor.info("[LLMEngine] 开始推理循环")
        while not self.is_finished():
            t = perf_counter()
            # 执行一步生成
            output, num_tokens = self.step()
            step_count += 1

            # 更新进度条
            if use_tqdm:
                if num_tokens > 0:
                    prefill_throughput = num_tokens / (perf_counter() - t)
                else:
                    decode_throughput = -num_tokens / (perf_counter() - t)
                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode": f"{int(decode_throughput)}tok/s",
                })

            # 收集输出
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1)

        # 按序列ID排序
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        # 解码token ID为文本
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]

        # 关闭进度条
        if use_tqdm:
            pbar.close()

        generate_time = monitor.end_timer("generate")
        monitor.info(f"[LLMEngine] 生成完成 - steps={step_count}, time={generate_time:.3f}s, throughput={len(prompts)/generate_time:.2f}req/s")
        monitor.record_memory("generate_end")

        return outputs

"""
## 代码优化建议
1. 错误处理 ：添加更完善的错误处理机制，如模型加载失败、生成过程中的异常等
2. 配置验证 ：在初始化时验证配置参数的有效性，避免运行时错误
3. 性能优化 ：
   - 优化批处理策略，提高吞吐量
   - 减少进程间通信开销
   - 优化内存使用
4. 可扩展性 ：
   - 支持更多的模型类型
   - 支持更多的采样策略
   - 支持分布式推理
"""