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


class LLMEngine:
    """LLM引擎类，负责管理模型运行、调度和生成文本"""

    def __init__(self, model, **kwargs):
        """初始化LLMEngine对象
        
        Args:
            model: 模型路径或名称
            **kwargs: 配置参数
        """
        # 提取Config类中的字段
        config_fields = {field.name for field in fields(Config)}
        # 过滤出配置参数
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        # 创建配置对象
        config = Config(model, **config_kwargs)
        # 进程列表
        self.ps = []
        # 事件列表
        self.events = []
        # 获取多进程上下文
        ctx = mp.get_context("spawn")
        # 创建张量并行进程
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        # 创建主进程的模型运行器
        self.model_runner = ModelRunner(config, 0, self.events)
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        # 设置结束标记的token ID
        config.eos = self.tokenizer.eos_token_id
        # 创建调度器
        self.scheduler = Scheduler(config)
        # 注册退出函数
        atexit.register(self.exit)

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
            prompt = self.tokenizer.encode(prompt)
        # 创建序列对象
        seq = Sequence(prompt, sampling_params)
        # 添加到调度器
        self.scheduler.add(seq)

    def step(self):
        """执行一步生成
        
        Returns:
            outputs: 完成的序列ID和token ID列表
            num_tokens: 处理的token数，正数表示prefill，负数表示decode
        """
        # 调度序列
        seqs, is_prefill = self.scheduler.schedule()
        # 运行模型
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        # 后处理序列
        self.scheduler.postprocess(seqs, token_ids)
        # 收集完成的输出
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        # 计算处理的token数
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
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
        # 创建进度条
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        # 如果采样参数不是列表，则复制为与提示数量相同的列表
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        # 添加所有请求
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        # 存储输出
        outputs = {}
        # 初始化吞吐量
        prefill_throughput = decode_throughput = 0.
        # 循环直到所有请求完成
        while not self.is_finished():
            t = perf_counter()
            # 执行一步生成
            output, num_tokens = self.step()
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
        return outputs
