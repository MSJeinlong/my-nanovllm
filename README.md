# My Nano-vLLM

基于官方nano-vllm实现，添加了一些优化，用于学习大模型推理引擎的实现原理。注意，本代码仅适合学习和研究，不建议在生产环境中使用。

## 功能特点

- 支持fused-moe

## 安装

```bash
pip install git+https://github.com/MSJeinlong/my-nanovllm.git
```

## 模型下载

建议从魔塔下载模型权重，因为模型权重较大，直接从huggingface下载会比较慢。

```bash
pip install modelscope
modelscope download --model Qwen/Qwen3.5-27B --local_dir ./dir
```

## 快速开始

使用方法请参考 `example.py`。API 接口与 vLLM 类似，在 `LLM.generate` 方法上有一些小差异：

```python
from nanovllm import LLM, SamplingParams
llm = LLM("/YOUR/MODEL/PATH", enforce_eager=True, tensor_parallel_size=1)
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
prompts = ["Hello, Nano-vLLM."]
outputs = llm.generate(prompts, sampling_params)
outputs[0]["text"]
```

## 性能测试

性能测试请参考 `bench.py`。

**测试配置：**

- 硬件：A100 (40GB)
- 模型：Qwen3-0.6B，Qwen3-8B
- 总请求数：256 个序列
- 输入长度：100–1024 个 token 之间随机采样
- 输出长度：100–1024 个 token 之间随机采样

**性能结果：**

| 推理引擎      | 模型         | 输出 Tokens | 时间 (秒) | 吞吐量 (tokens/秒) | 吞吐量提升  |
| --------- | :--------- | --------- | ------ | -------------- | :----- |
| vLLM      | Qwen3-0.6B | 133,966   | 19.87  | 6742.50        | 0.0    |
| Nano-vLLM | Qwen3-0.6B | 133,966   | 19.72  | 6791.98        | +0.73% |
| vLLM      | Qwen3-8B   | 133,966   | 82.78  | 1618.33        | 0.0    |
| Nano-vLLM | Qwen3-8B   | 133,966   | 79.14  | 1692.73        | 4.60%  |

