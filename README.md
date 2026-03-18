# My Nano-vLLM

基于官方nano-vllm实现，添加了一些优化，用于学习大模型推理引擎的实现原理。注意，本代码仅适合学习和研究，不建议在生产环境中使用。

## 功能特点
* 支持fused-moe

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

## Quick Start

See `example.py` for usage. The API mirrors vLLM's interface with minor differences in the `LLM.generate` method:
```python
from nanovllm import LLM, SamplingParams
llm = LLM("/YOUR/MODEL/PATH", enforce_eager=True, tensor_parallel_size=1)
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
prompts = ["Hello, Nano-vLLM."]
outputs = llm.generate(prompts, sampling_params)
outputs[0]["text"]
```

## Benchmark

See `bench.py` for benchmark.

**Test Configuration:**
- Hardware: RTX 4070 Laptop (8GB)
- Model: Qwen3-0.6B
- Total Requests: 256 sequences
- Input Length: Randomly sampled between 100–1024 tokens
- Output Length: Randomly sampled between 100–1024 tokens

**Performance Results:**
| Inference Engine | Output Tokens | Time (s) | Throughput (tokens/s) |
|----------------|-------------|----------|-----------------------|
| vLLM           | 133,966     | 98.37    | 1361.84               |
| Nano-vLLM      | 133,966     | 93.41    | 1434.13               |


