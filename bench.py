import os
import time
import argparse
from random import randint, seed
from nanovllm import LLM, SamplingParams
from nanovllm.utils.monitor import monitor
# from vllm import LLM, SamplingParams


def main():
    parser = argparse.ArgumentParser(description="Benchmark script for nano-vllm")
    parser.add_argument("--model-path", type=str, default="/data/cjl/Qwen3/Qwen3-0.6B/", help="Model path")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--scheduling-policy", type=str, default="priority", choices=["fcfs", "priority"], help="Scheduling policy")
    parser.add_argument("--preemption-mode", type=str, default="last_in", choices=["aggressive", "conservative", "last_in"], help="Preemption mode")
    parser.add_argument("--max-input-len", type=int, default=1024, help="Maximum input length")
    parser.add_argument("--max-output-len", type=int, default=1024, help="Maximum output length")
    parser.add_argument("--num-seqs", type=int, default=256, help="Number of sequences")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.7, help="GPU memory utilization")
    parser.add_argument("--max-model-len", type=int, default=4096, help="Maximum model length")
    parser.add_argument("--num-kvcache-blocks", type=int, default=50, help="Number of KV cache blocks")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Log level")
    
    args = parser.parse_args()
    
    # 设置日志级别
    log_level = getattr(monitor, args.log_level)
    monitor.set_log_level(log_level)
    
    seed(0)
    num_seqs = args.num_seqs
    max_input_len = args.max_input_len
    max_ouput_len = args.max_output_len

    path = os.path.expanduser(args.model_path)
    llm = LLM(
        path, 
        enforce_eager=False, 
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        scheduling_policy=args.scheduling_policy,
        preemption_mode=args.preemption_mode,
        gpu_memory_utilization=args.gpu_memory_utilization,
        num_kvcache_blocks=args.num_kvcache_blocks
    )

    prompt_token_ids = [[randint(0, 10000) for _ in range(randint(100, max_input_len))] for _ in range(num_seqs)]
    sampling_params = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, max_ouput_len)) for _ in range(num_seqs)]
    # uncomment the following line for vllm
    # prompt_token_ids = [dict(prompt_token_ids=p) for p in prompt_token_ids]

    llm.generate(["Benchmark: "], SamplingParams())
    t = time.time()
    llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
    t = (time.time() - t)
    total_tokens = sum(sp.max_tokens for sp in sampling_params)
    throughput = total_tokens / t
    tps = num_seqs / t  # TPS: Transactions Per Second, 每秒处理的请求数
    avg_per_req_throughput = throughput / num_seqs  # 单个请求的平均吞吐量
    
    print("\n=== Benchmark Configuration ===")
    print(f"Model: {args.model_path}")
    print(f"Tensor Parallel Size: {args.tensor_parallel_size}")
    print(f"Scheduling Policy: {args.scheduling_policy}")
    print(f"Preemption Mode: {args.preemption_mode}")
    print(f"Max Input Length: {args.max_input_len}")
    print(f"Max Output Length: {args.max_output_len}")
    print(f"Number of Sequences: {args.num_seqs}")
    print(f"GPU Memory Utilization: {args.gpu_memory_utilization}")
    print(f"Max Model Length: {args.max_model_len}")
    print(f"Number of KV Cache Blocks: {args.num_kvcache_blocks}")
    print(f"Log Level: {args.log_level}")
    
    print("\n=== Benchmark Results ===")
    print(f"Total: {total_tokens}tok, Time: {t:.2f}s, Throughput: {throughput:.2f}tok/s, TPS: {tps:.2f}req/s, Avg Per Req: {avg_per_req_throughput:.2f}tok/s")


if __name__ == "__main__":
    main()
