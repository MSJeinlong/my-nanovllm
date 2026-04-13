import os
import time
import argparse
import threading
from random import randint, seed
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np


@dataclass
class RequestMetrics:
    """单个请求的性能指标"""
    request_id: int
    start_time: float
    end_time: float
    latency: float
    input_tokens: int
    output_tokens: int
    throughput: float


@dataclass
class BenchmarkStats:
    """基准测试统计结果"""
    total_requests: int
    successful_requests: int
    failed_requests: int
    total_time: float
    total_input_tokens: int
    total_output_tokens: int
    
    avg_latency: float
    min_latency: float
    max_latency: float
    p50_latency: float
    p90_latency: float
    p95_latency: float
    p99_latency: float
    
    overall_throughput: float
    avg_tps: float
    avg_per_req_throughput: float


class ConcurrentBenchmark:
    """并发基准测试类"""
    
    def __init__(self, args):
        self.args = args
        self.llm = None
        self.SamplingParams = None
        self.metrics: List[RequestMetrics] = []
        self.metrics_lock = threading.Lock()
        
    def init_llm(self):
        """初始化LLM引擎"""
        path = os.path.expanduser(self.args.model_path)
        
        if self.args.backend == "nanovllm":
            from nanovllm import LLM, SamplingParams
            from nanovllm.utils.monitor import monitor
            
            self.SamplingParams = SamplingParams
            
            log_level = getattr(monitor, self.args.log_level)
            monitor.set_log_level(log_level)
            
            self.llm = LLM(
                path, 
                enforce_eager=False, 
                max_model_len=self.args.max_model_len,
                tensor_parallel_size=self.args.tensor_parallel_size,
                scheduling_policy=self.args.scheduling_policy,
                preemption_mode=self.args.preemption_mode,
                gpu_memory_utilization=self.args.gpu_memory_utilization
            )
        else:
            from vllm import LLM, SamplingParams
            
            self.SamplingParams = SamplingParams
            
            self.llm = LLM(
                path, 
                enforce_eager=False, 
                max_model_len=self.args.max_model_len,
                tensor_parallel_size=self.args.tensor_parallel_size,
                gpu_memory_utilization=self.args.gpu_memory_utilization
            )
        
    def generate_request_data(self, num_requests: int) -> Tuple[List, List]:
        """生成请求数据"""
        prompt_token_ids = [
            [randint(0, 10000) for _ in range(randint(100, self.args.max_input_len))] 
            for _ in range(num_requests)
        ]
        sampling_params = [
            self.SamplingParams(
                temperature=0.7,
                ignore_eos=True, 
                max_tokens=randint(100, self.args.max_output_len)
            ) 
            for _ in range(num_requests)
        ]
        return prompt_token_ids, sampling_params
    
    def run_concurrent_test(self) -> BenchmarkStats:
        """运行并发测试
        
        模拟多个客户端同时发送请求的场景，使用批量处理方式
        """
        print(f"\n{'='*60}")
        print(f"开始并发测试 (Backend: {self.args.backend}, Concurrency Level: {self.args.concurrency})")
        print(f"{'='*60}")
        
        seed(0)
        num_requests = self.args.num_requests
        
        prompt_token_ids, sampling_params = self.generate_request_data(num_requests)
        
        self.llm.generate(["Warmup: "], self.SamplingParams())
        
        all_metrics = []
        
        if self.args.concurrency == 1:
            metrics = self._run_single_threaded(prompt_token_ids, sampling_params)
            all_metrics.extend(metrics)
        else:
            metrics = self._run_batch_concurrent(prompt_token_ids, sampling_params)
            all_metrics.extend(metrics)
        
        return self._calculate_stats(all_metrics)
    
    def _run_single_threaded(self, prompt_token_ids: List, sampling_params: List) -> List[RequestMetrics]:
        """单线程顺序执行"""
        metrics = []
        total = len(prompt_token_ids)
        
        for i, (prompt, sp) in enumerate(zip(prompt_token_ids, sampling_params)):
            req_start = time.time()
            self.llm.generate([prompt], [sp], use_tqdm=False)
            req_end = time.time()
            
            metric = RequestMetrics(
                request_id=i,
                start_time=req_start,
                end_time=req_end,
                latency=req_end - req_start,
                input_tokens=len(prompt),
                output_tokens=sp.max_tokens,
                throughput=sp.max_tokens / (req_end - req_start) if (req_end - req_start) > 0 else 0
            )
            metrics.append(metric)
            
            if (i + 1) % 10 == 0:
                print(f"Progress: {i+1}/{total} requests completed")
        
        return metrics
    
    def _run_batch_concurrent(self, prompt_token_ids: List, sampling_params: List) -> List[RequestMetrics]:
        """批量并发执行
        
        将请求分成多个批次，模拟并发场景
        """
        metrics = []
        total = len(prompt_token_ids)
        batch_size = self.args.concurrency
        num_batches = (total + batch_size - 1) // batch_size
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, total)
            
            batch_prompts = prompt_token_ids[start_idx:end_idx]
            batch_params = sampling_params[start_idx:end_idx]
            
            batch_start = time.time()
            self.llm.generate(batch_prompts, batch_params, use_tqdm=False)
            batch_end = time.time()
            
            batch_latency = batch_end - batch_start
            avg_latency_per_req = batch_latency / len(batch_prompts)
            
            for i, (prompt, sp) in enumerate(zip(batch_prompts, batch_params)):
                metric = RequestMetrics(
                    request_id=start_idx + i,
                    start_time=batch_start,
                    end_time=batch_end,
                    latency=avg_latency_per_req,
                    input_tokens=len(prompt),
                    output_tokens=sp.max_tokens,
                    throughput=sp.max_tokens / avg_latency_per_req if avg_latency_per_req > 0 else 0
                )
                metrics.append(metric)
            
            print(f"Batch {batch_idx + 1}/{num_batches}: {len(batch_prompts)} requests, "
                  f"latency: {batch_latency:.2f}s")
        
        return metrics
    
    def run_stress_test(self) -> BenchmarkStats:
        """运行压力测试
        
        持续发送请求，测试系统在长时间运行下的稳定性
        """
        print(f"\n{'='*60}")
        print(f"开始压力测试 (Backend: {self.args.backend}, Duration: {self.args.duration}s, Rate: {self.args.request_rate} req/s)")
        print(f"{'='*60}")
        
        seed(0)
        
        self.llm.generate(["Warmup: "], self.SamplingParams())
        
        all_metrics = []
        request_id = 0
        start_time = time.time()
        last_report_time = start_time
        
        request_interval = 1.0 / self.args.request_rate if self.args.request_rate > 0 else 0
        
        while True:
            current_time = time.time()
            elapsed = current_time - start_time
            
            if elapsed >= self.args.duration:
                break
            
            prompt_token_ids, sampling_params = self.generate_request_data(1)
            prompt = prompt_token_ids[0]
            sp = sampling_params[0]
            
            req_start = time.time()
            self.llm.generate([prompt], [sp], use_tqdm=False)
            req_end = time.time()
            
            metric = RequestMetrics(
                request_id=request_id,
                start_time=req_start,
                end_time=req_end,
                latency=req_end - req_start,
                input_tokens=len(prompt),
                output_tokens=sp.max_tokens,
                throughput=sp.max_tokens / (req_end - req_start) if (req_end - req_start) > 0 else 0
            )
            all_metrics.append(metric)
            request_id += 1
            
            if current_time - last_report_time >= 5.0:
                recent_metrics = [m for m in all_metrics if m.start_time >= last_report_time]
                if recent_metrics:
                    avg_latency = np.mean([m.latency for m in recent_metrics])
                    print(f"Time: {elapsed:.1f}s, Requests: {len(all_metrics)}, "
                          f"Recent Avg Latency: {avg_latency:.2f}s")
                last_report_time = current_time
            
            if request_interval > 0:
                sleep_time = request_interval - (time.time() - req_start)
                if sleep_time > 0:
                    time.sleep(sleep_time)
        
        return self._calculate_stats(all_metrics)
    
    def run_sustained_load_test(self) -> BenchmarkStats:
        """运行持续负载测试
        
        保持固定的并发级别，持续发送请求
        """
        print(f"\n{'='*60}")
        print(f"开始持续负载测试 (Backend: {self.args.backend}, Concurrency: {self.args.concurrency}, Duration: {self.args.duration}s)")
        print(f"{'='*60}")
        
        seed(0)
        
        self.llm.generate(["Warmup: "], self.SamplingParams())
        
        all_metrics = []
        request_id = 0
        start_time = time.time()
        batch_count = 0
        
        while True:
            current_time = time.time()
            elapsed = current_time - start_time
            
            if elapsed >= self.args.duration:
                break
            
            prompt_token_ids, sampling_params = self.generate_request_data(self.args.concurrency)
            
            batch_start = time.time()
            self.llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
            batch_end = time.time()
            
            batch_latency = batch_end - batch_start
            avg_latency_per_req = batch_latency / len(prompt_token_ids)
            
            for i, (prompt, sp) in enumerate(zip(prompt_token_ids, sampling_params)):
                metric = RequestMetrics(
                    request_id=request_id + i,
                    start_time=batch_start,
                    end_time=batch_end,
                    latency=avg_latency_per_req,
                    input_tokens=len(prompt),
                    output_tokens=sp.max_tokens,
                    throughput=sp.max_tokens / avg_latency_per_req if avg_latency_per_req > 0 else 0
                )
                all_metrics.append(metric)
            
            request_id += len(prompt_token_ids)
            batch_count += 1
            
            if batch_count % 5 == 0:
                recent_metrics = all_metrics[-self.args.concurrency * 5:]
                avg_latency = np.mean([m.latency for m in recent_metrics])
                print(f"Time: {elapsed:.1f}s, Batches: {batch_count}, "
                      f"Total Requests: {len(all_metrics)}, Avg Latency: {avg_latency:.2f}s")
        
        return self._calculate_stats(all_metrics)
    
    def _calculate_stats(self, metrics: List[RequestMetrics]) -> BenchmarkStats:
        """计算统计结果"""
        if not metrics:
            return None
        
        latencies = [m.latency for m in metrics]
        total_input_tokens = sum(m.input_tokens for m in metrics)
        total_output_tokens = sum(m.output_tokens for m in metrics)
        
        total_time = max(m.end_time for m in metrics) - min(m.start_time for m in metrics)
        if total_time == 0:
            total_time = sum(latencies)
        
        return BenchmarkStats(
            total_requests=len(metrics),
            successful_requests=len(metrics),
            failed_requests=0,
            total_time=total_time,
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
            avg_latency=np.mean(latencies),
            min_latency=np.min(latencies),
            max_latency=np.max(latencies),
            p50_latency=np.percentile(latencies, 50),
            p90_latency=np.percentile(latencies, 90),
            p95_latency=np.percentile(latencies, 95),
            p99_latency=np.percentile(latencies, 99),
            overall_throughput=total_output_tokens / total_time if total_time > 0 else 0,
            avg_tps=len(metrics) / total_time if total_time > 0 else 0,
            avg_per_req_throughput=np.mean([m.throughput for m in metrics])
        )
    
    def print_results(self, stats: BenchmarkStats, test_type: str):
        """打印测试结果"""
        print(f"\n{'='*60}")
        print(f"测试类型: {test_type}")
        print(f"{'='*60}")
        
        print(f"\n=== 配置信息 ===")
        print(f"Backend: {self.args.backend}")
        print(f"Model: {self.args.model_path}")
        print(f"Tensor Parallel Size: {self.args.tensor_parallel_size}")
        
        if self.args.backend == "nanovllm":
            print(f"Scheduling Policy: {self.args.scheduling_policy}")
            print(f"Preemption Mode: {self.args.preemption_mode}")
            print(f"Number of KV Cache Blocks: {self.args.num_kvcache_blocks}")
        
        print(f"Max Input Length: {self.args.max_input_len}")
        print(f"Max Output Length: {self.args.max_output_len}")
        print(f"GPU Memory Utilization: {self.args.gpu_memory_utilization}")
        print(f"Max Model Length: {self.args.max_model_len}")
        
        if test_type == "并发测试":
            print(f"Concurrency Level: {self.args.concurrency}")
            print(f"Number of Requests: {self.args.num_requests}")
        elif test_type == "压力测试":
            print(f"Duration: {self.args.duration}s")
            print(f"Request Rate: {self.args.request_rate} req/s")
        elif test_type == "持续负载测试":
            print(f"Concurrency: {self.args.concurrency}")
            print(f"Duration: {self.args.duration}s")
        
        print(f"\n=== 测试结果 ===")
        print(f"Total Requests: {stats.total_requests}")
        print(f"Successful Requests: {stats.successful_requests}")
        print(f"Failed Requests: {stats.failed_requests}")
        print(f"Total Time: {stats.total_time:.2f}s")
        print(f"Total Input Tokens: {stats.total_input_tokens}")
        print(f"Total Output Tokens: {stats.total_output_tokens}")
        
        print(f"\n=== 延迟统计 ===")
        print(f"Average Latency: {stats.avg_latency:.4f}s")
        print(f"Min Latency: {stats.min_latency:.4f}s")
        print(f"Max Latency: {stats.max_latency:.4f}s")
        print(f"P50 Latency: {stats.p50_latency:.4f}s")
        print(f"P90 Latency: {stats.p90_latency:.4f}s")
        print(f"P95 Latency: {stats.p95_latency:.4f}s")
        print(f"P99 Latency: {stats.p99_latency:.4f}s")
        
        print(f"\n=== 吞吐量统计 ===")
        print(f"Overall Throughput: {stats.overall_throughput:.2f} tok/s")
        print(f"Average TPS: {stats.avg_tps:.2f} req/s")
        print(f"Average Per-Request Throughput: {stats.avg_per_req_throughput:.2f} tok/s")
        print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Enhanced benchmark script with concurrency and stress testing support for nano-vllm and vLLM")
    
    parser.add_argument("--backend", type=str, default="nanovllm", 
                        choices=["nanovllm", "vllm"], 
                        help="Backend to use: nanovllm or vllm (default: nanovllm)")
    parser.add_argument("--model-path", type=str, default="/data/cjl/Qwen3/Qwen3-0.6B/", help="Model path")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--scheduling-policy", type=str, default="priority", choices=["fcfs", "priority"], 
                        help="Scheduling policy (only for nanovllm)")
    parser.add_argument("--preemption-mode", type=str, default="last_in", choices=["aggressive", "conservative", "last_in"], 
                        help="Preemption mode (only for nanovllm)")
    parser.add_argument("--max-input-len", type=int, default=2048, help="Maximum input length")
    parser.add_argument("--max-output-len", type=int, default=2048, help="Maximum output length")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.7, help="GPU memory utilization")
    parser.add_argument("--max-model-len", type=int, default=8192, help="Maximum model length")
    parser.add_argument("--num-kvcache-blocks", type=int, default=-1, 
                        help="Number of KV cache blocks (only for nanovllm, -1 for auto)")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
                        help="Log level (only for nanovllm)")
    
    parser.add_argument("--test-mode", type=str, default="concurrent", 
                        choices=["concurrent", "stress", "sustained", "all"], 
                        help="Test mode: concurrent (并发测试), stress (压力测试), sustained (持续负载测试), all (所有测试)")
    
    parser.add_argument("--concurrency", type=int, default=16, help="Concurrency level for concurrent/sustained test")
    parser.add_argument("--num-requests", type=int, default=256, help="Total number of requests for concurrent test")
    parser.add_argument("--duration", type=int, default=60, help="Duration in seconds for stress/sustained test")
    parser.add_argument("--request-rate", type=float, default=10.0, help="Request rate (req/s) for stress test")
    
    args = parser.parse_args()
    
    benchmark = ConcurrentBenchmark(args)
    
    print(f"Initializing {args.backend} LLM engine...")
    benchmark.init_llm()
    print(f"{args.backend} LLM engine initialized successfully!")
    
    if args.test_mode == "concurrent":
        stats = benchmark.run_concurrent_test()
        benchmark.print_results(stats, "并发测试")
    
    elif args.test_mode == "stress":
        stats = benchmark.run_stress_test()
        benchmark.print_results(stats, "压力测试")
    
    elif args.test_mode == "sustained":
        stats = benchmark.run_sustained_load_test()
        benchmark.print_results(stats, "持续负载测试")
    
    elif args.test_mode == "all":
        print("\n" + "="*60)
        print("Running all benchmark tests...")
        print("="*60)
        
        stats1 = benchmark.run_concurrent_test()
        benchmark.print_results(stats1, "并发测试")
        
        stats2 = benchmark.run_stress_test()
        benchmark.print_results(stats2, "压力测试")
        
        stats3 = benchmark.run_sustained_load_test()
        benchmark.print_results(stats3, "持续负载测试")
        
        print("\n" + "="*60)
        print("All tests completed!")
        print("="*60)


if __name__ == "__main__":
    main()
