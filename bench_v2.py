import os
import time
import argparse
import statistics
from random import randint, seed
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime

# 全局变量
NanoLLM = None
VLLM = None
SamplingParams = None
VLLMSamplingParams = None
NANO_AVAILABLE = False
VLLM_AVAILABLE = False

@dataclass
class RequestMetrics:
    """单次请求的指标"""
    request_id: int
    start_time: float
    end_time: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    success: bool = True
    error: Optional[str] = None
    
    @property
    def latency(self) -> float:
        """总延迟（秒）"""
        return self.end_time - self.start_time
    
    @property
    def tps(self) -> float:
        """每秒生成的token数"""
        if self.latency > 0 and self.completion_tokens > 0:
            return self.completion_tokens / self.latency
        return 0.0

class LLMPerformanceTester:
    """大模型性能测试器"""
    
    def __init__(self, model_path, max_model_len, engine="nano-vllm"):
        self.model_path = model_path
        self.max_model_len = max_model_len
        self.engine = engine
        self.llm = None
        self.results: List[RequestMetrics] = []
        self.lock = threading.Lock()
        
        # 初始化模型
        self._init_model()
    
    def _init_model(self):
        """初始化模型"""
        global NanoLLM, VLLM, SamplingParams, VLLMSamplingParams, NANO_AVAILABLE, VLLM_AVAILABLE
        
        if self.engine == "nano-vllm":
            try:
                from nanovllm import LLM as NanoLLM, SamplingParams
                NANO_AVAILABLE = True
                print(f"Initializing nano-vllm with model: {self.model_path}")
                self.llm = NanoLLM(self.model_path, enforce_eager=False, max_model_len=self.max_model_len)
            except ImportError as e:
                print(f"导入 nano-vllm 失败: {e}")
                raise ImportError(f"nano-vllm is not available. Please install it first. 错误详情: {e}") from e
            except Exception as e:
                print(f"初始化 nano-vllm 失败: {e}")
                import traceback
                traceback.print_exc()
                raise RuntimeError(f"Failed to initialize nano-vllm. 错误详情: {e}") from e
        elif self.engine == "vllm":
            try:
                from vllm import LLM as VLLM, SamplingParams as VLLMSamplingParams
                VLLM_AVAILABLE = True
                # 对于vllm，使用VLLMSamplingParams
                global SamplingParams
                SamplingParams = VLLMSamplingParams
                print(f"Initializing vllm with model: {self.model_path}")
                self.llm = VLLM(self.model_path, max_model_len=self.max_model_len, gpu_memory_utilization=0.7)
            except ImportError as e:
                print(f"导入 vllm 失败: {e}")
                raise ImportError(f"vllm is not available. Please install it first. 错误详情: {e}") from e
            except Exception as e:
                print(f"初始化 vllm 失败: {e}")
                import traceback
                traceback.print_exc()
                raise RuntimeError(f"Failed to initialize vllm. 错误详情: {e}") from e
        else:
            raise ValueError(f"Unsupported engine: {self.engine}")
    
    def _generate(self, prompt, sampling_params):
        """执行单次生成"""
        with self.lock:
            if self.engine == "vllm":
                # vllm 需要不同的输入格式
                if isinstance(prompt, list):
                    prompt = {"prompt_token_ids": prompt}
                return self.llm.generate([prompt], sampling_params)
            else:
                # nano-vllm 格式
                return self.llm.generate([prompt], sampling_params)
    
    def _make_request(self, request_id: int, prompt, sampling_params) -> RequestMetrics:
        """执行单次请求并收集指标"""
        start_time = time.time()
        metrics = RequestMetrics(
            request_id=request_id,
            start_time=start_time,
            end_time=start_time
        )
        
        try:
            # 执行生成
            outputs = self._generate(prompt, sampling_params)
            
            # 解析结果
            if outputs and len(outputs) > 0:
                output = outputs[0]
                
                # 处理字典格式输出（nano-vllm）
                if isinstance(output, dict):
                    # 从 token_ids 计算 completion_tokens
                    if 'token_ids' in output:
                        metrics.completion_tokens = len(output['token_ids'])
                    # 从 text 字段估算 completion_tokens
                    elif 'text' in output:
                        # 简单估算：假设平均每个token对应4个字符
                        metrics.completion_tokens = len(output['text']) // 4
                    # 从 prompt 计算 prompt_tokens
                    metrics.prompt_tokens = len(prompt) if isinstance(prompt, list) else 0
                    # 计算总 tokens
                    metrics.total_tokens = metrics.prompt_tokens + metrics.completion_tokens
                # 处理对象格式输出（vllm）
                elif hasattr(output, "usage"):
                    metrics.prompt_tokens = output.usage.prompt_tokens
                    metrics.completion_tokens = output.usage.completion_tokens
                    metrics.total_tokens = output.usage.total_tokens
                # 其他格式
                else:
                    # 尝试估算
                    metrics.prompt_tokens = len(prompt) if isinstance(prompt, list) else 0
                    metrics.completion_tokens = sampling_params.max_tokens
                    metrics.total_tokens = metrics.prompt_tokens + metrics.completion_tokens
            
            metrics.end_time = time.time()
            metrics.success = True
            
        except Exception as e:
            metrics.end_time = time.time()
            metrics.success = False
            metrics.error = str(e)
            print(f"[Request {request_id}] 失败: {e}")
        
        return metrics
    
    def run_single_test(self, prompt, sampling_params):
        """运行单次测试"""
        print(f"\n{'='*60}")
        print("单次性能测试")
        print(f"{'='*60}")
        
        metrics = self._make_request(0, prompt, sampling_params)
        
        if metrics.success:
            print(f"\n结果详情:")
            print(f"  总延迟: {metrics.latency:.3f} 秒")
            print(f"  Prompt Tokens: {metrics.prompt_tokens}")
            print(f"  Completion Tokens: {metrics.completion_tokens}")
            print(f"  Total Tokens: {metrics.total_tokens}")
            print(f"  生成速度: {metrics.tps:.2f} tokens/秒")
        
        return metrics
    
    def run_concurrent_test(self, prompts, sampling_params_list, concurrency=5):
        """运行并发测试"""
        print(f"\n{'='*60}")
        print(f"并发性能测试 (请求数: {len(prompts)}, 并发: {concurrency})")
        print(f"{'='*60}")
        
        self.results = []
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {
                executor.submit(self._make_request, i, prompts[i], sampling_params_list[i]): i 
                for i in range(len(prompts))
            }
            
            for future in as_completed(futures):
                metrics = future.result()
                with self.lock:
                    self.results.append(metrics)
                # 打印单次请求结果
                if metrics.success:
                    print(f"[Request {metrics.request_id}] 成功 | "
                          f"延迟: {metrics.latency:.3f}s | "
                          f"Tokens: {metrics.completion_tokens} | "
                          f"TPS: {metrics.tps:.2f} tokens/秒")
                else:
                    print(f"[Request {metrics.request_id}] 失败: {metrics.error}")
        
        total_time = time.time() - start_time
        return self._calculate_statistics(total_time, len(prompts), concurrency)
    
    def run_stress_test(self, prompt_template, sampling_params, duration_seconds=60, concurrency=10):
        """运行压力测试"""
        print(f"\n{'='*60}")
        print(f"压力测试 (持续时间: {duration_seconds}秒, 并发: {concurrency})")
        print(f"{'='*60}")
        
        self.results = []
        stop_event = threading.Event()
        request_id = [0]
        
        def worker():
            while not stop_event.is_set():
                current_id = request_id[0]
                request_id[0] += 1
                # 生成随机长度的输入
                input_len = randint(100, 1024)
                prompt = [randint(0, 10000) for _ in range(input_len)]
                # 使用相同的采样参数
                metrics = self._make_request(current_id, prompt, sampling_params)
                with self.lock:
                    self.results.append(metrics)
        
        threads = []
        start_time = time.time()
        
        # 启动工作线程
        for _ in range(concurrency):
            t = threading.Thread(target=worker)
            t.daemon = True
            t.start()
            threads.append(t)
        
        # 运行指定时间
        time.sleep(duration_seconds)
        stop_event.set()
        
        # 等待所有线程完成当前请求
        for t in threads:
            t.join(timeout=10)
        
        total_time = time.time() - start_time
        return self._calculate_statistics(total_time, len(self.results), concurrency)
    
    def _calculate_statistics(self, total_time: float, num_requests: int, concurrency: int) -> Dict:
        """计算并打印统计结果"""
        successful = [r for r in self.results if r.success]
        failed = [r for r in self.results if not r.success]
        
        if not successful:
            print("警告: 所有请求均失败!")
            return {}
        
        # 基础指标
        latencies = [r.latency for r in successful]
        tps_values = [r.tps for r in successful if r.tps > 0]
        completion_tokens_list = [r.completion_tokens for r in successful]
        
        total_completion_tokens = sum(completion_tokens_list)
        total_prompt_tokens = sum(r.prompt_tokens for r in successful)
        
        # 计算指标
        stats = {
            "test_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_requests": num_requests,
            "successful_requests": len(successful),
            "failed_requests": len(failed),
            "success_rate": len(successful) / num_requests * 100,
            "concurrency": concurrency,
            "total_duration_sec": total_time,
            
            # 延迟指标 (秒)
            "latency_avg": statistics.mean(latencies),
            "latency_min": min(latencies),
            "latency_max": max(latencies),
            "latency_p50": statistics.median(latencies),
            "latency_p95": sorted(latencies)[int(len(latencies)*0.95)] if len(latencies) > 1 else latencies[0],
            "latency_p99": sorted(latencies)[int(len(latencies)*0.99)] if len(latencies) > 1 else latencies[0],
            
            # 吞吐量指标
            "throughput_qps": len(successful) / total_time,  # 每秒查询数
            "total_tokens_generated": total_completion_tokens,
            "tokens_per_sec": total_completion_tokens / total_time,  # 全局TPS
            "avg_tps_per_request": statistics.mean(tps_values) if tps_values else 0,
            
            # Token统计
            "avg_completion_tokens": statistics.mean(completion_tokens_list),
            "total_prompt_tokens": total_prompt_tokens,
        }
        
        # 打印结果
        print(f"\n{'='*60}")
        print("性能测试报告")
        print(f"{'='*60}")
        print(f"测试时间: {stats['test_time']}")
        print(f"总请求数: {stats['total_requests']} | "
              f"成功: {stats['successful_requests']} | "
              f"失败: {stats['failed_requests']} | "
              f"成功率: {stats['success_rate']:.1f}%")
        print(f"并发数: {stats['concurrency']} | "
              f"总耗时: {stats['total_duration_sec']:.2f}秒")
        
        print(f"\n--- 延迟指标 (Latency) ---")
        print(f"平均延迟: {stats['latency_avg']:.3f}s")
        print(f"最小延迟: {stats['latency_min']:.3f}s")
        print(f"最大延迟: {stats['latency_max']:.3f}s")
        print(f"P50延迟: {stats['latency_p50']:.3f}s")
        print(f"P95延迟: {stats['latency_p95']:.3f}s")
        print(f"P99延迟: {stats['latency_p99']:.3f}s")
        
        print(f"\n--- 吞吐量指标 (Throughput) ---")
        print(f"QPS (每秒查询数): {stats['throughput_qps']:.2f}")
        print(f"全局生成速度: {stats['tokens_per_sec']:.2f} tokens/秒")
        print(f"单请求平均TPS: {stats['avg_tps_per_request']:.2f} tokens/秒")
        print(f"总生成Tokens: {stats['total_tokens_generated']}")
        print(f"平均输出长度: {stats['avg_completion_tokens']:.1f} tokens")
        
        return stats

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="大模型性能测试工具")
    parser.add_argument('--model-path', type=str, default="/data/cjl/Qwen3/Qwen3-0.6B/",
                        help="模型路径，默认为 /data/cjl/Qwen3/Qwen3-0.6B/")
    parser.add_argument('--max-model-len', type=int, default=16384,
                        help="模型最大长度，默认为 16384")
    parser.add_argument('--engine', type=str, default="nano-vllm", choices=["nano-vllm", "vllm"],
                        help="推理架构，默认为 nano-vllm")
    parser.add_argument('--test-mode', type=str, default="all", choices=["single", "concurrent", "stress", "all"],
                        help="测试模式，默认为 all")
    parser.add_argument('--num-requests', type=int, default=100,
                        help="并发测试的请求数，默认为 100")
    parser.add_argument('--concurrency', type=int, default=5,
                        help="并发数，默认为 5")
    parser.add_argument('--duration', type=int, default=60,
                        help="压力测试持续时间（秒），默认为 60")
    
    args = parser.parse_args()
    
    # 初始化测试器
    try:
        tester = LLMPerformanceTester(
            model_path=args.model_path,
            max_model_len=args.max_model_len,
            engine=args.engine
        )
    except Exception as e:
        print(f"初始化测试器失败: {e}")
        return
    
    print(f"大模型性能测试工具")
    print(f"模型路径: {args.model_path}")
    print(f"模型最大长度: {args.max_model_len}")
    print(f"推理架构: {args.engine}")
    
    # 生成测试数据
    seed(0)
    max_input_len = 1024
    max_output_len = 1024
    
    # 预热
    print("\n预热模型...")
    warmup_prompt = [randint(0, 10000) for _ in range(100)]
    warmup_sampling_params = SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=50)
    tester._generate(warmup_prompt, warmup_sampling_params)
    print("预热完成")
    
    # 运行测试
    try:
        if args.test_mode in ["single", "all"]:
            # 单次测试
            test_prompt = [randint(0, 10000) for _ in range(randint(100, max_input_len))]
            test_sampling_params = SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, max_output_len))
            tester.run_single_test(test_prompt, test_sampling_params)
        
        if args.test_mode in ["concurrent", "all"]:
            # 并发测试
            prompts = [[randint(0, 10000) for _ in range(randint(100, max_input_len))] for _ in range(args.num_requests)]
            sampling_params_list = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, max_output_len)) for _ in range(args.num_requests)]
            input("\n按Enter开始并发测试...")
            tester.run_concurrent_test(prompts, sampling_params_list, concurrency=args.concurrency)
        
        if args.test_mode in ["stress", "all"]:
            # 压力测试
            test_sampling_params = SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, max_output_len))
            input("\n按Enter开始压力测试...")
            tester.run_stress_test(None, test_sampling_params, duration_seconds=args.duration, concurrency=args.concurrency)
            
    except KeyboardInterrupt:
        print("\n测试被用户中断")
    except Exception as e:
        print(f"测试出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()