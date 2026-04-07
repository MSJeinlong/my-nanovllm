from nanovllm.config import Config
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence
from nanovllm.utils.monitor import monitor

# 配置日志级别
monitor.set_log_level(monitor.DEBUG)

# 创建配置
config = Config()
config.max_num_seqs = 4
config.max_num_batched_tokens = 2048
config.eos = 2
config.scheduling_policy = 'priority'
config.preemption_mode = 'aggressive'
config.max_prefill_tokens = 512
config.min_decode_seqs = 1
config.num_kvcache_blocks = 16
config.kvcache_block_size = 16

# 创建调度器
scheduler = Scheduler(config)

# 创建测试序列
sequences = []
for i in range(5):
    seq = Sequence(
        seq_id=i,
        prompt_tokens=[1, 2, 3, 4, 5] * (i + 1),  # 不同长度的prompt
        max_tokens=50,
        ignore_eos=False
    )
    sequences.append(seq)
    # 添加序列到调度器
    scheduler.add(seq)
    print(f"Added sequence {i} to waiting queue")

# 模拟调度过程
print("\nSimulating scheduling process...")
for i in range(3):
    print(f"\n--- Scheduling round {i+1} ---")
    seqs, is_prefill = scheduler.schedule()
    print(f"Scheduled {len(seqs)} sequences, is_prefill: {is_prefill}")
    
    # 模拟序列完成
    if not is_prefill and seqs:
        # 随机标记一个序列为完成
        finished_seq = seqs[0]
        print(f"Marking sequence {finished_seq.seq_id} as finished")
        # 模拟后处理
        scheduler.postprocess([finished_seq], [config.eos])

# 打印最终状态
print("\n--- Final state ---")
print(f"Waiting queue length: {len(scheduler.waiting)}")
print(f"Running queue length: {len(scheduler.running)}")
print("\nScheduler stats:")
print(scheduler.get_stats())
