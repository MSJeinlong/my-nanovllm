from collections import deque, defaultdict
import heapq

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager
from nanovllm.utils.monitor import monitor


class Scheduler:
    """调度器类，负责调度序列的执行"""

    def __init__(self, config: Config):
        """初始化Scheduler对象

        Args:
            config: 配置对象
        """
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        
        # 调度策略配置
        self.scheduling_policy = config.scheduling_policy
        self.preemption_mode = config.preemption_mode
        self.max_prefill_tokens = config.max_prefill_tokens
        self.min_decode_seqs = config.min_decode_seqs

        monitor.info(f"[Scheduler] 初始化调度器 - max_num_seqs={self.max_num_seqs}, max_num_batched_tokens={self.max_num_batched_tokens}")
        monitor.info(f"[Scheduler] 调度策略 - policy={self.scheduling_policy}, preemption={self.preemption_mode}")

        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        
        # 等待队列 - 使用优先队列实现优先级调度
        self.waiting = []  # 优先队列，元素为 (priority, seq_id, seq)
        self.waiting_seq_map = {}  # 快速查找序列
        
        # 运行队列 - 使用双端队列
        self.running: deque[Sequence] = deque()
        
        # 统计信息
        self.stats = defaultdict(int)

        monitor.info(f"[Scheduler] 调度器初始化完成")

    def is_finished(self):
        """检查是否所有请求都已完成
        
        Returns:
            是否所有请求都已完成
        """
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        """添加序列到等待队列

        Args:
            seq: 要添加的序列
        """
        # 计算优先级
        if self.scheduling_policy == 'priority':
            # 优先级策略：较短的prompt优先，避免长prompt阻塞
            priority = len(seq)  # 长度越短优先级越高
        else:  # fcfs
            # 先来先服务策略
            priority = seq.seq_id  # 序列ID作为优先级
        
        # 添加到优先队列
        heapq.heappush(self.waiting, (priority, seq.seq_id, seq))
        self.waiting_seq_map[seq.seq_id] = seq
        
        # 记录队列状态
        monitor.debug(f"[Scheduler] 序列加入等待队列 - seq_id={seq.seq_id}, priority={priority}, "
                     f"waiting_size={len(self.waiting)}, running_size={len(self.running)}")
        self.stats['total_requests'] += 1

    def schedule(self) -> tuple[list[Sequence], bool]:
        """调度序列

        调度器的核心方法，负责从等待队列和运行队列中选择序列进行处理。
        处理分为两个阶段：
        1. Prefill阶段：处理新的序列，分配KV缓存块
        2. Decode阶段：处理正在运行的序列，生成下一个token

        当KV缓存不足时，会触发序列抢占机制，将正在运行的序列暂时挂起，
        以腾出空间给新的序列或其他需要继续运行的序列。

        Returns:
            tuple[list[Sequence], bool]: 调度的序列列表和是否为prefill阶段
        """
        monitor.start_timer("schedule")

        # prefill阶段 - 智能批处理
        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0
        remaining_waiting = []

        # 动态调整批处理大小
        available_blocks = len(self.block_manager.free_block_ids)
        if available_blocks > 0:
            # 基于可用块数调整批处理大小
            estimated_seqs = min(available_blocks // 2, self.max_num_seqs)
            max_batch_size = min(estimated_seqs, self.max_num_seqs)
        else:
            max_batch_size = self.min_decode_seqs

        # 从优先队列中取出序列进行调度
        while self.waiting and num_seqs < max_batch_size:
            priority, seq_id, seq = heapq.heappop(self.waiting)
            del self.waiting_seq_map[seq_id]
            
            # 检查是否超过最大token数或无法分配块
            prefill_tokens = len(seq) - seq.num_cached_tokens
            if num_batched_tokens + prefill_tokens > self.max_prefill_tokens or not self.block_manager.can_allocate(seq):
                monitor.debug(f"[Scheduler] prefill调度中断 - token_limit={num_batched_tokens + prefill_tokens}或无法分配块")
                remaining_waiting.append((priority, seq_id, seq))
                break
            
            num_seqs += 1
            self.block_manager.allocate(seq)
            num_batched_tokens += prefill_tokens
            seq.status = SequenceStatus.RUNNING
            self.running.append(seq)
            scheduled_seqs.append(seq)
            monitor.debug(f"[Scheduler] prefill调度序列 - seq_id={seq.seq_id}, priority={priority}, tokens={len(seq)}, cached={seq.num_cached_tokens}")
            monitor.debug(f"[Scheduler] 队列状态 - waiting={len(self.waiting)}, running={len(self.running)}")

        # 将未调度的序列放回队列
        for item in remaining_waiting:
            heapq.heappush(self.waiting, item)
            self.waiting_seq_map[item[1]] = item[2]

        # 如果有调度的序列，则返回prefill阶段
        if scheduled_seqs:
            schedule_time = monitor.end_timer("schedule")
            # 记录缓存使用情况
            used_blocks = self.block_manager.get_used_blocks()
            utilization = self.block_manager.get_cache_utilization()
            monitor.debug(f"[Scheduler] prefill调度完成 - seqs={len(scheduled_seqs)}, tokens={num_batched_tokens}, time={schedule_time:.4f}s")
            monitor.debug(f"[Scheduler] 缓存状态 - used_blocks={used_blocks}, total_blocks={self.block_manager.total_blocks}, "
                        f"utilization={utilization * 100:.2f}%")
            self.stats['prefill_steps'] += 1
            self.stats['prefill_seqs'] += len(scheduled_seqs)
            return scheduled_seqs, True

        # decode阶段 - 智能抢占
        preempted_count = 0
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()
            
            # 检查是否可以追加token
            while not self.block_manager.can_append(seq):
                # 检查当前序列是否需要新块
                need_new_block = len(seq) % self.block_manager.block_size == 1
                free_blocks = len(self.block_manager.free_block_ids)
                
                monitor.debug(f"[Scheduler] decode阶段空间不足 - seq_id={seq.seq_id}, tokens={len(seq)}, "
                           f"need_new_block={need_new_block}, free_blocks={free_blocks}")
                
                if self.running:
                    # 智能抢占策略：选择最合适的序列进行抢占
                    preempted_seq = self._select_preemptible_sequence()
                    if preempted_seq:
                        monitor.info(f"[Scheduler] 智能抢占序列 - seq_id={preempted_seq.seq_id}, "
                                   f"tokens={len(preempted_seq)}, remaining_tokens={preempted_seq.max_tokens - preempted_seq.num_completion_tokens}")
                        self.preempt(preempted_seq)
                        preempted_count += 1
                        # 重新检查当前序列是否可以追加
                        continue
                
                # 无法抢占其他序列，抢占当前序列
                monitor.info(f"[Scheduler] 抢占当前序列 - seq_id={seq.seq_id}, tokens={len(seq)}")
                self.preempt(seq)
                preempted_count += 1
                break
            else:
                num_seqs += 1
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)

        assert scheduled_seqs, "调度失败：没有序列可以调度"
        
        # 维护运行队列的顺序
        self.running.extendleft(reversed(scheduled_seqs))

        schedule_time = monitor.end_timer("schedule")
        # 记录缓存使用情况
        used_blocks = self.block_manager.get_used_blocks()
        utilization = self.block_manager.get_cache_utilization()
        if preempted_count > 0:
            monitor.debug(f"[Scheduler] decode调度完成 - seqs={len(scheduled_seqs)}, preempted={preempted_count}, time={schedule_time:.4f}s")
        else:
            monitor.debug(f"[Scheduler] decode调度完成 - seqs={len(scheduled_seqs)}, time={schedule_time:.4f}s")
        monitor.debug(f"[Scheduler] 缓存状态 - used_blocks={used_blocks}, total_blocks={self.block_manager.total_blocks}, "
                    f"utilization={utilization * 100:.2f}%")
        
        self.stats['decode_steps'] += 1
        self.stats['decode_seqs'] += len(scheduled_seqs)
        self.stats['preemptions'] += preempted_count

        return scheduled_seqs, False


    def _select_preemptible_sequence(self) -> Sequence:
        """智能选择要抢占的序列

        Returns:
            Sequence: 要抢占的序列
        """
        if not self.running:
            return None
        
        # 新策略：抢占运行队列中最后的一个请求（刚加入运行的请求）
        if self.preemption_mode == 'last_in':
            # 从运行队列末尾获取最后加入的序列
            if self.running:
                # deque 支持负索引，-1 表示最后一个元素
                last_seq = self.running[-1]
                monitor.debug(f"[Scheduler] 选择最后加入的序列进行抢占 - seq_id={last_seq.seq_id}")
                return last_seq
            return None
        
        # 基于多种因素选择要抢占的序列
        """
        选择最合适的序列进行抢占（不支持中间状态保存）。
        原则：优先选择投入计算最少（完成比例最低）的序列，以最小化浪费；
        在此基础上，优先选择资源占用较大（序列较长）的序列，以释放更多空间。
        支持 aggressive 和 conservative 两种模式，但两者都避免抢占高完成比例的序列。
        """
        candidates = []
        for seq in self.running:
            # 1. 计算完成比例 (已生成token / 最大token)，处理 max_tokens 为0的情况
            if seq.max_tokens > 0:
                completion_ratio = min(seq.num_completion_tokens / seq.max_tokens, 1.0)
            else:
                completion_ratio = 0.0  # 无上限时视为刚刚开始
            
            # 2. 浪费程度：完成比例越低，浪费越小 → 得分越高
            waste_score = 1.0 - completion_ratio
            
            # 3. 释放收益：序列长度越大，释放空间越多 → 得分越高（归一化因子可调）
            length_score = len(seq) / 1000.0
            
            # 4. 根据模式选择不同的评分权重
            if self.preemption_mode == 'aggressive':
                # 激进模式：极度看重释放空间，同时考虑避免浪费
                score = 0.3 * waste_score + 0.7 * length_score
            else:  # conservative
                # 保守模式：更看重避免浪费，同时考虑释放空间
                score = 0.8 * waste_score + 0.2 * length_score
            
            candidates.append((-score, seq))  # 负号用于最小堆
        
        # 选择分数最高的序列
        if candidates:
            candidates.sort()
            return candidates[0][1]
        return None

    def preempt(self, seq: Sequence):
        """抢占序列

        Args:
            seq: 要抢占的序列
        """
        monitor.warning(f"[Scheduler] 抢占序列 - seq_id={seq.seq_id}, tokens={len(seq)}")
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        
        # 将抢占的序列重新加入等待队列
        if self.scheduling_policy == 'priority':
            priority = len(seq)
        else:
            priority = seq.seq_id
        
        # 从运行队列中移除
        if seq in self.running:
            self.running.remove(seq)
        
        heapq.heappush(self.waiting, (priority, seq.seq_id, seq))
        self.waiting_seq_map[seq.seq_id] = seq
        
        monitor.debug(f"[Scheduler] 队列状态 - waiting={len(self.waiting)}, running={len(self.running)}")
        monitor.increment("preempted_sequences")
        self.stats['preemptions'] += 1

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[bool]:
        """后处理序列

        Args:
            seqs: 序列列表
            token_ids: 生成的token ID列表
        """
        finished_count = 0
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)
            # 检查是否完成
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                finish_reason = "eos" if token_id == self.eos else "max_tokens"
                monitor.info(f"[Scheduler] 序列完成 - seq_id={seq.seq_id}, reason={finish_reason}, "
                           f"prompt_tokens={seq.num_prompt_tokens}, completion_tokens={seq.num_completion_tokens}")
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                
                # 从运行队列中移除
                if seq in self.running:
                    self.running.remove(seq)
                
                finished_count += 1
                monitor.increment("finished_sequences")
                self.stats['finished_sequences'] += 1

        if finished_count > 0:
            monitor.debug(f"[Scheduler] 后处理完成 - finished={finished_count}, running={len(self.running)}, waiting={len(self.waiting)}")
            monitor.debug(f"[Scheduler] 队列状态 - waiting={len(self.waiting)}, running={len(self.running)}")
            
        # 定期清理等待队列中已完成的序列（防御性编程）
        if len(self.waiting) > 0 and finished_count > 0:
            self._clean_waiting_queue()

    def _clean_waiting_queue(self):
        """清理等待队列中已完成的序列
        
        防御性编程：确保等待队列中只包含活跃的序列
        """
        cleaned_waiting = []
        cleaned_count = 0
        
        while self.waiting:
            priority, seq_id, seq = heapq.heappop(self.waiting)
            if seq.status != SequenceStatus.FINISHED:
                cleaned_waiting.append((priority, seq_id, seq))
            else:
                cleaned_count += 1
                if seq_id in self.waiting_seq_map:
                    del self.waiting_seq_map[seq_id]
        
        # 重新构建等待队列
        for item in cleaned_waiting:
            heapq.heappush(self.waiting, item)
            self.waiting_seq_map[item[1]] = item[2]
        
        if cleaned_count > 0:
            monitor.debug(f"[Scheduler] 清理等待队列 - 移除 {cleaned_count} 个已完成序列")

    def get_stats(self):
        """获取调度器统计信息
        
        Returns:
            dict: 统计信息
        """
        stats = dict(self.stats)
        stats['waiting_seqs'] = len(self.waiting)
        stats['running_seqs'] = len(self.running)
        stats['free_blocks'] = len(self.block_manager.free_block_ids)
        return stats
