from dataclasses import dataclass


@dataclass
class SamplingParams:
    """采样参数类，用于配置模型生成文本时的采样策略"""
    # 温度参数，控制采样的随机性，值越大随机性越强
    temperature: float = 1.0
    # 生成的最大token数
    max_tokens: int = 64
    # 是否忽略结束标记，若为True则会一直生成直到达到max_tokens
    ignore_eos: bool = False

    def __post_init__(self):
        """初始化后验证参数"""
        # 确保温度参数大于一个很小的值，不允许使用贪婪采样
        assert self.temperature > 1e-10, "greedy sampling is not permitted"
