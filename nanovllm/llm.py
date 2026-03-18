from nanovllm.engine.llm_engine import LLMEngine


class LLM(LLMEngine):
    """LLM类，作为模型的主入口，继承自LLMEngine
    
    这个类是对外暴露的主要接口，用户可以通过实例化此类来使用模型。
    它直接继承自LLMEngine，没有添加额外的功能，主要是为了提供一个简洁的接口。
    """
    pass
