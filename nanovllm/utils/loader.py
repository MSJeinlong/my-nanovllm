import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    """默认权重加载器
    
    Args:
        param: 模型参数
        loaded_weight: 加载的权重
    """
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str):
    """加载模型权重
    
    Args:
        model: 模型对象
        path: 权重文件路径
    """
    # 获取打包模块映射
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    # 遍历所有safetensors文件
    for file in glob(os.path.join(path, "*.safetensors")):
        # 打开safetensors文件
        with safe_open(file, "pt", "cpu") as f:
            # 遍历所有权重名称
            for weight_name in f.keys():
                # 检查是否在打包模块映射中
                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        # 替换权重名称
                        param_name = weight_name.replace(k, v)
                        # 获取参数
                        param = model.get_parameter(param_name)
                        # 获取权重加载器
                        weight_loader = getattr(param, "weight_loader")
                        # 加载权重
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break
                else:
                    # 如果不在打包模块映射中
                    param = model.get_parameter(weight_name)
                    # 获取权重加载器，默认为default_weight_loader
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    # 加载权重
                    weight_loader(param, f.get_tensor(weight_name))
