import warnings

import torch
import torch.nn as nn
from types import SimpleNamespace

from liger_kernel.transformers import LigerRMSNorm
from liger_kernel.transformers import LigerLayerNorm as LayerNorm
from liger_kernel.transformers import LigerSwiGLUMLP



class FeedForward(LigerSwiGLUMLP):
    def __init__(self,
                 dim,
                 hidden_dim,
                 multiple_of,
                 ffn_dim_multiplier,
                 ):
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        config = SimpleNamespace(
            hidden_size=dim,
            intermediate_size=hidden_dim,
            hidden_act="silu",
        )
        super().__init__(config)


class RMSNorm(LigerRMSNorm):
    def __init__(self,
        hidden_size,
        eps=1e-6,
        offset=0.0,
        casting_mode="llama",
        init_fn="ones",
        in_place=True,
    ):
        super().__init__(hidden_size=hidden_size, eps=eps, offset=offset, casting_mode=casting_mode, init_fn=init_fn, in_place=in_place)
        self.hidden_size = hidden_size


class DynamicTanh(nn.Module):
    def __init__(self, num_features, alpha_init_value=0.5):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
    
    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        return x * self.weight + self.bias
    
    @staticmethod
    def convert_ln_to_dyt(module):
        module_output = module
        if isinstance(module, RMSNorm) or isinstance(module, LayerNorm):
            module_output = DynamicTanh(module.hidden_size)
        for name, child in module.named_children():
            module_output.add_module(name, DynamicTanh.convert_ln_to_dyt(child))
        del module
        return module_output