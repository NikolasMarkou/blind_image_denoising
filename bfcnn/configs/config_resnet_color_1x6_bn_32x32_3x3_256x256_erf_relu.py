from copy import deepcopy
from .config_base import config as base

# ---------------------------------------------------------------------

config = deepcopy(base)

config["model_denoise"] = {
    "filters": 32,
    "no_layers": 6,
    "kernel_size": 7,
    "block_kernels": [3, 3],
    "block_filters": [32, 32],
    "value_range": [0, 255],
    "type": "resnet",
    "batchnorm": True,
    "use_bias": False,
    "activation": "relu",
    "final_activation": "tanh",
    "base_activation": "linear",
    "input_shape": ["?", "?", 3],
    "kernel_initializer": "glorot_normal",
    "kernel_regularizer": {
        "type": "erf",
        "config": {
            "l2_coefficient": 0.0,
            "l1_coefficient": 0.025
        }
    }
  }

# ---------------------------------------------------------------------