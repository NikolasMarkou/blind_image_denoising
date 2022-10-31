from copy import deepcopy
from .config_resnet_color_1x6_bn_32x32_3x3_256x256_erf_relu import config as base
# ---------------------------------------------------------------------

config = deepcopy(base)

config["model_denoise"]["no_layers"] = 12

# ---------------------------------------------------------------------
