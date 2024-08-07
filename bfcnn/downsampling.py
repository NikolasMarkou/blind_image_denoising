import copy
import tensorflow as tf
from tensorflow import keras
from typing import List, Dict, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .constants import *
from .custom_logger import logger
from .utilities import \
    ConvType, \
    conv2d_wrapper


# ---------------------------------------------------------------------

def downsample(
        input_layer,
        downsample_type: str,
        conv_params: Dict = None,
        bn_params: Dict = None,
        ln_params: Dict = None):
    """

    """
    if downsample_type is None or len(downsample_type) <= 0:
        raise ValueError("downsample_type cannot be None or empty")
    downsample_type = downsample_type.lower().strip()

    #
    x = input_layer
    params = copy.deepcopy(conv_params)

    #
    if downsample_type in ["conv2d"]:
        params["kernel_size"] = (2, 2)
        params["strides"] = (2, 2)
        params["padding"] = "same"
        x = \
            conv2d_wrapper(
                input_layer=x,
                bn_params=bn_params,
                ln_params=ln_params,
                conv_params=params)
    elif downsample_type in ["maxpool"]:
        x = \
            tf.keras.layers.MaxPooling2D(
                pool_size=(2, 2), padding="same", strides=(2, 2))(x)
        if conv_params is not None:
            params["kernel_size"] = (1, 1)
            params["strides"] = (1, 1)
            x = \
                conv2d_wrapper(
                    input_layer=x,
                    bn_params=bn_params,
                    ln_params=ln_params,
                    conv_params=params)
    elif downsample_type in ["strides"]:
        x = x[:, ::2, ::2, :]
        
        if conv_params is not None:
            params["kernel_size"] = (1, 1)
            params["strides"] = (1, 1)
            params["padding"] = "same"
            x = \
                conv2d_wrapper(
                    input_layer=x,
                    bn_params=bn_params,
                    ln_params=ln_params,
                    conv_params=params)
    else:
        raise ValueError(
            f"don't know how to handle [{downsample_type}]")

    return x

# ---------------------------------------------------------------------
