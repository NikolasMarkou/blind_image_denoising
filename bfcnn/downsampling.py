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
from .regularizers import (
    SoftOrthonormalConstraintRegularizer)

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
    if downsample_type in ["conv2d_2x2"]:
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
        if conv_params is not None:
            params["kernel_size"] = (1, 1)
            params["strides"] = (2, 2)
            params["padding"] = "same"
            x = \
                conv2d_wrapper(
                    input_layer=x,
                    bn_params=bn_params,
                    ln_params=ln_params,
                    conv_params=params)
        else:
            x = x[:, ::2, ::2, :]
    elif downsample_type in ["conv2d_1x1_orthonormal"]:
        if conv_params is not None:
            params["kernel_size"] = (1, 1)
            params["strides"] = (2, 2)
            params["padding"] = "same"
            params["kernel_initializer"] = \
                tf.keras.initializers.truncated_normal(
                    mean=0.0,
                    stddev=DEFAULT_SOFTORTHONORMAL_STDDEV)
            params["kernel_regularizer"] = \
                SoftOrthonormalConstraintRegularizer(
                    lambda_coefficient=DEFAULT_SOFTORTHONORMAL_LAMBDA,
                    l1_coefficient=DEFAULT_SOFTORTHONORMAL_L1,
                    l2_coefficient=DEFAULT_SOFTORTHONORMAL_L2)
            x = conv2d_wrapper(
                input_layer=x,
                bn_params=bn_params,
                ln_params=ln_params,
                conv_params=params,
                conv_type=ConvType.CONV2D)
        else:
            x = x[:, ::2, ::2, :]
    else:
        raise ValueError(
            f"don't know how to handle [{downsample_type}]")

    return x

# ---------------------------------------------------------------------
