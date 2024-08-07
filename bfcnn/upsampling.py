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

def upsample(
        input_layer,
        upsample_type: str,
        conv_params: Dict = None,
        bn_params: Dict = None,
        ln_params: Dict = None):
    """

    """
    if upsample_type is None or len(upsample_type) <= 0:
        raise ValueError("upsample_type cannot be None or empty")
    upsample_type = upsample_type.lower().strip()

    #
    x = input_layer
    params = copy.deepcopy(conv_params)

    #
    if upsample_type == "conv2d_transpose":
        # lower level, upscale
        x = conv2d_wrapper(
            input_layer=x,
            bn_params=bn_params,
            ln_params=ln_params,
            conv_params=params,
            conv_type=ConvType.CONV2D_TRANSPOSE)
    elif upsample_type in ["upsample_bilinear_conv2d"]:
        params["kernel_size"] = (3, 3)
        params["strides"] = (1, 1)
        # lower level, upscale
        x = \
            tf.keras.layers.UpSampling2D(
                size=(2, 2),
                interpolation="bilinear")(x)
        x = conv2d_wrapper(
            input_layer=x,
            bn_params=bn_params,
            ln_params=ln_params,
            conv_params=params,
            conv_type=ConvType.CONV2D)
    elif upsample_type in ["upsample_nearest_conv2d"]:
        params["kernel_size"] = (3, 3)
        params["strides"] = (1, 1)
        params["padding"] = "same"
        # lower level, upscale
        x = \
            tf.keras.layers.UpSampling2D(
                size=(2, 2),
                interpolation="nearest")(x)
        x = conv2d_wrapper(
            input_layer=x,
            bn_params=bn_params,
            ln_params=ln_params,
            conv_params=params,
            conv_type=ConvType.CONV2D)
    elif upsample_type in ["conv2d_1x1_nearest"]:
        params["kernel_size"] = (1, 1)
        params["strides"] = (1, 1)
        params["padding"] = "same"
        params["kernel_initializer"] = \
            tf.keras.initializers.truncated_normal(mean=0.0, stddev=0.02)
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
        x = \
            tf.keras.layers.UpSampling2D(
                size=(2, 2),
                interpolation="nearest")(x)
    elif upsample_type in ["upsample_bilinear_conv2d_v1"]:
        params["kernel_size"] = (1, 1)
        params["strides"] = (1, 1)
        params["padding"] = "same"
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
        x = \
            tf.keras.layers.UpSampling2D(
                size=(2, 2),
                interpolation="bilinear")(x)
    elif upsample_type in ["upsample_laplacian_conv2d"]:
        # specialized for laplacian network
        params["kernel_size"] = (1, 1)
        params["strides"] = (1, 1)
        params["padding"] = "same"
        # if both functions are linear then interchange them to get some speedup
        if params.get("activation", "linear") == "linear":
            x = conv2d_wrapper(
                input_layer=x,
                bn_params=bn_params,
                ln_params=ln_params,
                conv_params=params,
                conv_type=ConvType.CONV2D)
            x = \
                tf.keras.layers.UpSampling2D(
                    size=(2, 2),
                    interpolation="bilinear")(x)
        else:
            x = \
                tf.keras.layers.UpSampling2D(
                    size=(2, 2),
                    interpolation="bilinear")(x)
            x = conv2d_wrapper(
                input_layer=x,
                bn_params=bn_params,
                ln_params=ln_params,
                conv_params=params,
                conv_type=ConvType.CONV2D)
    elif upsample_type in ["nn", "nearest"]:
        # upsampling performed by nearest neighbor interpolation
        x = \
            tf.keras.layers.UpSampling2D(
                size=(2, 2),
                interpolation="nearest")(x)
    elif upsample_type in ["bilinear"]:
        # upsampling performed by bilinear interpolation
        # this is the default in
        # https://github.com/MrGiovanni/UNetPlusPlus/
        # blob/master/pytorch/nnunet/network_architecture/generic_UNetPlusPlus.py
        x = \
            tf.keras.layers.UpSampling2D(
                size=(2, 2),
                interpolation="bilinear")(x)
    else:
        raise ValueError(
            f"don't know how to handle [{upsample_type}]")

    return x

# ---------------------------------------------------------------------
