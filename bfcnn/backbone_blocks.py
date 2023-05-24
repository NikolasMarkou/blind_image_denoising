import copy
import math
from enum import Enum

import tensorflow as tf
from tensorflow import keras
from typing import List, Tuple, Union, Dict, Iterable

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .custom_logger import logger
from .constants import *
from .custom_layers import \
    Multiplier, \
    RandomOnOff, \
    GradientDropout, \
    ChannelwiseMultiplier, \
    DifferentiableGateLayer
from .custom_layers_selector import \
    ScaleType, \
    ActivationType, \
    selector_block
from .utilities import \
    ConvType, \
    sparse_block, \
    dense_wrapper, \
    conv2d_wrapper, \
    mean_sigma_local, \
    mean_sigma_global


# ---------------------------------------------------------------------


def convnext_blocks_full(
        input_layer,
        **kwargs):
    """
    Create a series of convnext residual blocks

    :param input_layer: the input layer to perform on
    :return: filtered input_layer
    """
    # --- argument check
    if input_layer is None:
        raise ValueError("input_layer must be none")

    kwargs["bn_params"] = None
    kwargs["bn_first_conv_params"] = False
    kwargs["ln_after_first_conv_params"] = True

    return resnet_blocks_full(input_layer=input_layer, **kwargs)

# ---------------------------------------------------------------------


def resnet_blocks_full(
        input_layer,
        no_layers: int,
        first_conv_params: Dict,
        second_conv_params: Dict,
        third_conv_params: Dict,
        bn_params: Dict = None,
        gate_params: Dict = None,
        dropout_params: Dict = None,
        selector_params: Dict = None,
        multiplier_params: Dict = None,
        mean_sigma_params: Dict = None,
        channelwise_params: Dict = None,
        gradient_dropout_params: Dict = None,
        post_addition_activation: str = None,
        bn_first_conv_params: bool = False,
        ln_after_first_conv_params: bool = False,
        **kwargs):
    """
    Create a series of residual network blocks

    :param input_layer: the input layer to perform on
    :param no_layers: how many residual network blocks to add
    :param first_conv_params: the parameters of the first conv
    :param second_conv_params: the parameters of the middle conv
    :param third_conv_params: the parameters of the third conv
    :param bn_params: batch normalization parameters
    :param gate_params: gate optional parameters
    :param dropout_params: dropout optional parameters
    :param selector_params: selector mixer optional parameters
    :param multiplier_params: learnable multiplier optional parameters
    :param mean_sigma_params: mean sigma normalization parameters
    :param channelwise_params:
        if True add a learnable channelwise learnable multiplier
    :param gradient_dropout_params:
    :param post_addition_activation:
        activation after the residual addition, None to disable
    :param bn_first_conv_params:
        if True, add a BN before the first conv in residual block
    :param ln_after_first_conv_params:
        if True, add a LN after the first conv in residual block
    :return: filtered input_layer
    """
    # --- argument check
    if input_layer is None:
        raise ValueError("input_layer must be none")
    if no_layers < 0:
        raise ValueError("no_layers must be >= 0")

    # --- set variables
    use_gate = gate_params is not None
    use_dropout = dropout_params is not None
    use_selector = selector_params is not None
    use_mean_sigma = mean_sigma_params is not None
    use_multiplier = multiplier_params is not None
    use_channelwise = channelwise_params is not None
    use_gradient_dropout = gradient_dropout_params is not None
    use_post_addition_activation = post_addition_activation is not None

    # out squeeze and excite gating does not use global avg
    # followed by dense layer, because we are using this on large images
    # global averaging looses too much information
    if use_gate:
        gate_no_filters = 0
        if "filters" in second_conv_params:
            gate_no_filters = second_conv_params["filters"]
        elif "depth_multiplier" in second_conv_params:
            gate_no_filters = \
                first_conv_params["filters"] * \
                second_conv_params["depth_multiplier"]
        else:
            raise ValueError("don't know what to do here")

        gate_dense_0_params = dict(
            units=max(int(gate_no_filters / 8), 2),
            use_bias=False,
            activation="relu",
            kernel_regularizer="l2",
            kernel_initializer="glorot_normal"
        )

        gate_dense_1_params = dict(
            units=gate_no_filters,
            use_bias=False,
            activation="hard_sigmoid",
            kernel_regularizer="l2",
            kernel_initializer="glorot_normal"
        )

    # --- setup resnet along with its variants
    x = input_layer

    # create a series of residual blocks
    for i in range(no_layers):
        x_1st_conv = None
        x_2nd_conv = None
        x_3rd_conv = None
        gate_layer = None
        previous_layer = x

        if use_mean_sigma:
            x_mean_global, x_sigma_global = \
                mean_sigma_global(
                    input_layer=x,
                    axis=[1, 2])
            x = (x - x_mean_global) / (x_sigma_global + DEFAULT_EPSILON)

        if first_conv_params is not None and not bn_first_conv_params:
            x = conv2d_wrapper(input_layer=x,
                               conv_params=copy.deepcopy(first_conv_params),
                               bn_params=None,
                               channelwise_scaling=False)
            x_1st_conv = x
            gate_layer = x_1st_conv
        elif first_conv_params is not None and bn_first_conv_params:
            x = conv2d_wrapper(input_layer=x,
                               conv_params=copy.deepcopy(first_conv_params),
                               bn_params=bn_params,
                               channelwise_scaling=False)
            x_1st_conv = x
            gate_layer = x_1st_conv

        if ln_after_first_conv_params:
            x = tf.keras.layers.LayerNormalization(center=False, scale=True)(x)
            x_1st_conv = x

        if second_conv_params is not None:
            x = conv2d_wrapper(input_layer=x,
                               conv_params=copy.deepcopy(second_conv_params),
                               bn_params=bn_params,
                               channelwise_scaling=False)
            x_2nd_conv = x
            gate_layer = x_2nd_conv

        # compute activation per channel
        if use_gate:
            y = tf.reduce_mean(gate_layer, axis=[1, 2], keepdims=False)
            y = tf.keras.layers.Dense(**gate_dense_0_params)(y)
            # if x < -2.5: return 0
            # if x > 2.5: return 1
            # if -2.5 <= x <= 2.5: return 0.2 * x + 0.5
            y = tf.keras.layers.Dense(**gate_dense_1_params)(y)
            y = tf.expand_dims(y, axis=1)
            y = tf.expand_dims(y, axis=2)
            x = tf.keras.layers.Multiply()([x, y])

        if third_conv_params is not None:
            x = conv2d_wrapper(input_layer=x,
                               conv_params=copy.deepcopy(third_conv_params),
                               bn_params=bn_params,
                               channelwise_scaling=False)

        # optional channelwise multiplier
        if use_channelwise:
            x = ChannelwiseMultiplier(**channelwise_params)(x)

        # optional multiplier
        if use_multiplier:
            x = Multiplier(**multiplier_params)(x)

        # scale back to original
        if use_mean_sigma:
            x = (x * x_sigma_global) + x_mean_global

        # optional dropout on/off
        if use_dropout:
            x = RandomOnOff(**dropout_params)(x)

        # optional gradient dropout
        if use_gradient_dropout:
            if gradient_dropout_params.get("progressive", True):
                depth_percentage = max(0.1, min(0.9, float(i) / float(no_layers)))
                x = GradientDropout(probability_off=depth_percentage)(x)
            else:
                x = GradientDropout(probability_off=0.5)(x)

        # skip connector or selector mixer
        if use_selector:
            if x_1st_conv is not None:
                x_selector = x_1st_conv
            else:
                raise ValueError("don't know what selector layer to use")

            x = \
                selector_block(
                    input_1_layer=previous_layer,
                    input_2_layer=x,
                    selector_layer=x_selector,
                    **selector_params)
        else:
            # skip connection
            x = tf.keras.layers.Add()([x, previous_layer])
        # optional post addition activation
        if use_post_addition_activation:
            x = tf.keras.layers.Activation(post_addition_activation)(x)
    return x


# ---------------------------------------------------------------------


def unet_blocks(
        input_layer,
        no_levels: int,
        no_layers: int,
        first_conv_params: Dict,
        second_conv_params: Dict,
        third_conv_params: Dict,
        bn_params: Dict = None,
        gate_params: Dict = None,
        dropout_params: Dict = None,
        multiplier_params: Dict = None,
        **kwargs):
    """
    Create a unet block

    :return: filtered input_layer
    """
    # --- argument check
    if input_layer is None:
        raise ValueError("input_layer must be none")
    if no_layers < 0:
        raise ValueError("no_layers_per_level must be >= 0")

    # --- setup unet
    x = input_layer
    levels_x = []

    # --- downside
    for i in range(no_levels):
        if i > 0:
            x = \
                conv2d_wrapper(
                    x,
                    conv_params=first_conv_params,
                    bn_params=None)
        x = \
            resnet_blocks_full(
                input_layer=x,
                no_layers=no_layers,
                first_conv_params=first_conv_params,
                second_conv_params=second_conv_params,
                third_conv_params=third_conv_params,
                bn_params=bn_params,
                gate_params=gate_params,
                dropout_params=dropout_params,
                multiplier_params=multiplier_params
            )
        levels_x.append(x)
        x = \
            keras.layers.MaxPooling2D(
                pool_size=(2, 2),
                strides=(2, 2),
                padding="same")(x)

    # --- upside
    x = None
    for level_x in reversed(levels_x):
        if x is None:
            x = level_x
        else:
            x = \
                tf.keras.layers.UpSampling2D(
                    size=(2, 2),
                    interpolation="nearest")(x)
            x = \
                tf.keras.layers.Concatenate()([x, level_x])
        x = \
            conv2d_wrapper(
                x,
                conv_params=first_conv_params,
                bn_params=None)
        x = \
            resnet_blocks_full(
                input_layer=x,
                no_layers=no_layers,
                first_conv_params=first_conv_params,
                second_conv_params=second_conv_params,
                third_conv_params=third_conv_params,
                bn_params=bn_params,
                gate_params=gate_params,
                dropout_params=dropout_params,
                multiplier_params=multiplier_params
            )

    return x

# ---------------------------------------------------------------------


def details(input_layer):
    x = input_layer
    x_mean = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
    x_variance = \
        tf.reduce_mean(
            tf.square(x - x_mean), axis=[1, 2], keepdims=True)
    x_sigma = tf.sqrt(x_variance + DEFAULT_EPSILON)
    x = (x - x_mean) / x_sigma
    x = tf.math.pow(tf.nn.tanh(8.0 * x), 4.0) * x
    return x

# ---------------------------------------------------------------------

