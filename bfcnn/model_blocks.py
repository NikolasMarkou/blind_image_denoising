import copy
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
    ChannelwiseMultiplier
from .utilities import \
    ConvType, \
    sparse_block, \
    dense_wrapper, \
    conv2d_wrapper


# ---------------------------------------------------------------------


class ExpandType(Enum):
    SAME = 0

    COMPRESS = 1

    EXPAND = 2

    @staticmethod
    def from_string(type_str: str) -> "ExpandType":
        # --- argument checking
        if type_str is None:
            raise ValueError("type_str must not be null")
        if not isinstance(type_str, str):
            raise ValueError("type_str must be string")
        type_str = type_str.strip().upper()
        if len(type_str) <= 0:
            raise ValueError("stripped type_str must not be empty")

        # --- clean string and get
        return ExpandType[type_str]

    def to_string(self) -> str:
        return self.name


def resnet_blocks_full(
        input_layer,
        no_layers: int,
        first_conv_params: Dict,
        second_conv_params: Dict,
        third_conv_params: Dict,
        bn_params: Dict = None,
        gate_params: Dict = None,
        sparse_params: Dict = None,
        dropout_params: Dict = None,
        multiplier_params: Dict = None,
        channelwise_params: Dict = None,
        post_addition_activation: str = None,
        expand_type: ExpandType = ExpandType.SAME,
        **kwargs):
    """
    Create a series of residual network blocks

    :param input_layer: the input layer to perform on
    :param no_layers: how many residual network blocks to add
    :param first_conv_params: the parameters of the first conv
    :param second_conv_params: the parameters of the middle conv
    :param third_conv_params: the parameters of the third conv
    :param bn_params: batch normalization parameters
    :param sparse_params: sparse parameters
    :param gate_params: gate optional parameters
    :param dropout_params: dropout optional parameters
    :param multiplier_params: learnable optional parameters
    :param channelwise_params: if True add a learnable point-wise depthwise scaling conv2d
    :param post_addition_activation: activation after the residual addition, None to disable
    :param expand_type: whether to keep same size, compress or expand

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
    use_sparsity = sparse_params is not None
    use_multiplier = multiplier_params is not None
    use_channelwise = channelwise_params is not None
    use_post_addition_activation = post_addition_activation is not None

    # out squeeze and excite gating does not use global avg
    # followed by dense layer, because we are using this on large images
    # global averaging looses too much information
    if use_gate:
        pool_type = gate_params.get("pool_type", "avg")
        pool_size = gate_params.get("pool_size", (32, 32))

        pool_params = dict(
            pool_size=pool_size,
            padding="same",
            strides=None
        )

        if pool_type == "max":
            pool = tf.keras.layers.MaxPooling2D
        elif pool_type == "avg":
            pool = tf.keras.layers.AveragePooling2D
        else:
            raise ValueError(f"not valid pool_type [{pool_type}[")

        squeeze_excite_params = dict(
            kernel_size=1,
            strides=(1, 1),
            padding="same",
            use_bias=False,
            activation="linear",
            filters=third_conv_params["filters"],
            kernel_regularizer=third_conv_params.get("kernel_regularizer", "l1"),
            kernel_initializer=third_conv_params.get("kernel_initializer", "glorot_normal")
        )

    # --- setup resnet along with its variants
    x = input_layer

    # create a series of residual blocks
    for i in range(no_layers):
        previous_layer = x

        if first_conv_params is not None:
            x = conv2d_wrapper(input_layer=x,
                               conv_params=copy.deepcopy(first_conv_params),
                               bn_params=None,
                               channelwise_scaling=False)

        # sparsity goes here (first conv selects the signal),
        # then sparsity picks it up, second and third conv filter it
        if use_sparsity:
            x = sparse_block(x, **sparse_params)

        if second_conv_params is not None:
            x = conv2d_wrapper(input_layer=x,
                               conv_params=copy.deepcopy(second_conv_params),
                               bn_params=bn_params,
                               channelwise_scaling=False)
        if third_conv_params is not None:
            x = conv2d_wrapper(input_layer=x,
                               conv_params=copy.deepcopy(third_conv_params),
                               bn_params=bn_params,
                               channelwise_scaling=False)

        # compute activation per channel
        if use_gate:
            y = pool(**pool_params)(x)
            y = conv2d_wrapper(input_layer=y,
                               conv_params=copy.deepcopy(squeeze_excite_params),
                               bn_params=None,
                               channelwise_scaling=True)
            # if x < -2.5: return 0
            # if x > 2.5: return 1
            # if -2.5 <= x <= 2.5: return 0.2 * x + 0.5
            y = tf.keras.activations.hard_sigmoid(y)
            x = tf.keras.layers.Multiply()([x, y])

        # fix x dimensions and previous layers
        if expand_type == ExpandType.COMPRESS:
            expand_conv_params = copy.deepcopy(third_conv_params)
            expand_conv_params["strides"] = (2, 2)
            x = conv2d_wrapper(input_layer=x,
                               conv_params=copy.deepcopy(third_conv_params),
                               bn_params=bn_params,
                               conv_type=ConvType.CONV2D,
                               channelwise_scaling=False)
            previous_layer = \
                tf.keras.layers.MaxPooling2D(
                    pool_size=(3, 3),
                    strides=(2, 2),
                    padding="valid")
        elif expand_type == ExpandType.EXPAND:
            expand_conv_params = copy.deepcopy(third_conv_params)
            expand_conv_params["dilation_rate"] = (2, 2)
            x = conv2d_wrapper(input_layer=x,
                               conv_params=copy.deepcopy(third_conv_params),
                               bn_params=bn_params,
                               conv_type=ConvType.CONV2D_TRANSPOSE,
                               channelwise_scaling=False)
            previous_layer = \
                tf.keras.layers.UpSampling2D(
                    size=(2, 2),
                    interpolation="nearest")
        elif expand_type == ExpandType.SAME:
            # do nothing
            pass
        else:
            raise ValueError(f"dont know how to handle [{expand_type}]")

        # optional channelwise multiplier
        if use_channelwise:
            x = ChannelwiseMultiplier(**channelwise_params)(x)
        # optional multiplier
        if use_multiplier:
            x = Multiplier(**multiplier_params)(x)
        if use_dropout:
            x = RandomOnOff(**dropout_params)(x)

        # skip connection
        x = tf.keras.layers.Add()([x, previous_layer])
        # optional post addition activation
        if use_post_addition_activation:
            x = tf.keras.layers.Activation(post_addition_activation)(x)
    return x


# ---------------------------------------------------------------------

def resnet_compress_expand_full(
        input_layer,
        no_layers: int,
        **kwargs):
    """
    Create a series of residual network blocks

    :param input_layer: the input layer to perform on
    :param no_layers: how many residual network blocks to add

    :return: filtered input_layer
    """
    # --- argument check
    if input_layer is None:
        raise ValueError("input_layer must be none")
    if no_layers <= 0:
        raise ValueError("no_layers must be > 0")

    # --- setup resnet along with its variants
    x = input_layer

    # compress
    for i in range(no_layers):
        if i+1 % 3 == 0:
            expand_type = ExpandType.COMPRESS
        else:
            expand_type = ExpandType.SAME
        x = \
            resnet_blocks_full(
                input_layer=x,
                no_layers=1,
                expand_type=expand_type,
                **kwargs)

    # expand
    for i in range(no_layers):
        if i+1 % 3 == 0:
            expand_type = ExpandType.EXPAND
        else:
            expand_type = ExpandType.SAME
        x = \
            resnet_blocks_full(
                input_layer=x,
                no_layers=1,
                expand_type=expand_type,
                **kwargs)
    return x

# ---------------------------------------------------------------------


def resnet(
        input_layer,
        no_layers: int,
        first_conv_params: Dict,
        second_conv_params: Dict,
        bn_params: Dict = None,
        post_addition_activation: str = "relu",
        **kwargs):
    """
    Create a series of residual network blocks,
    this is the original resnet block implementation
    Deep Residual Learning for Image Recognition [2015]

    :param input_layer: the input layer to perform on
    :param no_layers: how many residual network blocks to add
    :param first_conv_params: the parameters of the first conv
    :param second_conv_params: the parameters of the middle conv
    :param bn_params: batch normalization parameters
    :param post_addition_activation: activation after the addition, None to disable

    :return: filtered input_layer
    """
    # --- argument check
    if input_layer is None:
        raise ValueError("input_layer must be none")
    if no_layers < 0:
        raise ValueError("no_layers must be >= 0")
    use_bn_params = bn_params is not None

    # --- setup resnet
    x = input_layer

    # --- create several number of residual blocks
    for i in range(no_layers):
        previous_layer = x
        x = conv2d_wrapper(input_layer=x,
                           conv_params=first_conv_params,
                           bn_params=None,
                           channelwise_scaling=False)
        x = conv2d_wrapper(input_layer=x,
                           conv_params=second_conv_params,
                           bn_params=bn_params,
                           channelwise_scaling=False)
        if use_bn_params:
            x = keras.layers.BatchNormalization(**bn_params)(x)
        # skip connection
        x = keras.layers.Add()([x, previous_layer])
        # post addition activation
        x = keras.layers.Activation(post_addition_activation)(x)
    return x


# ---------------------------------------------------------------------


def resnet_full_preactivation(
        input_layer,
        no_layers: int,
        first_conv_params: Dict,
        second_conv_params: Dict,
        bn_params: Dict = None,
        pre_activation: str = "relu",
        **kwargs):
    """
    Create a series of residual network blocks,
    this is the modified residual network
    Identity Mappings in Deep Residual Networks [2016]

    :param input_layer: the input layer to perform on
    :param no_layers: how many residual network blocks to add
    :param first_conv_params: the parameters of the first conv
    :param second_conv_params: the parameters of the middle conv
    :param bn_params: batch normalization parameters
    :param pre_activation: pre_activation parameter, default to "relu", None to disable

    :return: filtered input_layer
    """
    # --- argument check
    if input_layer is None:
        raise ValueError("input_layer must be none")
    if no_layers < 0:
        raise ValueError("no_layers must be >= 0")

    # --- setup resnet
    x = input_layer

    # --- create several number of residual blocks
    for i in range(no_layers):
        previous_layer = x
        x = conv2d_wrapper(input_layer=x,
                           conv_params=first_conv_params,
                           bn_params=bn_params,
                           pre_activation=pre_activation,
                           channelwise_scaling=False)
        x = conv2d_wrapper(input_layer=x,
                           conv_params=second_conv_params,
                           bn_params=bn_params,
                           pre_activation=pre_activation,
                           channelwise_scaling=False)
        # skip connection
        x = tf.keras.layers.Add()([x, previous_layer])
    return x


# ---------------------------------------------------------------------


def builder(input_layer, config: Dict):
    # --- argument checking
    if config is None:
        raise ValueError("config cannot be None")

    # --- get type and params
    block_type = config.get(TYPE_STR, None).lower()
    block_params = config.get(CONFIG_STR, {})

    # --- select block
    fn = None
    if block_type == "resnet":
        fn = resnet
    elif block_type == "resnet_full_preactivation":
        fn = resnet_full_preactivation
    elif block_type == "resnext":
        raise NotImplementedError("not implemented yet")
    elif block_type == "convnext":
        raise NotImplementedError("not implemented yet")
    elif block_type == "mobilenet_v1":
        raise NotImplementedError("not implemented yet")
    elif block_type == "mobilenet_v2":
        raise NotImplementedError("not implemented yet")
    elif block_type == "mobilenet_v3":
        raise NotImplementedError("not implemented yet")

    # --- build it
    return fn(input_layer=input_layer, **block_params)

# ---------------------------------------------------------------------