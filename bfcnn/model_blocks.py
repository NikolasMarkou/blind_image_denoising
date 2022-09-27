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
    sparse_block, \
    dense_wrapper, \
    conv2d_wrapper

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


def resnet_blocks(
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
        channelwise_scaling: bool = False,
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
    :param channelwise_scaling: if True add a learnable point-wise depthwise scaling conv2d

    :return: filtered input_layer
    """
    # --- argument check
    if input_layer is None:
        raise ValueError("input_layer must be none")
    if no_layers < 0:
        raise ValueError("no_layers must be >= 0")
    use_gate = gate_params is not None
    use_dropout = dropout_params is not None
    use_sparsity = sparse_params is not None
    use_multiplier = multiplier_params is not None

    elementwise_params = dict(
        multiplier=1.0,
        regularizer="l1",
        activation="relu"
    )

    dense_params = dict(
        use_bias=False,
        activation="relu",
        units=third_conv_params["filters"],
        kernel_regularizer=third_conv_params.get("kernel_regularizer", "l1"),
        kernel_initializer=third_conv_params.get("kernel_initializer", "glorot_normal")
    )

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
        x = conv2d_wrapper(input_layer=x,
                           conv_params=third_conv_params,
                           bn_params=bn_params,
                           channelwise_scaling=False)
        if use_sparsity:
            x = sparse_block(x, **sparse_params)
        # learn the proper scale of the previous layer
        if channelwise_scaling:
            # add a very small l1 penalty
            x = \
                ChannelwiseMultiplier(
                    multiplier=1.0,
                    regularizer=keras.regularizers.L1(DEFAULT_CHANNELWISE_MULTIPLIER_L1),
                    trainable=True,
                    activation="linear")(x)
        # compute activation per channel
        if use_gate:
            y = tf.keras.layers.GlobalAveragePooling2D()(x)
            y = \
                dense_wrapper(
                    input_layer=y,
                    bn_params=bn_params,
                    dense_params=dense_params,
                    elementwise_params=elementwise_params)
            # if x < -2.5: return 0
            # if x > 2.5: return 1
            # if -2.5 <= x <= 2.5: return 0.2 * x + 0.5
            y = tf.keras.activations.hard_sigmoid(2.5 - y)
            y = tf.expand_dims(y, axis=1)
            y = tf.expand_dims(y, axis=1)
            x = tf.keras.layers.Multiply()([x, y])
        # optional multiplier
        if use_multiplier:
            x = Multiplier(**multiplier_params)(x)
        if use_dropout:
            x = RandomOnOff(**dropout_params)(x)
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
