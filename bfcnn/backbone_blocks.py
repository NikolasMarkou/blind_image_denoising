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
    conv2d_wrapper, \
    mean_sigma_local, \
    mean_sigma_global

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
        selector_params: Dict = None,
        multiplier_params: Dict = None,
        mean_sigma_params: Dict = None,
        channelwise_params: Dict = None,
        post_addition_activation: str = None,
        expand_type: ExpandType = ExpandType.SAME,
        stop_gradient: bool = False,
        bn_first_conv_params: bool = False,
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
    :param selector_params: selector mixer optional parameters
    :param multiplier_params: learnable multiplier optional parameters
    :param mean_sigma_params: mean sigma normalization parameters
    :param channelwise_params:
        if True add a learnable channelwise learnable multiplier
    :param post_addition_activation:
        activation after the residual addition, None to disable
    :param expand_type: whether to keep same size, compress or expand
    :param stop_gradient: if True, stop gradient before the branch
    :param bn_first_conv_params:
        if True, add a BN before the first conv in residual block
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
    use_selector = selector_params is not None
    use_mean_sigma = mean_sigma_params is not None
    use_multiplier = multiplier_params is not None
    use_channelwise = channelwise_params is not None
    use_post_addition_activation = post_addition_activation is not None

    # out squeeze and excite gating does not use global avg
    # followed by dense layer, because we are using this on large images
    # global averaging looses too much information
    if use_gate:
        pool_bias = gate_params.get("bias", 2.5)

        gate_dense_0_params = dict(
            units=max(int(third_conv_params["filters"] / 4), 2),
            use_bias=False,
            activation="relu",
            kernel_regularizer=third_conv_params.get("kernel_regularizer", "l1"),
            kernel_initializer=third_conv_params.get("kernel_initializer", "glorot_normal")
        )

        gate_dense_1_params = dict(
            units=third_conv_params["filters"],
            use_bias=False,
            activation="hard_sigmoid",
            kernel_regularizer=third_conv_params.get("kernel_regularizer", "l1"),
            kernel_initializer=third_conv_params.get("kernel_initializer", "glorot_normal")
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

        if stop_gradient:
            x = tf.stop_gradient(x)

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

        if second_conv_params is not None:
            x = conv2d_wrapper(input_layer=x,
                               conv_params=copy.deepcopy(second_conv_params),
                               bn_params=bn_params,
                               channelwise_scaling=False)
            x_2nd_conv = x
            gate_layer = x_2nd_conv
        # ---
        if third_conv_params is not None:
            x = conv2d_wrapper(input_layer=x,
                               conv_params=copy.deepcopy(third_conv_params),
                               bn_params=bn_params,
                               channelwise_scaling=False)
            x_3rd_conv = x
            gate_layer = x_3rd_conv

        # compute activation per channel
        if use_gate:
            y = tf.reduce_mean(gate_layer, axis=[1, 2], keepdims=False)
            y = tf.keras.layers.Dense(**gate_dense_0_params)(y)
            # if x < -2.5: return 0
            # if x > 2.5: return 1
            # if -2.5 <= x <= 2.5: return 0.2 * x + 0.5
            y = tf.keras.layers.Dense(**gate_dense_1_params)(y)
            y = tf.expand_dims(y, axis=2)
            y = tf.expand_dims(y, axis=3)
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

        # optional sparsity
        if use_sparsity:
            x = sparse_block(x, **sparse_params)

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

        # skip connector or selector mixer
        if use_selector:
            x = \
                soft_selector_block(
                    input_1_layer=previous_layer,
                    input_2_layer=x,
                    selector_layer=x_2nd_conv,
                    bn_params=None,
                    filters_compress=max(int(third_conv_params["filters"] / 4), 2),
                    filters_target=third_conv_params["filters"],
                    kernel_regularizer=third_conv_params.get("kernel_regularizer", "l1"),
                    kernel_initializer=third_conv_params.get("kernel_initializer", "glorot_normal"))
        else:
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
            keras.layers.AveragePooling2D(
                pool_size=(3, 3),
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
                    interpolation="bilinear")(x)
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


def lunet_blocks(
        input_layer,
        no_levels: int,
        no_layers: int,
        base_conv_params: Dict,
        first_conv_params: Dict,
        second_conv_params: Dict,
        third_conv_params: Dict,
        bn_params: Dict = None,
        gate_params: Dict = None,
        dropout_params: Dict = None,
        multiplier_params: Dict = None,
        add_laplacian: bool = True,
        **kwargs):
    """
    Create a lunet block

    :return: filtered input_layer
    """
    # --- argument check
    if input_layer is None:
        raise ValueError("input_layer must be none")
    if no_layers < 0:
        raise ValueError("no_layers_per_level must be >= 0")

    level_x = input_layer
    levels_x = []
    strides = (2, 2)
    kernel_size = (3, 3)
    interpolation = "bilinear"
    if add_laplacian:
        for level in range(0, no_levels - 1):
            level_x_down = \
                keras.layers.AveragePooling2D(
                    pool_size=kernel_size,
                    strides=strides,
                    padding="same")(level_x)
            level_x_smoothed = \
                keras.layers.UpSampling2D(
                    size=strides,
                    interpolation=interpolation)(level_x_down)
            level_x_diff = level_x - level_x_smoothed
            level_x = level_x_down
            levels_x.append(level_x_diff)
        levels_x.append(level_x)
    else:
        levels_x.append(level_x)
        for level in range(0, no_levels - 1):
            level_x = \
                keras.layers.AveragePooling2D(
                    pool_size=kernel_size,
                    strides=strides,
                    padding="same")(level_x)
            levels_x.append(level_x)

    # --- upside
    x = None
    for level_x in reversed(levels_x):
        level_x = \
            conv2d_wrapper(
                level_x,
                conv_params=base_conv_params,
                bn_params=None)
        if x is None:
            x = level_x
        else:
            level_x = \
                resnet_blocks_full(
                    input_layer=level_x,
                    no_layers=no_layers,
                    first_conv_params=first_conv_params,
                    second_conv_params=second_conv_params,
                    third_conv_params=third_conv_params,
                    bn_params=bn_params,
                    gate_params=gate_params,
                    dropout_params=dropout_params,
                    multiplier_params=multiplier_params
                )
            x = \
                tf.keras.layers.UpSampling2D(
                    size=strides,
                    interpolation=interpolation)(x)
            x = \
                tf.keras.layers.Add()([x, level_x])
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


def renderer(
        signals: List[tf.Tensor],
        masks: List[tf.Tensor]):
    # accumulated mask and signal
    acc_signal = None

    for i in range(len(signals)):
        signal = signals[i]
        mask = masks[i]

        if acc_signal is None:
            acc_signal = tf.multiply(signal, mask)
        else:
            acc_signal = \
                tf.multiply(signal, mask) + \
                tf.multiply(acc_signal, 1.0 - mask)

    return acc_signal

# ---------------------------------------------------------------------


def hard_selector_block(
        input_1_layer,
        input_2_layer,
        selector_layer,
        filters_compress: int,
        filters_target: int,
        bn_params: Dict = None,
        kernel_regularizer: str = "l1",
        kernel_initializer: str = "glorot_normal",
        **kwargs):
    """
    from 2 input layers,
    select a combination of the 2 with bias on the first one

    :return: filtered input_layer
    """
    # --- argument checking
    if filters_target is None:
        raise ValueError("filters_target should not be None")

    # --- set variables
    # out squeeze and excite gating does not use global avg
    # followed by dense layer, because we are using this on large images
    # global averaging looses too much information
    selector_dense_0_params = dict(
        units=filters_compress,
        use_bias=False,
        activation="relu",
        kernel_regularizer=kernel_regularizer,
        kernel_initializer=kernel_initializer)

    selector_dense_1_params = dict(
        units=filters_target,
        use_bias=False,
        activation="relu",
        kernel_regularizer=kernel_regularizer,
        kernel_initializer=kernel_initializer)

    # --- setup network
    x = selector_layer

    # transformation
    x = tf.reduce_mean(x, axis=[1, 2], keepdims=False)

    if filters_compress is not None:
        x = dense_wrapper(
            input_layer=x,
            dense_params=selector_dense_0_params,
            bn_params=None)

    x = dense_wrapper(
        input_layer=x,
        dense_params=selector_dense_1_params,
        bn_params=bn_params)

    # if x < -2.5: return 0
    # if x > 2.5: return 1
    # if -2.5 <= x <= 2.5: return 0.2 * x + 0.5
    x = tf.keras.activations.hard_sigmoid(2.5 - x)

    return \
        tf.keras.layers.Multiply()([input_1_layer, x]) + \
        tf.keras.layers.Multiply()([input_2_layer, 1.0 - x])

# ---------------------------------------------------------------------


def soft_selector_block(
        input_1_layer,
        input_2_layer,
        selector_layer,
        filters_compress: int,
        filters_target: int,
        bn_params: Dict = None,
        kernel_regularizer: str = "l1",
        kernel_initializer: str = "glorot_normal",
        **kwargs):
    """
    from 2 input layers,
    select a combination of the 2 with bias on the first one

    :return: filtered input_layer
    """
    # --- argument checking
    if filters_target is None:
        raise ValueError("filters_target should not be None")

    # --- set variables
    # out squeeze and excite gating does not use global avg
    # followed by dense layer, because we are using this on large images
    # global averaging looses too much information
    selector_dense_0_params = dict(
        units=filters_compress,
        use_bias=False,
        activation="relu",
        kernel_regularizer=kernel_regularizer,
        kernel_initializer=kernel_initializer)

    selector_dense_1_params = dict(
        units=filters_target,
        use_bias=False,
        activation="linear",
        kernel_regularizer=kernel_regularizer,
        kernel_initializer=kernel_initializer)

    # --- setup network
    x = selector_layer

    # transformation
    x = tf.reduce_mean(x, axis=[1, 2], keepdims=False)

    if filters_compress is not None:
        x = dense_wrapper(
            input_layer=x,
            dense_params=selector_dense_0_params,
            bn_params=None)

    x = dense_wrapper(
        input_layer=x,
        dense_params=selector_dense_1_params,
        bn_params=bn_params)

    x = tf.keras.activations.sigmoid(2.5 - x)

    return \
        tf.keras.layers.Multiply()([input_1_layer, x]) + \
        tf.keras.layers.Multiply()([input_2_layer, 1.0 - x])


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
