import tensorflow as tf
from tensorflow import keras
from typing import List, Tuple, Union, Dict, Iterable

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .custom_logger import logger
from .custom_layers import Multiplier
from .model_blocks import resnet_blocks
from .utilities import conv2d_wrapper, mean_sigma_local
from .constants import DEFAULT_BN_EPSILON, DEFAULT_BN_MOMENTUM

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
                resnet_blocks(
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
            resnet_blocks(
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


def build_model_lunet(
        input_dims,
        no_levels: int,
        no_layers: int,
        kernel_size: int,
        filters: int,
        activation: str = "relu",
        final_activation: str = "linear",
        use_bn: bool = True,
        use_bias: bool = False,
        kernel_regularizer="l1",
        kernel_initializer="glorot_normal",
        channel_index: int = 2,
        dropout_rate: float = -1,
        add_skip_with_input: bool = True,
        add_sparsity: bool = False,
        add_gates: bool = False,
        add_var: bool = False,
        add_final_bn: bool = False,
        add_intermediate_results: bool = False,
        add_learnable_multiplier: bool = False,
        add_projection_to_input: bool = True,
        add_concat_input: bool = False,
        add_laplacian: bool = True,
        name="lunet",
        **kwargs) -> keras.Model:
    """
    builds a lu-net model

    :param input_dims: Models input dimensions
    :param no_levels: Number of unet layers
    :param no_layers: Number of resnet layers per unet level
    :param kernel_size: kernel size of the conv layers
    :param filters: number of filters per convolutional layer
    :param activation: intermediate activation
    :param final_activation: activation of the final layer
    :param channel_index: Index of the channel in dimensions
    :param dropout_rate: probability of resnet block shutting off
    :param use_bn: Use Batch Normalization
    :param use_bias: use bias
    :param kernel_regularizer: Kernel weight regularizer
    :param kernel_initializer: Kernel weight initializer
    :param add_skip_with_input: if true skip with input
    :param add_sparsity: if true add sparsity layer
    :param add_gates: if true add gate layer
    :param add_var: if true add variance for each block
    :param add_final_bn: add a batch norm after the resnet blocks
    :param add_intermediate_results: if true output results before projection
    :param add_learnable_multiplier:
    :param add_projection_to_input: if true project to input tensor channel number
    :param add_concat_input: if true concat input to intermediate before projecting
    :param add_laplacian: if true each level of lunet is a laplacian, if false a gaussian
    :param name: name of the model
    :return: unet model
    """
    # --- logging
    logger.info("building unet")
    logger.info(f"parameters not used: {kwargs}")

    # --- setup parameters
    bn_params = dict(
        center=use_bias,
        scale=True,
        momentum=DEFAULT_BN_MOMENTUM,
        epsilon=DEFAULT_BN_EPSILON
    )

    gate_params = dict(
        kernel_size=1,
        filters=filters,
        strides=(1, 1),
        padding="same",
        use_bias=use_bias,
        activation=activation,
        kernel_regularizer=kernel_regularizer,
        kernel_initializer=kernel_initializer
    )

    base_conv_params = dict(
        kernel_size=3,
        filters=filters,
        strides=(1, 1),
        padding="same",
        use_bias=use_bias,
        activation=activation,
        kernel_regularizer=kernel_regularizer,
        kernel_initializer=kernel_initializer,
    )

    first_conv_params = dict(
        filters=filters,
        strides=(1, 1),
        padding="same",
        use_bias=use_bias,
        activation=activation,
        kernel_size=kernel_size,
        kernel_regularizer=kernel_regularizer,
        kernel_initializer=kernel_initializer,
    )

    second_conv_params = dict(
        kernel_size=3,
        filters=filters * 2,
        strides=(1, 1),
        padding="same",
        use_bias=use_bias,
        activation=activation,
        kernel_regularizer=kernel_regularizer,
        kernel_initializer=kernel_initializer
    )

    third_conv_params = dict(
        groups=2,
        kernel_size=1,
        filters=filters,
        strides=(1, 1),
        padding="same",
        use_bias=use_bias,
        # this must be the same as the base
        activation=activation,
        kernel_regularizer=kernel_regularizer,
        kernel_initializer=kernel_initializer
    )

    final_conv_params = dict(
        kernel_size=1,
        strides=(1, 1),
        padding="same",
        use_bias=use_bias,
        # this must be linear because it is capped later
        activation="linear",
        filters=input_dims[channel_index],
        kernel_regularizer=kernel_regularizer,
        kernel_initializer=kernel_initializer
    )

    multiplier_params = dict(
        multiplier=1.0,
        trainable=True,
        regularizer="l1",
        activation="linear"
    )

    dropout_params = dict(
        rate=dropout_rate
    )

    lunet_params = dict(
        no_levels=no_levels,
        no_layers=no_layers,
        bn_params=bn_params,
        add_laplacian=add_laplacian,
        base_conv_params=base_conv_params,
        first_conv_params=first_conv_params,
        second_conv_params=second_conv_params,
        third_conv_params=third_conv_params,
    )

    # make it linear so it gets sparse afterwards
    if add_sparsity:
        base_conv_params["activation"] = "linear"

    if add_gates:
        lunet_params["gate_params"] = gate_params

    if add_learnable_multiplier:
        lunet_params["multiplier_params"] = multiplier_params

    if dropout_rate != -1:
        lunet_params["dropout_params"] = dropout_params

    # --- build model
    # set input
    input_layer = \
        keras.Input(
            name="input_tensor",
            shape=input_dims)
    x = input_layer
    y = input_layer

    if add_var:
        _, x_var = \
            mean_sigma_local(
                input_layer=x,
                kernel_size=(5, 5))
        x = tf.keras.layers.Concatenate()([x, x_var])

    # add lunet blocks
    x = \
        lunet_blocks(
            input_layer=x,
            **lunet_params)

    # optional batch norm
    if add_final_bn:
        x = tf.keras.layers.BatchNormalization(**bn_params)(x)

    # optional concat and mix with input
    if add_concat_input:
        y_tmp = y
        if use_bn:
            y_tmp = tf.keras.layers.BatchNormalization(**bn_params)(y_tmp)
        x = tf.keras.layers.Concatenate()([x, y_tmp])

    # --- output layer branches here,
    # to allow space for intermediate results
    output_layer = x

    # --- output to original channels / projection
    if add_projection_to_input:
        output_layer = \
            tf.keras.layers.Conv2D(
                **final_conv_params)(output_layer)

        # learnable multiplier
        if add_learnable_multiplier:
            output_layer = \
                Multiplier(**multiplier_params)(output_layer)

        # cap it off to limit values
        output_layer = \
            tf.keras.layers.Activation(
                activation=final_activation)(output_layer)

    # --- skip with input layer
    if add_skip_with_input:
        # TODO add mixer here
        # low noise performs better with skip input
        # high noise performs better with direct reconstruction
        output_layer = \
            tf.keras.layers.Add()([output_layer, y])

    output_layer = \
        tf.keras.layers.Layer(name="output_tensor")(output_layer)

    # return intermediate results if flag is turned on
    output_layers = [output_layer]
    if add_intermediate_results:
        output_layers.append(
            tf.keras.layers.Layer(name="intermediate_tensor")(x))

    return \
        tf.keras.Model(
            name=name,
            trainable=True,
            inputs=input_layer,
            outputs=output_layers)

# ---------------------------------------------------------------------
