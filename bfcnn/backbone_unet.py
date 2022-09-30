import tensorflow as tf

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .constants import *
from .custom_logger import logger
from .utilities import conv2d_wrapper, mean_sigma_local
from .model_blocks import resnet_blocks_full, unet_blocks

# ---------------------------------------------------------------------


def builder(
        input_dims,
        no_levels: int,
        no_layers: int,
        kernel_size: int,
        filters: int,
        activation: str = "relu",
        use_bn: bool = True,
        use_bias: bool = False,
        kernel_regularizer="l1",
        kernel_initializer="glorot_normal",
        channel_index: int = 2,
        dropout_rate: float = -1,
        add_sparsity: bool = False,
        add_gates: bool = False,
        add_var: bool = False,
        add_initial_bn: bool = False,
        add_final_bn: bool = False,
        add_learnable_multiplier: bool = False,
        add_concat_input: bool = False,
        name="unet",
        **kwargs) -> keras.Model:
    """
    builds a u-net model

    :param input_dims: Models input dimensions
    :param no_levels: Number of unet layers
    :param no_layers: Number of resnet layers per unet level
    :param kernel_size: kernel size of the conv layers
    :param filters: number of filters per convolutional layer
    :param activation: intermediate activation
    :param channel_index: Index of the channel in dimensions
    :param dropout_rate: probability of resnet block shutting off
    :param use_bn: Use Batch Normalization
    :param use_bias: use bias
    :param kernel_regularizer: Kernel weight regularizer
    :param kernel_initializer: Kernel weight initializer
    :param add_sparsity: if true add sparsity layer
    :param add_gates: if true add gate layer
    :param add_var: if true add variance for each block
    :param add_final_bn: add a batch norm after the resnet blocks
    :param add_learnable_multiplier:
    :param add_concat_input: if true concat input to intermediate before projecting
    :param name: name of the model
    :return: unet model
    """
    # --- logging
    logger.info("building unet backbone")
    logger.info(f"parameters not used: {kwargs}")

    # --- setup parameters
    bn_params = dict(
        center=use_bias,
        scale=True,
        momentum=DEFAULT_BN_MOMENTUM,
        epsilon=DEFAULT_BN_EPSILON
    )

    # this make it 68% sparse
    sparse_params = dict(
        threshold_sigma=1.0,
    )

    base_conv_params = dict(
        filters=filters,
        strides=(1, 1),
        padding="same",
        use_bias=use_bias,
        activation="linear",
        kernel_size=kernel_size,
        kernel_regularizer=kernel_regularizer,
        kernel_initializer=kernel_initializer
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

    first_conv_params = dict(
        kernel_size=1,
        filters=filters,
        strides=(1, 1),
        padding="same",
        use_bias=use_bias,
        activation=activation,
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
        activation="linear",
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

    unet_params = dict(
        no_levels=no_levels,
        no_layers=no_layers,
        bn_params=bn_params,
        first_conv_params=first_conv_params,
        second_conv_params=second_conv_params,
        third_conv_params=third_conv_params,
    )

    # make it linear so it gets sparse afterwards
    if add_sparsity:
        base_conv_params["activation"] = "linear"

    if add_gates:
        unet_params["gate_params"] = gate_params

    if add_learnable_multiplier:
        unet_params["multiplier_params"] = multiplier_params

    if dropout_rate != -1:
        unet_params["dropout_params"] = dropout_params

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

    # add base layer
    x = \
        conv2d_wrapper(
            input_layer=x,
            bn_params=None,
            conv_params=base_conv_params,
            channelwise_scaling=None)
    if add_initial_bn:
        x = tf.keras.layers.BatchNormalization(**bn_params)(x)

    # add unet blocks
    x = \
        unet_blocks(
            input_layer=x,
            **unet_params)

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
    output_layer = \
        tf.keras.layers.Layer(name="intermediate_output")(x)

    return \
        tf.keras.Model(
            name=name,
            trainable=True,
            inputs=input_layer,
            outputs=output_layer)

# ---------------------------------------------------------------------
