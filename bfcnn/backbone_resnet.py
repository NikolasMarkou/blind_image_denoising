import copy
import tensorflow as tf

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .constants import *
from .custom_logger import logger
from .utilities import conv2d_wrapper, mean_sigma_local
from .custom_layers import ChannelwiseMultiplier, Multiplier
from .backbone_blocks import resnet_blocks_full, sparse_block


# ---------------------------------------------------------------------


def builder(
        input_dims,
        no_layers: int,
        kernel_size: int,
        filters: int,
        activation: str = "relu",
        base_activation: str = "linear",
        use_bn: bool = True,
        use_bias: bool = False,
        kernel_regularizer="l1",
        kernel_initializer="glorot_normal",
        dropout_rate: float = -1,
        stop_gradient: bool = False,
        add_clip: bool = False,
        add_gates: bool = False,
        add_selector: bool = False,
        add_sparsity: bool = False,
        add_final_bn: bool = False,
        add_initial_bn: bool = False,
        add_concat_input: bool = False,
        add_sparse_features: bool = False,
        add_channelwise_scaling: bool = False,
        add_learnable_multiplier: bool = False,
        add_mean_sigma_normalization: bool = False,
        name="resnet",
        **kwargs) -> keras.Model:
    """
    builds a resnet model

    :param input_dims: Models input dimensions
    :param no_layers: Number of resnet layers
    :param kernel_size: kernel size of the conv layers
    :param filters: number of filters per convolutional layer
    :param activation: activation of the convolutional layers
    :param base_activation: activation of the base layer,
        residual blocks outputs must conform to this
    :param dropout_rate: probability of resnet block shutting off
    :param use_bn: use batch normalization
    :param use_bias: use bias
    :param kernel_regularizer: Kernel weight regularizer
    :param kernel_initializer: Kernel weight initializer
    :param add_channelwise_scaling: if True for each full convolutional kernel add a scaling depthwise
    :param add_learnable_multiplier:
    :param stop_gradient: if True stop gradients in each resnet block
    :param add_sparsity: if true add sparsity layer
    :param add_gates: if true add gate layer
    :param add_mean_sigma_normalization: if true add variance for each block
    :param add_initial_bn: add a batch norm before the resnet blocks
    :param add_final_bn: add a batch norm after the resnet blocks
    :param add_concat_input: if true concat input to intermediate before projecting
    :param add_selector: if true add a selector block in skip connections
    :param add_clip: if True squash results with a tanh activation
    :param add_sparse_features: if true set feature map to be sparse
    :param name: name of the model

    :return: resnet model
    """
    # --- logging
    logger.info("building resnet backbone")
    logger.info(f"parameters not used: {kwargs}")

    # --- setup parameters
    bn_params = \
        dict(
            scale=True,
            center=use_bias,
            momentum=DEFAULT_BN_MOMENTUM,
            epsilon=DEFAULT_BN_EPSILON
        )

    base_conv_params = dict(
        filters=filters,
        strides=(1, 1),
        padding="same",
        use_bias=use_bias,
        activation=base_activation,
        kernel_size=kernel_size,
        kernel_regularizer=kernel_regularizer,
        kernel_initializer=kernel_initializer
    )

    first_conv_params = dict(
        kernel_size=3,
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
        filters=filters,
        strides=(1, 1),
        padding="same",
        use_bias=use_bias,
        activation=base_activation,
        kernel_regularizer=kernel_regularizer,
        kernel_initializer=kernel_initializer
    )

    resnet_params = dict(
        bn_params=None,
        sparse_params=None,
        no_layers=no_layers,
        selector_params=None,
        stop_gradient=stop_gradient,
        channelwise_params=None,
        first_conv_params=first_conv_params,
        second_conv_params=second_conv_params,
        third_conv_params=None,
    )

    channelwise_params = dict(
        multiplier=1.0,
        regularizer=keras.regularizers.L1(DEFAULT_CHANNELWISE_MULTIPLIER_L1),
        trainable=True,
        activation="relu"
    )

    multiplier_params = dict(
        multiplier=1.0,
        regularizer=keras.regularizers.L1(DEFAULT_CHANNELWISE_MULTIPLIER_L1),
        trainable=True,
        activation="relu"
    )

    if use_bn:
        resnet_params["bn_params"] = bn_params

    # make it linear so it gets sparse afterwards
    if add_sparsity:
        resnet_params["sparse_params"] = \
            dict(
                threshold_sigma=1.0,
            )

    if add_selector:
        resnet_params["selector_params"] = dict()

    if add_gates:
        resnet_params["gate_params"] = \
            dict(
                kernel_size=1,
                filters=filters,
                strides=(1, 1),
                padding="same",
                use_bias=use_bias,
                activation=activation,
                kernel_regularizer=kernel_regularizer,
                kernel_initializer=kernel_initializer
            )

    if add_mean_sigma_normalization:
        resnet_params["mean_sigma_params"] = \
            dict(
                pool_size=(11, 11)
            )

    if dropout_rate != -1:
        resnet_params["dropout_params"] = \
            dict(
                rate=dropout_rate
            )

    if add_channelwise_scaling:
        resnet_params["channelwise_params"] = \
            copy.deepcopy(channelwise_params)

    if add_learnable_multiplier:
        resnet_params["multiplier_params"] = \
            copy.deepcopy(multiplier_params)

    # --- build model
    # set input
    input_layer = \
        keras.Input(
            name="input_tensor",
            shape=input_dims)
    x = input_layer
    y = input_layer

    # add base layer
    x = \
        conv2d_wrapper(
            input_layer=x,
            bn_params=None,
            conv_params=base_conv_params,
            channelwise_scaling=None)

    if add_initial_bn:
        x = tf.keras.layers.BatchNormalization(**bn_params)(x)

    # add resnet blocks
    x = \
        resnet_blocks_full(
            input_layer=x,
            **resnet_params)

    # optional batch norm
    if add_final_bn:
        x = tf.keras.layers.BatchNormalization(**bn_params)(x)

    # optional concat and mix with input
    if add_concat_input:
        x = tf.keras.layers.Concatenate()([x, y])

    # optional sparsity, 80% per layer becomes zero
    if add_sparse_features:
        x = \
            sparse_block(
                input_layer=x,
                symmetrical=True,
                bn_params=None,
                threshold_sigma=1.0)

    # optional final channelwise multiplier
    if add_channelwise_scaling:
        x = ChannelwiseMultiplier(**channelwise_params)(x)

    # optional final multiplier
    if add_learnable_multiplier:
        x = Multiplier(**multiplier_params)(x)

    # optional clipping to [-1, +1]
    if add_clip:
        x = tf.tanh(x)

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
