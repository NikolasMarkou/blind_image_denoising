import copy
import tensorflow as tf
from typing import List

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .constants import *
from .custom_logger import logger
from .utilities import conv2d_wrapper
from .backbone_blocks import resnet_blocks_full
from .custom_layers import ChannelwiseMultiplier, Multiplier


# ---------------------------------------------------------------------


def builder(
        input_dims,
        no_layers: int,
        kernel_size: int,
        filters: int,
        block_kernels: List[int] = [3, 3],
        block_filters: List[int] = [32, 32],
        block_groups: List[int] = None,
        block_depthwise: List[int] = None,
        block_regularizer: List[str] = None,
        block_activation: List[str] = None,
        activation: str = "relu",
        base_activation: str = "linear",
        base_conv_params: Dict = None,
        use_bn: bool = True,
        use_bias: bool = False,
        kernel_regularizer="l1",
        kernel_initializer="glorot_normal",
        dropout_rate: float = -1,
        add_gelu: bool = False,
        add_gates: bool = False,
        add_final_bn: bool = False,
        add_initial_bn: bool = False,
        add_concat_input: bool = False,
        add_gradient_dropout: bool = False,
        add_channelwise_scaling: bool = False,
        add_learnable_multiplier: bool = False,
        add_mean_sigma_normalization: bool = False,
        selector_params: Dict = None,
        output_layer_name: str = "intermediate_output",
        name="resnet",
        **kwargs) -> keras.Model:
    """
    builds a resnet model

    :param input_dims: Models input dimensions
    :param no_layers: Number of resnet layers
    :param kernel_size: kernel size of base convolutional layer
    :param filters: filters of base convolutional layer
    :param block_kernels: kernel size of per res-block convolutional layer
    :param block_filters: filters per res-block convolutional layer
    :param block_groups: groups to use pe res-block
    :param block_depthwise:
        depthwise multipliers per block, leave empty or full of -1 to disable
    :param block_regularizer: regularizer for each block
    :param block_activation: activation for each block
    :param activation: activation of the convolutional layers
    :param base_activation: activation of the base layer,
        residual blocks outputs must conform to this
    :param base_conv_params: base convolution parameters to override defaults
    :param dropout_rate: probability of resnet block shutting off
    :param use_bn: use batch normalization
    :param use_bias: use bias (bias free means this should be off
    :param kernel_regularizer: Kernel weight regularizer
    :param kernel_initializer: Kernel weight initializer
    :param add_channelwise_scaling: if True for each full convolutional kernel add a scaling depthwise
    :param add_learnable_multiplier: if True add a learnable multiplier
    :param add_gates: if true add gate layer
    :param add_gelu: if true add gelu layers
    :param add_mean_sigma_normalization: if true add variance for each block
    :param add_initial_bn: add a batch norm before the resnet blocks
    :param add_final_bn: add a batch norm after the resnet blocks
    :param add_concat_input: if true concat input to intermediate before projecting
    :param add_gradient_dropout: if True add a gradient dropout layer
    :param selector_params:
    :param output_layer_name: the output layer name
    :param name: name of the model

    :return: resnet model
    """
    # --- logging
    logger.info("building resnet backbone")
    logger.info(f"parameters not used: {kwargs}")

    # --- argument fixing
    if block_depthwise is None or \
            len(block_depthwise) == 0:
        block_depthwise = [-1] * len(block_kernels)

    if block_groups is None or \
            len(block_groups) == 0:
        block_groups = [1] * len(block_kernels)

    if block_regularizer is None or \
            len(block_regularizer) == 0:
        block_regularizer = [kernel_regularizer] * len(block_kernels)

    if block_activation is None or len(block_activation) == 0:
        block_activation = [activation] * len(block_kernels)

    # --- argument checking
    if len(block_kernels) <= 0:
        raise ValueError("len(block_kernels) must be >= 0 ")
    if len(block_kernels) > 3:
        raise ValueError("len(block_kernels) must be <= 3")
    if len(block_filters) <= 0:
        raise ValueError("len(block_filters) must be >= 0 ")
    if len(block_kernels) != len(block_filters):
        raise ValueError("len(block_filters) must == len(block_kernels)")
    if len(block_kernels) != len(block_groups):
        raise ValueError("len(block_filters) must == len(block_groups)")
    if len(block_regularizer) != len(block_groups):
        raise ValueError("len(block_regularizer) must == len(block_groups)")
    if len(block_activation) != len(block_groups):
        raise ValueError("len(block_activation) must == len(block_groups)")
    if block_depthwise is not None and \
            (len(block_depthwise) != len(block_kernels)):
        raise ValueError("len(block_depthwise) must == len(block_kernels)")

    # --- setup parameters
    bn_params = \
        dict(
            scale=True,
            center=use_bias,
            momentum=DEFAULT_BN_MOMENTUM,
            epsilon=DEFAULT_BN_EPSILON
        )

    if base_conv_params is None:
        base_conv_params = dict(
            kernel_size=kernel_size,
            filters=filters,
            strides=(1, 1),
            padding="same",
            use_bias=use_bias,
            activation=base_activation,
            kernel_regularizer=kernel_regularizer,
            kernel_initializer=kernel_initializer
        )

    convs_params = [None] * 3
    no_blocks = len(block_kernels)
    for i in range(no_blocks):
        if block_depthwise[i] == -1:
            # normal Conv2d configuration
            convs_params[i] = dict(
                kernel_size=block_kernels[i],
                filters=block_filters[i],
                strides=(1, 1),
                padding="same",
                use_bias=use_bias,
                activation=block_activation[i],
                groups=block_groups[i],
                kernel_regularizer=block_regularizer[i],
                kernel_initializer=kernel_initializer,
            )
        else:
            # DepthwiseConv2D configuration
            convs_params[i] = dict(
                kernel_size=block_kernels[i],
                depth_multiplier=block_depthwise[i],
                strides=(1, 1),
                padding="same",
                use_bias=use_bias,
                activation=block_activation[i],
                depthwise_regularizer=block_regularizer[i],
                depthwise_initializer=kernel_initializer,
            )
    # set the final activation to be the same as the base activation
    convs_params[no_blocks-1]["activation"] = base_activation

    resnet_params = dict(
        bn_params=None,
        sparse_params=None,
        no_layers=no_layers,
        selector_params=selector_params,
        multiplier_params=None,
        channelwise_params=None,
        first_conv_params=convs_params[0],
        second_conv_params=convs_params[1],
        third_conv_params=convs_params[2],
    )

    channelwise_params = dict(
        multiplier=1.0,
        regularizer=keras.regularizers.L1(DEFAULT_CHANNELWISE_MULTIPLIER_L1),
        trainable=True,
        activation="relu"
    )

    multiplier_params = dict(
        multiplier=1.0,
        regularizer=keras.regularizers.L1(DEFAULT_MULTIPLIER_L1),
        trainable=True,
        activation="relu"
    )

    if use_bn:
        resnet_params["bn_params"] = bn_params

    if add_gelu:
        resnet_params["gelu_params"] = dict()

    if add_gradient_dropout:
        resnet_params["gradient_dropout_params"] = dict()

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
            conv_params=base_conv_params)

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

    # optional final channelwise multiplier
    # if add_channelwise_scaling:
    #     x = ChannelwiseMultiplier(**channelwise_params)(x)

    # optional final multiplier
    # if add_learnable_multiplier:
    #     x = Multiplier(**multiplier_params)(x)

    # --- output layer branches here,
    output_layer = \
        tf.keras.layers.Layer(name=output_layer_name)(x)

    return \
        tf.keras.Model(
            name=name,
            trainable=True,
            inputs=input_layer,
            outputs=output_layer)

# ---------------------------------------------------------------------
