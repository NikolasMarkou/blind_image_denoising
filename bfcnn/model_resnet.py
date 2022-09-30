import keras
import numpy as np
import tensorflow as tf
from typing import List, Tuple, Union, Dict, Iterable

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .constants import *
from .custom_logger import logger
from .custom_layers import Multiplier
from .model_blocks import resnet_blocks_full
from .utilities import conv2d_wrapper, mean_sigma_local

# ---------------------------------------------------------------------


def build_model_resnet(
        input_dims,
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
        channelwise_scaling: bool = False,
        add_skip_with_input: bool = True,
        add_sparsity: bool = False,
        add_gates: bool = False,
        add_var: bool = False,
        add_final_bn: bool = False,
        add_intermediate_results: bool = False,
        add_learnable_multiplier: bool = False,
        add_projection_to_input: bool = True,
        add_concat_input: bool = False,
        name="resnet",
        **kwargs) -> keras.Model:
    """
    builds a resnet model

    :param input_dims: Models input dimensions
    :param no_layers: Number of resnet layers
    :param kernel_size: kernel size of the conv layers
    :param filters: number of filters per convolutional layer
    :param activation: activation of the convolutional layers
    :param final_activation: activation of the final layer
    :param channel_index: Index of the channel in dimensions
    :param dropout_rate: probability of resnet block shutting off
    :param use_bn: Use Batch Normalization
    :param use_bias: use bias
    :param kernel_regularizer: Kernel weight regularizer
    :param kernel_initializer: Kernel weight initializer
    :param channelwise_scaling: if True for each full convolutional kernel add a scaling depthwise
    :param add_skip_with_input: if true skip with input
    :param add_sparsity: if true add sparsity layer
    :param add_gates: if true add gate layer
    :param add_var: if true add variance for each block
    :param add_final_bn: add a batch norm after the resnet blocks
    :param add_intermediate_results: if true output results before projection
    :param add_learnable_multiplier:
    :param add_projection_to_input: if true project to input tensor channel number
    :param add_concat_input: if true concat input to intermediate before projecting
    :param name: name of the model
    :return: resnet model
    """
    # --- logging
    logger.info("building resnet")
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
        depth_multiplier=2,
        strides=(1, 1),
        padding="same",
        use_bias=use_bias,
        activation=activation,
        depthwise_regularizer=kernel_regularizer,
        depthwise_initializer=kernel_initializer
    )

    third_conv_params = dict(
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
        kernel_regularizer="l2",
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

    channelwise_params = dict(
        multiplier=1.0,
        regularizer=keras.regularizers.L1(DEFAULT_CHANNELWISE_MULTIPLIER_L1),
        trainable=True,
        activation="relu"
    )

    resnet_params = dict(
        bn_params=None,
        sparse_params=None,
        no_layers=no_layers,
        channelwise_params=channelwise_params,
        first_conv_params=first_conv_params,
        second_conv_params=second_conv_params,
        third_conv_params=third_conv_params,
    )

    if use_bn:
        resnet_params["bn_params"] = bn_params

    # make it linear so it gets sparse afterwards
    if add_sparsity:
        resnet_params["sparse_params"] = sparse_params

    if add_gates:
        resnet_params["gate_params"] = gate_params

    if add_learnable_multiplier:
        resnet_params["multiplier_params"] = multiplier_params

    if dropout_rate != -1:
        resnet_params["dropout_params"] = dropout_params

    if channelwise_scaling:
        resnet_params["channelwise_params"] = channelwise_params

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
    if use_bn:
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
        y_tmp = y
        if use_bn:
            y_tmp = tf.keras.layers.BatchNormalization(**bn_params)(y_tmp)
        x = tf.keras.layers.Concatenate()([x, y_tmp])

    # --- output layer branches here,
    # cap it off to limit values
    output_layer = x

    # --- output to original channels / projection
    if add_projection_to_input:
        output_layer = \
            conv2d_wrapper(
                input_layer=output_layer,
                bn_params=None,
                conv_params=final_conv_params,
                channelwise_scaling=channelwise_params)

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


