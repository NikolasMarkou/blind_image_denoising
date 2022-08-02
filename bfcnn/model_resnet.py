import os
import json
import keras
import itertools
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import List, Tuple, Union, Dict, Iterable

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .custom_logger import logger
from .constants import EPSILON_DEFAULT
from .custom_layers import \
    Multiplier, \
    RandomOnOff, \
    ChannelwiseMultiplier
from .activations import differentiable_relu, differentiable_relu_layer
from .utilities import \
    sparse_block, \
    dense_wrapper, \
    conv2d_wrapper, \
    mean_sigma_local, \
    mean_sigma_global

# ---------------------------------------------------------------------


def resnet_blocks(
        input_layer,
        no_layers: int,
        first_conv_params: Dict,
        second_conv_params: Dict,
        third_conv_params: Dict,
        depthwise_scaling: bool = False,
        bn_params: Dict = None,
        gate_params: Dict = None,
        dropout_params: Dict = None,
        multiplier_params: Dict = None,
        **kwargs):
    """
    Create a series of residual network blocks

    :param input_layer: the input layer to perform on
    :param no_layers: how many residual network blocks to add
    :param first_conv_params: the parameters of the first conv
    :param second_conv_params: the parameters of the middle conv
    :param third_conv_params: the parameters of the third conv
    :param depthwise_scaling: if True add a learnable point-wise depthwise scaling conv2d
    :param bn_params: batch normalization parameters
    :param gate_params: gate optional parameters
    :param dropout_params: dropout optional parameters
    :param multiplier_params: learnable optional parameters

    :return: filtered input_layer
    """
    # --- argument check
    if input_layer is None:
        raise ValueError("input_layer must be none")
    if no_layers < 0:
        raise ValueError("no_layers must be >= 0")
    use_bn = bn_params is not None
    use_gate = gate_params is not None
    use_dropout = dropout_params is not None
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

    if use_gate:
        g = tf.keras.layers.GlobalAveragePooling2D()(input_layer)
        g = \
            dense_wrapper(
                input_layer=g,
                bn_params=bn_params,
                dense_params=dense_params,
                elementwise_params=elementwise_params)

    # --- create several number of residual blocks
    for i in range(no_layers):
        previous_layer = x
        x = conv2d_wrapper(input_layer=x,
                           conv_params=first_conv_params,
                           bn_params=bn_params,
                           depthwise_scaling=depthwise_scaling)
        x = conv2d_wrapper(input_layer=x,
                           conv_params=second_conv_params,
                           bn_params=bn_params,
                           depthwise_scaling=depthwise_scaling)
        x = conv2d_wrapper(input_layer=x,
                           conv_params=third_conv_params,
                           bn_params=bn_params,
                           depthwise_scaling=depthwise_scaling)
        # compute activation per channel
        if use_gate:
            y0 = tf.keras.layers.GlobalAveragePooling2D()(x)
            y0 = y0 * (1.0 - 0.9 * g)
            if use_bn:
                y0 = tf.keras.layers.BatchNormalization(**bn_params)(y0)
            y = \
                dense_wrapper(
                    input_layer=y0,
                    bn_params=None,
                    dense_params=dense_params,
                    elementwise_params=elementwise_params)
            g = g + y0
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
        depthwise_scaling: bool = False,
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
    :param activation: intermediate activation
    :param final_activation: activation of the final layer
    :param channel_index: Index of the channel in dimensions
    :param dropout_rate: probability of resnet block shutting off
    :param use_bn: Use Batch Normalization
    :param use_bias: use bias
    :param kernel_regularizer: Kernel weight regularizer
    :param kernel_initializer: Kernel weight initializer
    :param depthwise_scaling: if True for each full convolutional kernel add a scaling depthwise
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
        axis=None,
        center=use_bias,
        scale=True,
        momentum=0.999,
        epsilon=1e-4
    )

    # this make it 68% sparse
    sparse_params = dict(
        symmetric=True,
        max_value=3.0,
        threshold_sigma=1.0,
        per_channel_sparsity=False
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

    resnet_params = dict(
        bn_params=None,
        no_layers=no_layers,
        depthwise_scaling=depthwise_scaling,
        first_conv_params=first_conv_params,
        second_conv_params=second_conv_params,
        third_conv_params=third_conv_params,
    )

    if use_bn:
        resnet_params["bn_params"] = bn_params

    # make it linear so it gets sparse afterwards
    if add_sparsity:
        base_conv_params["activation"] = "linear"

    if add_gates:
        resnet_params["gate_params"] = gate_params

    if add_learnable_multiplier:
        resnet_params["multiplier_params"] = multiplier_params

    if dropout_rate != -1:
        resnet_params["dropout_params"] = dropout_params

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
            depthwise_scaling=depthwise_scaling)

    if add_sparsity:
        x = \
            sparse_block(
                input_layer=x,
                bn_params=None,
                **sparse_params)

    # add resnet blocks
    x = \
        resnet_blocks(
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
    # to allow space for intermediate results
    output_layer = x

    # --- output to original channels / projection
    if add_projection_to_input:
        output_layer = \
            conv2d_wrapper(
                input_layer=output_layer,
                bn_params=None,
                conv_params=final_conv_params,
                depthwise_scaling=depthwise_scaling)

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
