"""
unet with laplacian down sampling and convnext blocks
"""

import copy

import numpy as np
import tensorflow as tf
from typing import List, Dict, Union, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .constants import *
from .custom_logger import logger
from .utilities import (
    conv2d_wrapper,
    activation_wrapper)
from .upsampling import upsample
from .downsampling import downsample
from .custom_layers import (
    ConvNextBlock,
    GaussianFilter,
    StochasticDepth,
    AdditiveAttentionGate,
    ChannelLearnableMultiplier,
    ConvolutionalSelfAttention)
from .regularizers import (
    SoftOrthogonalConstraintRegularizer,
    SoftOrthonormalConstraintRegularizer)

# ---------------------------------------------------------------------


def builder(
        input_dims,
        depth: int = 5,
        width: int = 1,
        gaussian_kernel_size: int = 3,
        encoder_kernel_size: int = 5,
        decoder_kernel_size: int = 3,
        filters: int = 32,
        max_filters: int = -1,
        filters_level_multiplier: float = 2.0,
        activation: str = "leaky_relu_01",
        upsample_type: str = "bilinear",
        downsample_type: str = "strides",
        use_bn: bool = False,
        use_ln: bool = True,
        use_gamma: bool = True,
        use_bias: bool = False,
        use_concat: bool = True,
        use_laplacian: bool = True,
        use_laplacian_averaging: bool = True,
        use_mix_project: bool = True,
        use_self_attention: bool = False,
        use_attention_gates: bool = False,
        use_complex_base: bool = False,
        use_global_pool_information: bool = False,
        use_soft_orthogonal_regularization: bool = False,
        use_soft_orthonormal_regularization: bool = False,
        kernel_regularizer="l2",
        kernel_initializer="glorot_normal",
        dropout_rate: float = -1,
        depth_drop_rate: float = 0.0,
        spatial_dropout_rate: float = -1,
        convolutional_self_attention_dropout_rate: float = 0.0,
        multiple_scale_outputs: bool = True,
        use_output_normalization: bool = False,
        output_layer_name: str = "intermediate_output",
        name="unet_laplacian",
        **kwargs) -> tf.keras.Model:
    """
    builds a modified unet model that uses convnext blocks and laplacian downsampling

    :param input_dims: Models input dimensions
    :param depth: number of levels to go down
    :param width: number of horizontals nodes, if -1 it gets set to depth
    :param gaussian_kernel_size: laplacian gaussian kernel size
    :param encoder_kernel_size: kernel size of encoder convolutional layer
    :param decoder_kernel_size: kernel size of decoder convolutional layer
    :param filters_level_multiplier: every down level increase the number of filters by a factor of
    :param filters: filters of base convolutional layer
    :param max_filters: max number of filters
    :param activation: activation of the first 1x1 kernel
    :param upsample_type: string describing the upsample type
    :param downsample_type: string describing the downsample type
    :param use_bn: use batch normalization
    :param use_ln: use layer normalization
    :param use_gamma: if True (True by default) use gamma learning in convnext
    :param use_soft_gamma: if True (False by default) use soft gamma learning in convnext
    :param use_bias: use bias (bias free means this should be off)
    :param use_attention_gates: if True add attention gates between depths
    :param use_complex_base: if True add two convolutions at the base
    :param use_mix_project: if True mix different depths with a 1x1 projection (SKOOTS: Skeleton oriented object segmentation for mitochondria)
    :param use_self_attention: if True add a convolutional self-attention element at the bottom layer
    :param use_concat: if True concatenate otherwise add skip layers (True by default)
    :param use_laplacian: if True use laplacian estimation between depths
    :param use_soft_orthogonal_regularization: if True use soft orthogonal regularization on the 1x1 kernels
    :param use_soft_orthonormal_regularization: if true use soft orthonormal regularization on the 1x1 middle kernels
    :param kernel_regularizer: Kernel weight regularizer
    :param kernel_initializer: Kernel weight initializer
    :param dropout_rate: probability of dropout, negative to turn off
    :param spatial_dropout_rate: probability of spatial dropout, negative to turn off
    :param depth_drop_rate: probability of residual block dropout, negative or zero to turn off
    :param multiple_scale_outputs: if True for each scale give an output
    :param use_output_normalization: if True apply normalization to each output
    :param output_layer_name: the output layer's name
    :param name: name of the model

    :return: unet with convnext blocks model
    """
    # --- argument checking
    logger.info("building unet_laplacian backbone")
    if len(kwargs) > 0:
        logger.info(f"parameters not used: {kwargs}")

    if width is None or width <= 0:
        width = 1

    if depth <= 0 or width <= 0:
        raise ValueError("depth and width must be > 0")

    if convolutional_self_attention_dropout_rate < 0 or convolutional_self_attention_dropout_rate > 1:
        raise ValueError("convolutional_self_attention_dropout_rate must be >= 0 and <= 1")

    def activation_str_fix_fn(activation_str: str = None) -> str:
        if activation_str is None:
            activation_str = ""
        activation_str = activation_str.strip().lower()
        if len(activation_str) <= 0:
            activation_str = activation
        return activation_str

    downsample_activation = activation_str_fix_fn(activation)
    upsample_activation = activation_str_fix_fn(activation)
    upsample_type = upsample_type.strip().lower()
    downsample_type = downsample_type.strip().lower()
    kernel_initializer = kernel_initializer.strip().lower()

    if (use_soft_orthonormal_regularization and
            use_soft_orthogonal_regularization):
        raise ValueError(
            "only one use_soft_orthonormal_regularization or "
            "use_soft_orthogonal_regularization must be turned on")

    # --- setup parameters
    bn_params = None
    if use_bn:
        bn_params = \
            dict(
                scale=True,
                center=use_bias,
                momentum=DEFAULT_BN_MOMENTUM,
                epsilon=DEFAULT_BN_EPSILON
            )

    ln_params = None
    if use_ln:
        ln_params = \
            dict(
                scale=True,
                center=use_bias,
                epsilon=DEFAULT_LN_EPSILON
            )

    dropout_params = None
    if dropout_rate > 0.0:
        dropout_params = {"rate": dropout_rate}

    dropout_2d_params = None
    if spatial_dropout_rate > 0.0:
        dropout_2d_params = {"rate": spatial_dropout_rate}

    depth_drop_rates = (
        list(np.linspace(0.0, max(0.0, depth_drop_rate), width)))

    base_conv_params = dict(
        kernel_size=encoder_kernel_size,
        filters=filters,
        strides=(1, 1),
        padding="same",
        use_bias=use_bias,
        activation=activation,
        kernel_regularizer=kernel_regularizer,
        kernel_initializer=kernel_initializer
    )

    conv_params = []
    conv_params_up = []
    conv_params_down = []
    conv_params_res_1 = []
    conv_params_res_2 = []
    conv_params_res_3 = []

    for d in range(depth):
        filters_level = \
            int(round(filters * max(1, filters_level_multiplier ** d)))
        if max_filters > 0:
            filters_level = min(max_filters, filters_level)
        filters_level_next = \
            int(round(filters * max(1, filters_level_multiplier ** (d + 1))))
        if max_filters > 0:
            filters_level_next = min(max_filters, filters_level_next)

        # default conv
        params = copy.deepcopy(base_conv_params)
        params["filters"] = filters_level
        params["kernel_size"] = 3
        params["activation"] = "linear"
        conv_params.append(params)

        # 1st residual conv
        conv_params_res_1.append(dict(
            kernel_size=encoder_kernel_size,
            depth_multiplier=1,
            strides=(1, 1),
            padding="same",
            use_bias=use_bias,
            activation="linear",
            depthwise_regularizer=kernel_regularizer,
            depthwise_initializer=kernel_initializer
        ))

        # 2nd residual conv
        params = copy.deepcopy(base_conv_params)
        params["kernel_size"] = 1
        params["activation"] = activation
        params["filters"] = filters_level * 4
        conv_params_res_2.append(params)

        # 3rd residual conv
        params = copy.deepcopy(base_conv_params)
        params["kernel_size"] = 1
        params["activation"] = "linear"
        params["filters"] = filters_level
        conv_params_res_3.append(params)

        # conv2d params when moving down the scale
        params = copy.deepcopy(base_conv_params)
        params["filters"] = filters_level_next
        params["activation"] = downsample_activation
        conv_params_down.append(params)

        # conv2d params when moving up the scale
        params = copy.deepcopy(base_conv_params)
        params["filters"] = filters_level
        params["activation"] = upsample_activation
        conv_params_up.append(params)

    # --- book keeping
    nodes_dependencies = {}
    for d in range(0, depth, 1):
        if d == (depth - 1):
            # add only left dependency
            nodes_dependencies[(d, 1)] = [(d, 0)]
        else:
            # add left and bottom dependency
            nodes_dependencies[(d, 1)] = [(d, 0), (d + 1, 1)]

    nodes_output = {}
    nodes_to_visit = list(nodes_dependencies.keys())
    nodes_visited = set([(depth - 1, 0), (depth - 1, 1)])

    # --- build model
    # set input
    input_layer = \
        tf.keras.Input(
            name=INPUT_TENSOR_STR,
            shape=input_dims)
    x = input_layer

    if use_complex_base:
        # first plain conv
        params = copy.deepcopy(base_conv_params)
        params["filters"] = max(filters, 96)
        params["kernel_size"] = (5, 5)
        params["strides"] = (1, 1)
        params["activation"] = "linear"

        x = \
            conv2d_wrapper(
                input_layer=x,
                ln_params=None,
                bn_params=None,
                conv_params=params)

        # second plain conv
        params = copy.deepcopy(base_conv_params)
        params["filters"] = filters
        params["kernel_size"] = (1, 1)
        params["strides"] = (1, 1)
        params["activation"] = activation

        x = \
            conv2d_wrapper(
                input_layer=x,
                ln_params=None,
                bn_params=None,
                conv_params=params)
    else:
        # first plain conv
        params = copy.deepcopy(base_conv_params)
        params["filters"] = filters
        params["kernel_size"] = (5, 5)
        params["strides"] = (1, 1)
        params["activation"] = activation

        x = \
            conv2d_wrapper(
                input_layer=x,
                ln_params=None,
                bn_params=None,
                conv_params=params)

    # --- build backbone
    for d in range(depth):
        for w in range(width):
            # get skip for residual
            x_skip = x

            if use_self_attention and d == depth-1:
                x = (
                    ConvolutionalSelfAttention(
                        use_gamma=True,
                        bn_params=bn_params,
                        ln_params=ln_params,
                        attention_channels=filters,
                        output_activation="linear",
                        use_soft_orthonormal_regularization=True,
                        dropout=convolutional_self_attention_dropout_rate,
                    )(x))
            else:
                x = \
                    ConvNextBlock(
                        name=f"encoder_{d}_{w}",
                        conv_params_1=conv_params_res_1[d],
                        conv_params_2=conv_params_res_2[d],
                        conv_params_3=conv_params_res_3[d],
                        ln_params=ln_params,
                        bn_params=bn_params,
                        dropout_params=dropout_params,
                        use_gamma=use_gamma,
                        dropout_2d_params=dropout_2d_params,
                        use_soft_orthogonal_regularization=use_soft_orthogonal_regularization,
                        use_soft_orthonormal_regularization=use_soft_orthonormal_regularization)(x)

            if x_skip.shape[-1] == x.shape[-1]:
                if len(depth_drop_rates) <= width and depth_drop_rates[w] > 0.0:
                    x = StochasticDepth(depth_drop_rates[w])(x)
                x = tf.keras.layers.Add()([x_skip, x])

        if use_output_normalization:
            if use_bn:
                x = tf.keras.layers.BatchNormalization(center=use_bias)(x)
            if use_ln:
                x = tf.keras.layers.LayerNormalization(center=use_bias)(x)

        node_level = (d, 0)
        nodes_visited.add(node_level)
        nodes_output[node_level] = x

        if d != (depth - 1):
            if use_laplacian or use_laplacian_averaging:
                if use_laplacian_averaging:
                    x_tmp_smooth = (
                        tf.keras.layers.AveragePooling2D(
                            padding="same",
                            strides=(1, 1),
                            pool_size=(gaussian_kernel_size, gaussian_kernel_size))(x)
                    )
                else:
                    x_tmp_smooth = \
                        GaussianFilter(
                            kernel_size=(gaussian_kernel_size, gaussian_kernel_size),
                            strides=(1, 1))(x)
                nodes_output[node_level] = \
                    tf.keras.layers.Subtract()([x, x_tmp_smooth])
                x = x_tmp_smooth

            x = (
                downsample(input_layer=x,
                           downsample_type=downsample_type,
                           ln_params=None,
                           bn_params=None,
                           conv_params=conv_params_down[d]))

    # --- add global information gathering
    if use_global_pool_information:
        params = copy.deepcopy(conv_params_res_3[depth - 1])
        params["kernel_size"] = (1, 1)
        params["activation"] = activation
        if use_soft_orthogonal_regularization:
            params["kernel_regularizer"] = \
                SoftOrthogonalConstraintRegularizer(
                    lambda_coefficient=DEFAULT_SOFTORTHOGONAL_LAMBDA,
                    l1_coefficient=DEFAULT_SOFTORTHOGONAL_L1,
                    l2_coefficient=DEFAULT_SOFTORTHOGONAL_L2)
        if use_soft_orthonormal_regularization:
            params["kernel_regularizer"] = \
                SoftOrthonormalConstraintRegularizer(
                    lambda_coefficient=DEFAULT_SOFTORTHONORMAL_LAMBDA,
                    l1_coefficient=DEFAULT_SOFTORTHONORMAL_L1,
                    l2_coefficient=DEFAULT_SOFTORTHONORMAL_L2)

        x_bottom = nodes_output[(depth - 1, 0)]

        x_bottom_pool = (
            tf.keras.layers.Concatenate(axis=-1)([
                tf.keras.layers.GlobalMaxPool2D(keepdims=True)(x_bottom),
                tf.keras.layers.GlobalAveragePooling2D(keepdims=True)(x_bottom)
        ]))

        # apply to all depths above
        for d in range(depth-1):
            params = copy.deepcopy(conv_params_res_3[d])
            params["kernel_size"] = (1, 1)
            params["activation"] = "sigmoid"
            params["name"] = f"gpi_{d}"

            node_level = (d, 0)
            x_bottom_pool_tmp = (
                conv2d_wrapper(
                    input_layer=x_bottom_pool,
                    ln_params=None,
                    bn_params=None,
                    conv_params=params))

            nodes_output[node_level] = (
                tf.keras.layers.Multiply()([
                    nodes_output[node_level],
                    x_bottom_pool_tmp])
            )
        del x_bottom_pool, x_bottom

    # --- VERY IMPORTANT
    # add this, so it works correctly
    nodes_output[(depth - 1, 1)] = nodes_output[(depth - 1, 0)]

    # --- build the encoder side based on dependencies
    while len(nodes_to_visit) > 0:
        node = nodes_to_visit.pop(0)
        logger.info(f"node: [{node}, "
                    f"nodes_visited: {nodes_visited}, "
                    f"nodes_to_visit: {nodes_to_visit}, "
                    f"dependencies: {nodes_dependencies[node]}")
        # make sure a node is not visited twice
        if node in nodes_visited:
            logger.info(f"node: [{node}] already processed")
            continue
        # make sure that all the dependencies for a node are matched
        dependencies = nodes_dependencies[node]
        dependencies_matched = \
            all([
                (d in nodes_output) and (d in nodes_visited or d == node)
                for d in dependencies
            ])
        if not dependencies_matched:
            logger.info(f"node: [{node}] dependencies not matched, continuing")
            nodes_to_visit.append(node)
            continue

        # sort it so all same level dependencies are first and added
        # as residual before finally concatenating the previous scale
        dependencies = \
            sorted(list(dependencies),
                   key=lambda d: d[0],
                   reverse=False)
        logger.info(f"processing node: {node}, "
                    f"dependencies: {dependencies}, "
                    f"nodes_output: {list(nodes_output.keys())}")

        x_input = []

        logger.debug(f"node: [{node}], dependencies: {dependencies}")
        for dependency in dependencies:
            logger.debug(f"processing dependency: {dependency}")
            x = nodes_output[dependency]

            if dependency[0] == node[0]:
                pass
            elif dependency[0] > node[0]:
                # based on normalization before upsample
                # SKOOTS: Skeleton oriented object segmentation for mitochondria, 2023
                logger.info("upsampling here")
                x = \
                    upsample(
                        input_layer=x,
                        upsample_type=upsample_type,
                        ln_params=None,
                        bn_params=None,
                        conv_params=conv_params_up[node[0]])
            else:
                raise ValueError(f"node: {node}, dependencies: {dependencies}, "
                                 f"should not supposed to be here")

            x_input.append(x)

        # add attention gates,
        # first input is assumed to be the higher depth
        # and the second input is assumed to be the lower depth
        if use_attention_gates and len(x_input) == 2:
            logger.debug(f"adding AttentionGate at depth: [{node[0]}]")

            x_input[0] = (
                AdditiveAttentionGate(
                    use_bias=use_bias,
                    use_bn=use_bn,
                    use_ln=use_ln,
                    use_soft_orthogonal_regularization=use_soft_orthogonal_regularization,
                    use_soft_orthonormal_regularization=use_soft_orthonormal_regularization,
                    attention_channels=conv_params_res_3[node[0]]["filters"],
                    kernel_initializer=kernel_initializer
                )(x_input))

        if len(x_input) == 1:
            x = x_input[0]
        elif len(x_input) > 0:
            if use_concat:
                x = tf.keras.layers.Concatenate()(x_input)
            else:
                x = tf.keras.layers.Add()(x_input)

            # project the concatenated result using a convolution
            if use_mix_project:
                # https://www.researchgate.net/figure/UNet-Architecture-with-ConvNext-computational-blocks-offers-superior-accuracy-per_fig2_370621145
                params = copy.deepcopy(conv_params_res_3[node[0]])
                params["kernel_size"] = (1, 1)
                params["activation"] = activation
                x = conv2d_wrapper(
                    input_layer=x,
                    ln_params=None,
                    bn_params=None,
                    conv_params=params)
        else:
            raise ValueError("this must never happen")

        # --- convnext block
        for w in range(width):
            d = node[0]
            x_skip = x

            params = copy.deepcopy(conv_params_res_1[d])
            params["kernel_size"] = (decoder_kernel_size, decoder_kernel_size)
            x = \
                ConvNextBlock(
                    name=f"decoder_{node[0]}_{w}",
                    conv_params_1=params,
                    conv_params_2=conv_params_res_2[node[0]],
                    conv_params_3=conv_params_res_3[node[0]],
                    ln_params=ln_params,
                    bn_params=bn_params,
                    use_gamma=use_gamma,
                    dropout_params=dropout_params,
                    dropout_2d_params=dropout_2d_params,
                    use_soft_orthogonal_regularization=use_soft_orthogonal_regularization,
                    use_soft_orthonormal_regularization=use_soft_orthonormal_regularization)(x)

            if x_skip.shape[-1] == x.shape[-1]:
                if len(depth_drop_rates) <= width and depth_drop_rates[w] > 0.0:
                    x = StochasticDepth(depth_drop_rates[w])(x)
                x = tf.keras.layers.Add()([x_skip, x])
        if use_output_normalization:
            if use_bn:
                x = tf.keras.layers.BatchNormalization(center=use_bias)(x)
            if use_ln:
                x = tf.keras.layers.LayerNormalization(center=use_bias)(x)
        nodes_output[node] = x
        nodes_visited.add(node)

    # --- output layer here
    output_layers = []

    # depth outputs
    if multiple_scale_outputs:
        tmp_output_layers = []
        for d in range(1, depth, 1):
            d = d
            w = 1

            if d < 0 or w < 0:
                logger.error(f"there is no node[{d},{w}] please check your assumptions")
                continue
            x = nodes_output[(d, w)]
            tmp_output_layers.append(x)
        # reverse here so deeper levels come on top
        output_layers += tmp_output_layers[::-1]

    # add as last the best output
    output_layers += [
        nodes_output[(0, 1)]
    ]

    # !!! IMPORTANT !!!
    # reverse it so the deepest output is first
    # otherwise we will get the most shallow output
    output_layers = output_layers[::-1]

    for i in range(len(output_layers)):
        x = output_layers[i]
        output_layers[i] = (
            tf.keras.layers.Layer(
                name=f"{output_layer_name}_{i}")(x))

    return \
        tf.keras.Model(
            name=name,
            trainable=True,
            inputs=[input_layer],
            outputs=output_layers)

# ---------------------------------------------------------------------
