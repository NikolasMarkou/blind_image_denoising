"""
unet+++ backbone
"""

import copy
import tensorflow as tf
from tensorflow import keras
from typing import List, Dict, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .constants import *
from .custom_logger import logger
from .custom_layers import RandomOnOff
from .utilities import ConvType, conv2d_wrapper
from .backbone_blocks import self_attention_block
from .regularizers import SoftOrthogonalConstraintRegularizer

# ---------------------------------------------------------------------


def builder(
        input_dims,
        depth: int = 5,
        width: int = -1,
        backbone_kernel_size: int = 7,
        kernel_size: int = -1,
        filters: int = 32,
        filters_level_multiplier: float = 2.0,
        activation: str = "gelu",
        second_activation: str = None,
        upsample_type: str = "conv2d_transpose",
        downsample_type: str = "maxpool",
        use_bn: bool = True,
        use_ln: bool = False,
        use_bias: bool = False,
        use_noise_regularization: bool = False,
        use_dropout_long_skips: bool = False,
        use_attention_block: bool = False,
        use_soft_orthogonal_regularization: bool = False,
        kernel_regularizer="l2",
        kernel_initializer="glorot_normal",
        dropout_rate: float = -1,
        spatial_dropout_rate: float = -1,
        multiple_scale_outputs: bool = False,
        output_layer_name: str = "intermediate_output",
        name="unet_ppp",
        **kwargs) -> keras.Model:
    """
    builds a modified unet++ model that uses convnext blocks

    :param input_dims: Models input dimensions
    :param depth: number of levels to go down
    :param width: number of horizontals nodes, if -1 it gets set to depth
    :param kernel_size: kernel size of the rest of convolutional layers
    :param backbone_kernel_size: kernel size of backbone convolutional layer
    :param filters_level_multiplier: every down level increase the number of filters by a factor of
    :param filters: filters of base convolutional layer
    :param activation: activation of the first 1x1 kernel
    :param second_activation: activation of the second 1x1 kernel
    :param upsample_type:
    :param downsample_type:
    :param dropout_rate: probability of dropout, negative to turn off
    :param spatial_dropout_rate: probability of dropout, negative to turn off
    :param use_bn: use batch normalization
    :param use_ln: use layer normalization
    :param use_bias: use bias (bias free means this should be off)
    :param use_noise_regularization: if true add a gaussian noise layer on each scale
    :param use_dropout_long_skips: if True on long skips use random on-off
    :param use_attention_block: if True add a self attention block at the deepest level
    :param use_soft_orthogonal_regularization: if True use soft orthogonal regularization on the 1x1 kernels
    :param kernel_regularizer: Kernel weight regularizer
    :param kernel_initializer: Kernel weight initializer
    :param multiple_scale_outputs:
    :param output_layer_name: the output layer name
    :param name: name of the model

    :return: resnet model
    """
    # --- argument checking
    logger.info("building unet_pp backbone")
    if len(kwargs) > 0:
        logger.info(f"parameters not used: {kwargs}")

    if width is None or width <= 0:
        width = depth

    if depth <= 0 or width <= 0:
        raise ValueError("depth must be > 0")

    if kernel_size is None or kernel_size <= 0:
        kernel_size = backbone_kernel_size

    if kernel_size <= 0 or backbone_kernel_size <= 0:
        raise ValueError(
            f"kernel_size: [{kernel_size}] and "
            f"backbone_kernel_size: [{backbone_kernel_size}] must be > 0")

    if second_activation is None:
        second_activation = ""
    second_activation = second_activation.strip().lower()
    if len(second_activation) <= 0:
        second_activation = activation

    upsample_type = upsample_type.strip().lower()
    downsample_type = downsample_type.strip().lower()

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

    base_conv_params = dict(
        kernel_size=backbone_kernel_size,
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
    conv_params_res_1 = []
    conv_params_res_2 = []
    conv_params_res_3 = []

    for i in range(depth):
        filters_level = \
            int(round(filters * max(1, filters_level_multiplier ** i)))

        # conv2d params when moving horizontally the scale
        params = copy.deepcopy(base_conv_params)
        params["filters"] = filters_level
        conv_params.append(params)

        # 1st residual conv
        params = \
            dict(
                kernel_size=backbone_kernel_size,
                depth_multiplier=1,
                strides=(1, 1),
                padding="same",
                use_bias=use_bias,
                activation="linear",
                depthwise_regularizer=kernel_regularizer,
                depthwise_initializer=kernel_initializer
            )
        conv_params_res_1.append(params)

        # 2nd residual conv
        params = copy.deepcopy(base_conv_params)
        params["kernel_size"] = 1
        params["activation"] = activation
        params["filters"] = filters_level * 4
        if use_soft_orthogonal_regularization:
            logger.info("added SoftOrthogonalConstraintRegularizer")
            params["kernel_regularizer"] = \
                SoftOrthogonalConstraintRegularizer(l1_coefficient=0.0001, l2_coefficient=0.0)
        conv_params_res_2.append(params)

        # 3rd residual conv
        params = copy.deepcopy(base_conv_params)
        params["kernel_size"] = 1
        params["activation"] = second_activation
        params["filters"] = filters_level
        if use_soft_orthogonal_regularization:
            logger.info("added SoftOrthogonalConstraintRegularizer")
            params["kernel_regularizer"] = \
                SoftOrthogonalConstraintRegularizer(l1_coefficient=0.0001, l2_coefficient=0.0)
        conv_params_res_3.append(params)

        # conv2d params when moving up the scale
        params = copy.deepcopy(base_conv_params)
        params["filters"] = filters_level
        params["kernel_size"] = (2, 2)
        params["strides"] = (2, 2)
        params["activation"] = conv_params_res_3[-1]["activation"]
        conv_params_up.append(params)

    attention_conv_params = copy.deepcopy(conv_params_res_3[-1])
    attention_conv_params["kernel_size"] = (1, 1)

    # --- book keeping
    nodes_dependencies = {}
    for j in range(0, depth + 1, 1):
        for i in range(1, width - j, 1):
            logger.debug(f"node {j, i} requires {j, i - 1} and {j + 1, i - 1}")
            # bottom and left dependencies
            if j < (depth - 1):
                nodes_dependencies[(j, i)] = [(j, i - 1), (j + 1, i - 1)]
            else:
                nodes_dependencies[(j, i)] = [(j, i - 1)]
            # NOTE only 2 skip connections per node make it scalable otherwise it gets very big
            # original design had connections with all horizontal
            if i >= 2:
                nodes_dependencies[(j, i)].append((j, i - 2))
            nodes_dependencies[(j, i)] = set(nodes_dependencies[(j, i)])
    nodes_output = {}
    nodes_to_visit = list(nodes_dependencies.keys())
    nodes_visited = set((depth-1, 0))

    # --- build model
    # set input
    input_layer = \
        keras.Input(
            name="input_tensor",
            shape=input_dims)
    x = input_layer

    # all the down sampling, backbone
    for i in range(depth):
        if i == 0:
            params = copy.deepcopy(base_conv_params)
            params["filters"] = max(32, filters)
            x = \
                conv2d_wrapper(
                    input_layer=x,
                    bn_post_params=bn_params,
                    ln_post_params=ln_params,
                    conv_params=params)
        else:
            params = copy.deepcopy(conv_params_res_1[i])
            if downsample_type == "maxpool":
                x = \
                    tf.keras.layers.MaxPooling2D(
                        pool_size=(2, 2), padding="same", strides=(2, 2))(x)
            elif downsample_type == "maxpool_3x3":
                x = \
                    tf.keras.layers.MaxPooling2D(
                        pool_size=(3, 3), padding="same", strides=(2, 2))(x)
            elif downsample_type in ["avg_3x3", "average_3x3", "avgpool_3x3"]:
                x = \
                    tf.keras.layers.AveragePooling2D(
                        pool_size=(3, 3), padding="same", strides=(2, 2))(x)
            elif downsample_type == "strides":
                params["strides"] = (2, 2)
            else:
                raise ValueError(
                    f"don't know how to handle downsample_type: [{downsample_type}]")
            x = \
                conv2d_wrapper(
                    input_layer=x,
                    bn_post_params=bn_params,
                    ln_post_params=ln_params,
                    conv_params=params)
            if use_noise_regularization:
                x = tf.keras.layers.GaussianNoise(stddev=0.1, seed=0)(x)
        x = \
            conv2d_wrapper(
                input_layer=x,
                bn_post_params=bn_params,
                ln_post_params=ln_params,
                conv_params=conv_params_res_2[i])
        x = \
            conv2d_wrapper(
                input_layer=x,
                bn_post_params=None,
                ln_post_params=None,
                conv_params=conv_params_res_3[i])
        node_level = (i, 0)
        nodes_visited.add(node_level)
        nodes_output[node_level] = x

    # add attention black at the deepest level of the encoder
    if use_attention_block:
        x = nodes_output[(depth-1, 0)]
        nodes_output[(depth - 1, 0)] = \
            tf.keras.layers.Concatenate(axis=-1)([
                self_attention_block(
                    input_layer=x,
                    conv_params=attention_conv_params,
                    use_logit_norm=True),
                x])

    i = None
    x = None

    while len(nodes_to_visit) > 0:
        node = nodes_to_visit.pop(0)
        logger.info(f"node: [{node}, "
                    f"nodes_visited: {nodes_visited}, "
                    f"nodes_to_visit: {nodes_to_visit}")
        logger.info(f"dependencies: {nodes_dependencies[node]}")
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
            logger.info(f"node: [{node}] dependencies not matches, continuing")
            nodes_to_visit.append(node)
            continue
        # sort it so all same level dependencies are first and added
        # as residual before finally concatenating the previous scale
        dependencies = \
            sorted(list(dependencies),
                   key=lambda d: d[0],
                   reverse=False)
        logger.debug(f"processing node: {node}, "
                     f"dependencies: {dependencies}, "
                     f"nodes_output: {list(nodes_output.keys())}")

        x_input = []

        logger.debug(f"node: [{node}], dependencies: {dependencies}")
        for d in dependencies:
            logger.debug(f"processing dependency: {d}")
            x = nodes_output[d]

            if d[0] == node[0]:
                # same level
                if dropout_params is not None:
                    x = tf.keras.layers.Dropout(rate=dropout_params["rate"])(x)
                if dropout_2d_params is not None:
                    x = tf.keras.layers.SpatialDropout2D(rate=dropout_2d_params["rate"])(x)
            elif d[0] > node[0]:
                # --- different up-scaling options
                if upsample_type == "conv2d_transpose":
                    # lower level, upscale
                    x = conv2d_wrapper(
                        input_layer=x,
                        bn_post_params=None,
                        ln_post_params=None,
                        conv_params=conv_params_up[node[0]],
                        conv_type=ConvType.CONV2D_TRANSPOSE)
                elif upsample_type in ["google_conv2d_transpose", "google_conv2d_transpose_bilinear"]:
                    params = copy.deepcopy(conv_params_up[node[0]])
                    params["kernel_size"] = (3, 3)
                    params["strides"] = (1, 1)
                    # lower level, upscale
                    x = \
                        tf.keras.layers.UpSampling2D(
                            size=(2, 2),
                            interpolation="bilinear")(x)
                    x = conv2d_wrapper(
                        input_layer=x,
                        bn_post_params=None,
                        ln_post_params=None,
                        conv_params=params,
                        conv_type=ConvType.CONV2D)
                elif upsample_type in ["google_conv2d_transpose_nearest"]:
                    params = copy.deepcopy(conv_params_up[node[0]])
                    params["kernel_size"] = (3, 3)
                    params["strides"] = (1, 1)
                    # lower level, upscale
                    x = \
                        tf.keras.layers.UpSampling2D(
                            size=(2, 2),
                            interpolation="nearest")(x)
                    x = conv2d_wrapper(
                        input_layer=x,
                        bn_post_params=None,
                        ln_post_params=None,
                        conv_params=params,
                        conv_type=ConvType.CONV2D)
                elif upsample_type == "convnext":
                    x = \
                        tf.keras.layers.UpSampling2D(
                            size=(2, 2),
                            interpolation="nearest")(x)
                    params = copy.deepcopy(conv_params_res_1[node[0]])
                    params["kernel_size"] = (kernel_size, kernel_size)
                    x = \
                        conv2d_wrapper(
                            input_layer=x,
                            bn_post_params=bn_params,
                            ln_post_params=ln_params,
                            conv_params=params)
                    x = \
                        conv2d_wrapper(
                            input_layer=x,
                            bn_post_params=bn_params,
                            ln_post_params=ln_params,
                            conv_params=conv_params_res_2[node[0]])
                    x = \
                        conv2d_wrapper(
                            input_layer=x,
                            bn_post_params=None,
                            ln_post_params=None,
                            conv_params=conv_params_res_3[node[0]])
                elif upsample_type in ["nn", "nearest"]:
                    # upsampling performed by a simple
                    # UpSampling2D with nearest neighbor interpolation
                    x = \
                        tf.keras.layers.UpSampling2D(
                            size=(2, 2),
                            interpolation="nearest")(x)
                elif upsample_type == "bilinear":
                    # upsampling performed by a simple
                    # UpSampling2D with bilinear interpolation
                    # this is the default in
                    # https://github.com/MrGiovanni/UNetPlusPlus/
                    # blob/master/pytorch/nnunet/network_architecture/generic_UNetPlusPlus.py
                    x = \
                        tf.keras.layers.UpSampling2D(
                            size=(2, 2),
                            interpolation="bilinear")(x)
                else:
                    raise ValueError(
                        f"don't know how to handle [{upsample_type}]")
            else:
                raise ValueError(f"node: {node}, dependencies: {dependencies}, "
                                 f"should not supposed to be here")

            if use_dropout_long_skips:
                if d[0] == node[0] and (d[1]+1) < node[1]:
                    # same level but not directly on the left
                    x = RandomOnOff(rate=0.5)(x)

            x_input.append(x)

        if len(x_input) == 1:
            x = x_input[0]
        elif len(x_input) > 0:
            x = tf.keras.layers.Concatenate()(x_input)
        else:
            raise ValueError("this must never happen")

        # --- convnext block
        params = copy.deepcopy(conv_params_res_1[node[0]])
        params["kernel_size"] = (kernel_size, kernel_size)
        x = \
            conv2d_wrapper(
                input_layer=x,
                bn_post_params=bn_params,
                ln_post_params=ln_params,
                conv_params=params)
        x = \
            conv2d_wrapper(
                input_layer=x,
                bn_post_params=bn_params,
                ln_post_params=ln_params,
                conv_params=conv_params_res_2[node[0]])
        x = \
            conv2d_wrapper(
                input_layer=x,
                bn_post_params=None,
                ln_post_params=None,
                conv_params=conv_params_res_3[node[0]])

        nodes_output[node] = x
        nodes_visited.add(node)

    # --- add dropout to the deepest layer
    last_node = max(0, width-depth-1)
    if dropout_params is not None:
        nodes_output[(depth-1, last_node)] = \
            tf.keras.layers.Dropout(rate=dropout_params["rate"])(
                nodes_output[(depth-1, last_node)])
    if dropout_2d_params is not None:
        nodes_output[(depth-1, last_node)] = \
            tf.keras.layers.SpatialDropout2D(rate=dropout_2d_params["rate"])(
                nodes_output[(depth-1, last_node)])

    # --- output layer here
    output_layers = []

    # depth outputs
    if multiple_scale_outputs:
        tmp_output_layers = []
        for i in range(1, depth, 1):
            d = i
            w = width - i - 1

            if d < 0 or w < 0:
                logger.error(f"there is no node[{d},{w}] please check your assumptions")
                continue
            x = nodes_output[(d, w)]
            x = tf.keras.layers.Layer(
                name=f"{output_layer_name}_{d}_{w}")(x)
            tmp_output_layers.append(x)
        # reverse here so deeper levels come on top
        output_layers += tmp_output_layers[::-1]

    # add as last the best output
    output_layers += [
        tf.keras.layers.Layer(name=f"{output_layer_name}_{0}_{width-1}")(
            nodes_output[(0, width - 1)])
    ]

    # IMPORTANT
    # reverse it so the deepest output is first
    # otherwise we will get the most shallow output
    output_layers = output_layers[::-1]

    return \
        tf.keras.Model(
            name=name,
            trainable=True,
            inputs=input_layer,
            outputs=output_layers)

# ---------------------------------------------------------------------
