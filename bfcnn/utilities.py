# ---------------------------------------------------------------------

__author__ = "Nikolas Markou"
__version__ = "0.1.0"
__license__ = "None"

# ---------------------------------------------------------------------

import os
import copy
import json
import keras
import itertools
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import List, Tuple, Union, Dict

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .custom_logger import logger


# ---------------------------------------------------------------------

def load_config(
        config: Union[str, Dict, Path]) -> Dict:
    """
    Load configuration from multiple sources

    :param config: dict configuration or path to json configuration
    :return: dictionary configuration
    """
    if config is None:
        raise ValueError("config should not be empty")
    if isinstance(config, Dict):
        return config
    if isinstance(config, str) or isinstance(config, Path):
        if not os.path.isfile(str(config)):
            return ValueError(
                "configuration path [{0}] is not valid".format(
                    str(config)
                ))
        with open(str(config), "r") as f:
            return json.load(f)
    raise ValueError("don't know how to handle config [{0}]".format(config))

# ---------------------------------------------------------------------


def get_conv2d_weights(
        model: keras.Model) -> np.ndarray:
    """
    Get the conv2d weights from the model concatenated
    """
    weights = []
    for layer in model.layers:
        layer_config = layer.get_config()
        if "layers" not in layer_config:
            continue
        layer_weights = layer.get_weights()
        for i, l in enumerate(layer_config["layers"]):
            if l["class_name"] == "Conv2D":
                for w in layer_weights[i]:
                    w_flat = w.flatten()
                    weights.append(w_flat)
    return np.concatenate(weights)

# ---------------------------------------------------------------------


def normal_empirical_cdf(
        target_cdf: float = 0.5,
        mean: float = 0.0,
        sigma: float = 1.0,
        samples: int = 1000000,
        bins: int = 1000):
    """
    computes the value x for target_cdf
    """
    # --- argument checking
    if target_cdf <= 0.0001 or target_cdf >= 0.9999:
        raise ValueError(
            "target_cdf [{0}] must be between 0 and 1".format(target_cdf))
    if sigma <= 0:
        raise ValueError("sigma [{0}] must be > 0".format(sigma))

    # --- computer empirical cumulative sum
    z = \
        np.random.normal(
            loc=mean,
            scale=sigma,
            size=samples)
    h, x1 = np.histogram(z, bins=bins, density=True)
    dx = x1[1] - x1[0]
    f1 = np.cumsum(h) * dx

    # --- find the proper bin
    for i in range(bins):
        if f1[i] >= target_cdf:
            if i == 0:
                return x1[i]
            return x1[i]

    return -1


# ---------------------------------------------------------------------


def coords_layer(
        input_layer):
    """
    Create a coords layer

    :param input_layer:
    :return:
    """
    # --- argument checking
    if input_layer is None:
        raise ValueError("input_layer cannot be empty")
    shape = keras.backend.int_shape(input_layer)
    if len(shape) != 4:
        raise ValueError("input_layer must be a 4d tensor")
    # ---
    height = shape[1]
    width = shape[2]
    x_grid = np.linspace(0, 1, width)
    y_grid = np.linspace(0, 1, height)
    xx_grid, yy_grid = np.meshgrid(x_grid, y_grid)
    xx_grid = \
        tf.constant(
            value=xx_grid,
            dtype=tf.float32,
            shape=(1, xx_grid.shape[0], xx_grid.shape[1], 1))
    yy_grid = \
        tf.constant(
            value=yy_grid,
            dtype=tf.float32,
            shape=(1, yy_grid.shape[0], yy_grid.shape[1], 1))
    xx_grid = tf.repeat(xx_grid, axis=0, repeats=tf.shape(input_layer)[0])
    yy_grid = tf.repeat(yy_grid, axis=0, repeats=tf.shape(input_layer)[0])
    return keras.layers.Concatenate(axis=3)([input_layer, yy_grid, xx_grid])


# ---------------------------------------------------------------------


def mean_sigma_local(
        input_layer,
        kernel_size: Tuple[int, int] = (5, 5)):
    """
    Create a mean sigma

    :param input_layer:
    :param kernel_size:
    :return:
    """
    # --- argument checking
    if input_layer is None:
        raise ValueError("input_layer cannot be empty")
    shape = keras.backend.int_shape(input_layer)
    if len(shape) != 4:
        raise ValueError("input_layer must be a 4d tensor")

    # --- define functions
    def func_diff_2(args):
        x, y = args
        return tf.square(x - y)

    def func_sqrt_robust(args):
        x = args
        return tf.sqrt(tf.abs(x) + 0.00001)
    # ---
    mean = \
        keras.layers.AveragePooling2D(
            strides=(1, 1),
            padding="SAME",
            pool_size=kernel_size)(input_layer)
    diff_2 = \
        keras.layers.Lambda(func_diff_2, trainable=False)([input_layer, mean])
    variance = \
        keras.layers.AveragePooling2D(
            strides=(1, 1),
            padding="SAME",
            pool_size=kernel_size)(diff_2)
    sigma = \
        keras.layers.Lambda(func_sqrt_robust, trainable=False)(variance)

    return mean, sigma


# ---------------------------------------------------------------------


def mean_sigma_global(
        input_layer,
        axis: List[int] = [1, 2, 3]):
    """
    Create a global mean sigma per channel

    :param input_layer:
    :param axis:
    :return:
    """
    # --- argument checking
    if input_layer is None:
        raise ValueError("input_layer cannot be empty")
    shape = keras.backend.int_shape(input_layer)
    if len(shape) != 4:
        raise ValueError("input_layer must be a 4d tensor")

    # --- compute mean and sigma
    def func(x):
        mean = keras.backend.mean(x, axis=axis, keepdims=True)
        diff_2 = keras.backend.square(x - mean)
        variance = keras.backend.mean(diff_2, axis=axis, keepdims=True)
        sigma = keras.backend.sqrt(keras.backend.abs(variance) + 0.00001)
        return mean, sigma

    return keras.layers.Lambda(func, trainable=False)(input_layer)

# ---------------------------------------------------------------------


def sparse_block(
        input_layer,
        threshold_sigma: float = 1.0,
        max_value: float = None,
        symmetric: bool = True,
        per_channel_sparsity: bool = False):
    """
    Create sparsity in an input layer

    :param input_layer:
    :param threshold_sigma: sparsity of the results
    :param max_value: max allowed value
    :param symmetric: if true allow negative values else zero them off
    :param per_channel_sparsity: if true perform sparsity on per channel level

    :return: sparse results
    """
    # --- argument checking
    if threshold_sigma < 0:
        raise ValueError("threshold_sigma must be >= 0")
    if max_value is not None and max_value < 0:
        raise ValueError("max_value must be >= 0")
    if max_value is not None and threshold_sigma >= max_value:
        raise ValueError("threshold_sigma must be < max_value")

    # --- computation is relu((mean-x)/sigma) with custom threshold
    # compute axis to perform mean/sigma calculation on
    shape = keras.backend.int_shape(input_layer)
    int_shape = len(shape)
    k = 1
    if per_channel_sparsity:
        k = 2
    axis = list([i + 1 for i in range(max(int_shape - k, 1))])
    mean, sigma = \
        mean_sigma_global(
            input_layer,
            axis=axis)

    def func_norm(args):
        x, mean_x, sigma_x = args
        return (x - mean_x) / sigma_x

    x_bn = \
        keras.layers.Lambda(func_norm, trainable=False)([
            input_layer, mean, sigma])

    if symmetric:
        def func_abs_sign(args):
            x = args
            x_abs = keras.backend.abs(x)
            x_sign = keras.backend.sign(x)
            return x_abs, x_sign
        x_abs, x_sign = \
            keras.layers.Lambda(func_abs_sign, trainable=False)(x_bn)
        x_relu = \
            keras.layers.ReLU(
                threshold=threshold_sigma,
                max_value=max_value)(x_abs)
        x_result = \
            keras.layers.Multiply()([
                x_relu,
                x_sign,
            ])
    else:
        x_result = \
            keras.layers.ReLU(
                threshold=threshold_sigma,
                max_value=max_value)(x_bn)

    return x_result


# ---------------------------------------------------------------------


def conv2d_sparse(
        input_layer,
        groups: int = 1,
        filters: int = 32,
        padding: str = "same",
        strides: Tuple[int, int] = (1, 1),
        kernel_size: Tuple[int, int] = (3, 3),
        kernel_regularizer: str = "l1",
        kernel_initializer: str = "glorot_normal",
        threshold_sigma: float = 1.0,
        max_value: float = None,
        symmetric: bool = True):
    """
    Create a conv2d layer that is always sparse by %x percent
    This works by applying relu with a given threshold calculated
    by the normal distribution (Cumulative distribution function)
    that is forced by the batch normalization

    :param input_layer:
    :param threshold_sigma: sparsity of the results
    :param max_value: max allowed value
    :param symmetric: if true allow negative values else zero them off
    :param groups: groups to perform convolution in
    :param filters:
    :param padding:
    :param strides:
    :param kernel_size:
    :param kernel_regularizer:
    :param kernel_initializer:
    :return: sparse convolution results
    """
    # --- setup parameters
    # convolution parameters
    conv2d_params = dict(
        groups=groups,
        use_bias=False,
        filters=filters,
        padding=padding,
        strides=strides,
        activation="linear",
        kernel_size=kernel_size,
        kernel_regularizer=kernel_regularizer,
        kernel_initializer=kernel_initializer
    )

    # --- run convolution and then sparse it
    x_conv = keras.layers.Conv2D(**conv2d_params)(input_layer)

    return \
        sparse_block(
            input_layer=x_conv,
            threshold_sigma=threshold_sigma,
            symmetric=symmetric,
            max_value=max_value)


# ---------------------------------------------------------------------


def merge_iterators(*iterators):
    """
    Merge different iterators together
    """
    empty = {}
    for values in itertools.zip_longest(*iterators, fillvalue=empty):
        for value in values:
            if value is not empty:
                yield value


# ---------------------------------------------------------------------


def layer_denormalize(args):
    """
    Convert input [-0.5, +0.5] to [v0, v1] range
    """
    y, v_min, v_max = args
    y_clip = tf.clip_by_value(y, clip_value_min=-0.5, clip_value_max=0.5)
    return (y_clip + 0.5) * (v_max - v_min) + v_min


# ---------------------------------------------------------------------


def layer_normalize(args):
    """
    Convert input from [v0, v1] to [-0.5, +0.5] range
    """
    y, v_min, v_max = args
    y_clip = tf.clip_by_value(y, clip_value_min=v_min, clip_value_max=v_max)
    return (y_clip - v_min) / (v_max - v_min) - 0.5


# ---------------------------------------------------------------------


def build_normalize_model(
        input_dims,
        min_value: float = 0.0,
        max_value: float = 255.0,
        name: str = "normalize") -> keras.Model:
    """
    Wrap a normalize layer in a model

    :param input_dims: Models input dimensions
    :param min_value: Minimum value
    :param max_value: Maximum value
    :param name: name of the model
    :return:
    """
    model_input = keras.Input(shape=input_dims)

    # --- normalize input from [min_value, max_value] to [-0.5, +0.5]
    model_output = \
        keras.layers.Lambda(layer_normalize, trainable=False)([
            model_input, float(min_value), float(max_value)])

    # --- wrap model
    return keras.Model(
        name=name,
        inputs=model_input,
        outputs=model_output,
        trainable=False)

# ---------------------------------------------------------------------


def build_denormalize_model(
        input_dims,
        min_value: float = 0.0,
        max_value: float = 255.0,
        name: str = "denormalize") -> keras.Model:
    """
    Wrap a denormalize layer in a model

    :param input_dims: Models input dimensions
    :param min_value: Minimum value
    :param max_value: Maximum value
    :param name: name of the model
    :return:
    """
    model_input = keras.Input(shape=input_dims)

    # --- normalize input from [min_value, max_value] to [-0.5, +0.5]
    model_output = \
        keras.layers.Lambda(layer_denormalize, trainable=False)([
            model_input, float(min_value), float(max_value)])

    # --- wrap model
    return keras.Model(
        name=name,
        inputs=model_input,
        outputs=model_output,
        trainable=False)

# ---------------------------------------------------------------------


def build_resnet_model(
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
        name="resnet") -> keras.Model:
    """
    Build a resnet model

    :param input_dims: Models input dimensions
    :param no_layers: Number of resnet layers
    :param kernel_size: kernel size of the conv layers
    :param filters: number of filters per convolutional layer
    :param activation: intermediate activation
    :param final_activation: activation of the final layer
    :param channel_index: Index of the channel in dimensions
    :param use_bn: Use Batch Normalization
    :param use_bias: use bias
    :param kernel_regularizer: Kernel weight regularizer
    :param kernel_initializer: Kernel weight initializer
    :param name: Name of the model
    :return:
    """
    # --- variables
    bn_params = dict(
        center=use_bias,
        scale=True,
        momentum=0.999,
        epsilon=1e-4
    )
    conv_params = dict(
        filters=filters,
        strides=(1, 1),
        padding="same",
        use_bias=use_bias,
        activation=activation,
        kernel_size=kernel_size,
        kernel_regularizer=kernel_regularizer,
        kernel_initializer=kernel_initializer
    )
    depth_conv_params = dict(
        depth_multiplier=2,
        strides=(1, 1),
        padding="same",
        use_bias=use_bias,
        activation=activation,
        kernel_size=kernel_size,
        depthwise_regularizer=kernel_regularizer,
        depthwise_initializer=kernel_initializer
    )
    # intermediate conv
    intermediate_conv_params = copy.deepcopy(conv_params)
    intermediate_conv_params["kernel_size"] = 1

    # final conv
    final_conv_params = copy.deepcopy(conv_params)
    final_conv_params["kernel_size"] = 1
    final_conv_params["activation"] = final_activation
    final_conv_params["filters"] = input_dims[channel_index]
    del final_conv_params["kernel_regularizer"]

    # --- set input
    input_layer = keras.Input(shape=input_dims)

    # --- add base layer
    x = keras.layers.Conv2D(**conv_params)(input_layer)
    if use_bn:
        x = keras.layers.BatchNormalization(**bn_params)(x)

    # --- add resnet layers
    for i in range(no_layers):
        previous_layer = x
        x = keras.layers.DepthwiseConv2D(**depth_conv_params)(x)
        if use_bn:
            x = keras.layers.BatchNormalization(**bn_params)(x)
        x = keras.layers.Conv2D(**intermediate_conv_params)(x)
        if use_bn:
            x = keras.layers.BatchNormalization(**bn_params)(x)
        x = keras.layers.Add()([previous_layer, x])

    # --- output to original channels
    output_layer = \
        keras.layers.Conv2D(**final_conv_params)(x)

    output_layer = \
        keras.layers.Add()([output_layer, input_layer])

    return keras.Model(
        name=name,
        inputs=input_layer,
        outputs=output_layer)

# ---------------------------------------------------------------------


def build_sparse_resnet_model(
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
        name="resnet") -> keras.Model:
    """
    Build a resnet model with a sparsity constraint

    :param input_dims: Models input dimensions
    :param no_layers: Number of resnet layers
    :param kernel_size: kernel size of the conv layers
    :param filters: number of filters per convolutional layer
    :param activation: intermediate activation
    :param final_activation: activation of the final layer
    :param channel_index: Index of the channel in dimensions
    :param use_bn: Use Batch Normalization
    :param use_bias: use bias
    :param kernel_regularizer: Kernel weight regularizer
    :param kernel_initializer: Kernel weight initializer
    :param name: Name of the model
    :return:
    """
    # --- variables
    bn_params = dict(
        center=use_bias,
        scale=True,
        momentum=0.999,
        epsilon=1e-4
    )
    conv_params = dict(
        filters=filters,
        strides=(1, 1),
        padding="same",
        use_bias=use_bias,
        activation=activation,
        kernel_size=kernel_size,
        kernel_regularizer=kernel_regularizer,
        kernel_initializer=kernel_initializer
    )
    depth_conv_params = dict(
        depth_multiplier=2,
        strides=(1, 1),
        padding="same",
        use_bias=use_bias,
        activation="linear",
        kernel_size=kernel_size,
        depthwise_regularizer=kernel_regularizer,
        depthwise_initializer=kernel_initializer
    )
    # intermediate conv
    intermediate_conv_params = copy.deepcopy(conv_params)
    intermediate_conv_params["kernel_size"] = 1

    # final conv
    final_conv_params = copy.deepcopy(conv_params)
    final_conv_params["kernel_size"] = 1
    final_conv_params["activation"] = final_activation
    final_conv_params["filters"] = input_dims[channel_index]
    del final_conv_params["kernel_regularizer"]

    # --- set input
    input_layer = keras.Input(shape=input_dims)

    # --- add base layer
    x = keras.layers.Conv2D(**conv_params)(input_layer)
    if use_bn:
        x = keras.layers.BatchNormalization(**bn_params)(x)

    # --- add resnet layers
    for i in range(no_layers):
        previous_layer = x
        x = keras.layers.DepthwiseConv2D(**depth_conv_params)(x)
        x = \
            sparse_block(
                x,
                threshold_sigma=1.0,
                symmetric=True,
                per_channel_sparsity=True)
        x = keras.layers.Conv2D(**intermediate_conv_params)(x)
        if use_bn:
            x = keras.layers.BatchNormalization(**bn_params)(x)
        x = keras.layers.Add()([previous_layer, x])

    # --- output to original channels
    output_layer = \
        keras.layers.Conv2D(**final_conv_params)(x)

    output_layer = \
        keras.layers.Add()([output_layer, input_layer])

    return keras.Model(
        name=name,
        inputs=input_layer,
        outputs=output_layer)

# ---------------------------------------------------------------------


def build_sparse_resnet_mean_sigma_model(
        input_dims,
        no_layers: int,
        kernel_size: int,
        filters: int,
        final_activation: str = "linear",
        kernel_regularizer="l1",
        kernel_initializer="glorot_normal",
        channel_index: int = 2,
        name="sparse_resnet_mean_sigma") -> keras.Model:
    """
    Build a mean variance sparse resnet model

    :param input_dims: Models input dimensions
    :param no_layers: Number of resnet layers
    :param kernel_size: kernel size of the conv layers
    :param filters: number of filters per convolutional layer
    :param final_activation: activation of the final layer
    :param channel_index: Index of the channel in dimensions
    :param kernel_regularizer: Kernel weight regularizer
    :param kernel_initializer: Kernel weight initializer
    :param name: Name of the model
    :return:
    """
    # --- variables
    base_conv_params = dict(
        filters=filters,
        padding="same",
        use_bias=False,
        activation="relu",
        kernel_size=5,
        kernel_regularizer=kernel_regularizer,
        kernel_initializer=kernel_initializer
    )
    conv_params = dict(
        filters=filters,
        padding="same",
        use_bias=False,
        activation="relu",
        kernel_size=1,
        kernel_regularizer=kernel_regularizer,
        kernel_initializer=kernel_initializer
    )
    sparse_conv_params = dict(
        threshold_sigma=1.0,
        max_value=None,
        filters=filters,
        padding="same",
        symmetric=False,
        kernel_size=kernel_size,
        kernel_regularizer=kernel_regularizer,
        kernel_initializer=kernel_initializer
    )
    final_conv_params = dict(
        kernel_size=1,
        padding="same",
        strides=(1, 1),
        use_bias=False,
        activation=final_activation,
        filters=input_dims[channel_index],
        kernel_regularizer=kernel_regularizer,
        kernel_initializer=kernel_initializer
    )

    # --- set input
    input_layer = keras.Input(shape=input_dims)

    # --- add base layer
    x = input_layer
    mean, sigma = mean_sigma_local(x, kernel_size=(5, 5))
    x = keras.layers.Conv2D(**base_conv_params)(x - mean)

    # --- add resnet layers
    for i in range(no_layers):
        previous_layer = x
        x = keras.layers.Concatenate()([x, sigma])
        x = conv2d_sparse(x, **sparse_conv_params)
        x = keras.layers.Conv2D(**conv_params)(x)
        x = keras.layers.Add()([previous_layer, x])
    x = keras.layers.Concatenate()([x, sigma])

    # --- output to original channels
    output_layer = \
        keras.layers.Conv2D(**final_conv_params)(x)

    output_layer = \
        keras.layers.Add()([output_layer, input_layer])

    return keras.Model(
        name=name,
        inputs=input_layer,
        outputs=output_layer)

# ---------------------------------------------------------------------


def build_gatenet_model(
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
        name="gatenet") -> keras.Model:
    """
    Build a gatenet model

    :param input_dims: Models input dimensions
    :param no_layers: Number of resnet layers
    :param kernel_size: kernel size of the conv layers
    :param filters: number of filters per convolutional layer
    :param activation: intermediate activation
    :param final_activation: activation of the final layer
    :param channel_index: Index of the channel in dimensions
    :param use_bn: Use Batch Normalization
    :param use_bias: use bias
    :param kernel_regularizer: Kernel weight regularizer
    :param kernel_initializer: Kernel weight initializer
    :param name: Name of the model
    :return:
    """
    # --- variables
    bn_params = dict(
        center=use_bias,
        scale=True,
        momentum=0.999,
        epsilon=1e-4
    )
    conv_params = dict(
        filters=filters,
        strides=(1, 1),
        padding="same",
        use_bias=use_bias,
        activation=activation,
        kernel_size=kernel_size,
        kernel_regularizer=kernel_regularizer,
        kernel_initializer=kernel_initializer
    )
    gate_conv_params = dict(
        filters=filters,
        strides=(1, 1),
        padding="same",
        use_bias=use_bias,
        activation="linear",
        kernel_size=kernel_size,
        kernel_regularizer=kernel_regularizer,
        kernel_initializer=kernel_initializer
    )
    depth_conv_params = dict(
        depth_multiplier=2,
        strides=(1, 1),
        padding="same",
        use_bias=use_bias,
        activation=activation,
        kernel_size=kernel_size,
        depthwise_regularizer=kernel_regularizer,
        depthwise_initializer=kernel_initializer
    )
    # intermediate conv
    intermediate_conv_params = copy.deepcopy(conv_params)
    intermediate_conv_params["kernel_size"] = 1

    # final conv
    final_conv_params = copy.deepcopy(conv_params)
    final_conv_params["kernel_size"] = 1
    final_conv_params["activation"] = final_activation
    final_conv_params["filters"] = input_dims[channel_index]
    del final_conv_params["kernel_regularizer"]

    # --- set input
    input_layer = \
        keras.Input(shape=input_dims)

    # --- add base layer
    x = input_layer
    mean = \
        keras.layers.AveragePooling2D(
            strides=(1, 1),
            padding="SAME",
            pool_size=(5, 5))(x)
    diff = x - mean
    x = keras.layers.Conv2D(**conv_params)(diff)

    # --- add signal/gate resnet layers
    s_layer = x
    g_layer = x
    for i in range(no_layers):
        previous_s_layer = s_layer
        previous_g_layer = g_layer

        # --- add extra layers
        if i == 0:
            s_layer = previous_s_layer
            g_layer = previous_g_layer
        else:
            s_layer = \
                keras.layers.Concatenate()([
                    previous_s_layer,
                    diff])
            g_layer = \
                keras.layers.Concatenate()([
                    previous_s_layer,
                    previous_g_layer
                ])

        # --- expand
        s_layer = \
            keras.layers.DepthwiseConv2D(**depth_conv_params)(s_layer)
        g_layer = \
            keras.layers.Conv2D(**conv_params)(g_layer)

        # --- normalize
        if use_bn:
            s_layer = \
                keras.layers.BatchNormalization(**bn_params)(s_layer)
            g_layer = \
                keras.layers.BatchNormalization(**bn_params)(g_layer)

        # --- compress
        s_layer = \
            keras.layers.Conv2D(**intermediate_conv_params)(s_layer)
        g_layer = \
            keras.layers.Conv2D(**gate_conv_params)(g_layer)

        # --- compute activation per channel
        g_layer_activation = \
            keras.layers.GlobalAvgPool2D()(g_layer)
        g_layer_activation = \
            (keras.layers.Activation("tanh")(
                g_layer_activation * 2) + 1.0) / 2.0

        # --- compute activation per pixel
        p_layer_activation = \
            keras.backend.mean(g_layer, keepdims=True, axis=[3])
        p_layer_activation = \
            (keras.layers.Activation("tanh")(
                p_layer_activation * 2) + 1.0) / 2.0

        # mask channels
        s_layer = \
            keras.layers.Multiply()([s_layer, g_layer_activation])

        # mask positions
        s_layer = \
            keras.layers.Multiply()([s_layer, p_layer_activation])

        # add skip connection
        if i != 0:
            s_layer = \
                keras.layers.Add()([previous_s_layer, s_layer])
            g_layer = \
                keras.layers.Add()([previous_g_layer, g_layer])

    # --- output to original channels
    output_layer = \
        keras.layers.Conv2D(**final_conv_params)(s_layer)

    output_layer = \
        keras.layers.Add()([output_layer, input_layer])

    return keras.Model(
        name=name,
        inputs=input_layer,
        outputs=output_layer)

# ---------------------------------------------------------------------
