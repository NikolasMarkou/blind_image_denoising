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
from .activations import differentiable_relu, differentiable_relu_layer
from .custom_layers import Multiplier, RandomOnOff, ChannelwiseMultiplier

# ---------------------------------------------------------------------


def load_image(
        path: Union[str, Path],
        color_mode: str = "rgb",
        target_size: Tuple[int, int] = None,
        normalize: bool = True) -> np.ndarray:
    """
    load image from file

    :param path:
    :param color_mode: grayscale or rgb
    :param target_size: size or None
    :param normalize: if true normalize to (-1,+1)
    :return: loaded normalized image
    """
    x = \
        tf.keras.preprocessing.image.load_img(
            path=path,
            color_mode=color_mode,
            target_size=target_size)

    x = tf.keras.preprocessing.image.img_to_array(x)
    x = np.array([x])
    if normalize:
        x = ((x / 255.0) * 2.0) - 1.0
    return x

# ---------------------------------------------------------------------


def load_config(
        config: Union[str, Dict, Path]) -> Dict:
    """
    Load configuration from multiple sources

    :param config: dict configuration or path to json configuration
    :return: dictionary configuration
    """
    try:
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
    except Exception as e:
        logger.error(e)
        raise ValueError(f"failed to load [{config}]")

# ---------------------------------------------------------------------


def input_shape_fixer(
        input_shape: Iterable):
    for i, shape in enumerate(input_shape):
        if shape == "?" or \
                shape == "" or \
                shape == "-1":
            input_shape[i] = None
    return input_shape

# ---------------------------------------------------------------------


def merge_iterators(
        *iterators):
    """
    Merge different iterators together

    :param iterators:
    """
    empty = {}
    for values in itertools.zip_longest(*iterators, fillvalue=empty):
        for value in values:
            if value is not empty:
                yield value

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


def random_choice(x, size, axis=0):
    """
    Randomly select size options from x
    """
    dim_x = tf.cast(tf.shape(x)[axis], tf.int64)
    indices = tf.range(0, dim_x, dtype=tf.int64)
    sample_index = tf.random.shuffle(indices)[:size]
    sample = tf.gather(x, sample_index, axis=axis)
    return sample

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
    shape = tf.keras.backend.int_shape(input_layer)
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
    return tf.keras.layers.Concatenate(axis=3)([input_layer, yy_grid, xx_grid])

# ---------------------------------------------------------------------


def conv2d_wrapper(
        input_layer,
        conv_params: Dict,
        bn_params: Dict = None,
        depthwise_scaling: bool = False):
    """
    wraps a conv2d with a preceding normalizer

    :param input_layer: the layer to operate on
    :param conv_params: conv2d parameters
    :param bn_params: batchnorm parameters, None to disable bn
    :param depthwise_scaling: if True add a learnable point-wise depthwise scaling conv2d at the end
    :return: transformed input
    """
    # --- argument checking
    if input_layer is None:
        raise ValueError("input_layer cannot be None")
    if conv_params is None:
        raise ValueError("conv_params cannot be None")

    # --- prepare arguments
    use_bn = bn_params is not None

    # --- perform the transformations
    x = input_layer
    if use_bn:
        x = tf.keras.layers.BatchNormalization(**bn_params)(x)
    # ideally this should be orthonormal
    x = tf.keras.layers.Conv2D(**conv_params)(x)
    # learn the proper scale of the previous layer
    if depthwise_scaling:
        x = tf.keras.layers.DepthwiseConv2D(
            use_bias=False,
            strides=(1, 1),
            padding="same",
            depth_multiplier=1,
            kernel_size=(1, 1),
            activation="linear",
            depthwise_initializer="ones",
            depthwise_regularizer="l1")(x)
    return x

# ---------------------------------------------------------------------


def dense_wrapper(
        input_layer,
        dense_params: Dict,
        bn_params: Dict = None,
        elementwise_params: Dict = None):
    """
    wraps a dense layer with a preceding normalizer

    :param input_layer: the layer to operate on
    :param dense_params: dense parameters
    :param bn_params: batchnorm parameters, None to disable bn
    :param elementwise_params: if True add a learnable elementwise scaling
    :return: transformed input
    """
    # --- argument checking
    if input_layer is None:
        raise ValueError("input_layer cannot be None")
    if dense_params is None:
        raise ValueError("dense_params cannot be None")

    # --- prepare arguments
    use_bn = bn_params is not None
    use_elementwise = elementwise_params is not None

    # --- perform the transformations
    x = input_layer
    if use_bn:
        x = tf.keras.layers.BatchNormalization(**bn_params)(x)
    # ideally this should be orthonormal
    x = tf.keras.layers.Dense(**dense_params)(x)
    # learn the proper scale of the previous layer
    if use_elementwise:
        x = ChannelwiseMultiplier(**elementwise_params)(x)
    return x

# ---------------------------------------------------------------------


def learnable_per_channel_multiplier_layer(
        input_layer,
        multiplier: float = 1.0,
        activation: str = "linear",
        kernel_regularizer: str = "l1",
        trainable: bool = True):
    """
    Constant learnable multiplier layer

    :param input_layer: input layer to be multiplied
    :param multiplier: multiplication constant
    :param activation: activation after the filter (linear by default)
    :param kernel_regularizer: regularize kernel weights (None by default)
    :param trainable: whether this layer is trainable or not
    :return: multiplied input_layer
    """
    # --- initialise to set kernel to required value
    def kernel_init(shape, dtype):
        kernel = np.zeros(shape)
        for i in range(shape[2]):
            kernel[:, :, i, 0] = \
                np.random.normal(scale=EPSILON_DEFAULT, loc=0.0)
        return kernel
    x = \
        tf.keras.layers.DepthwiseConv2D(
            kernel_size=1,
            padding="same",
            strides=(1, 1),
            use_bias=False,
            depth_multiplier=1,
            trainable=trainable,
            activation=activation,
            kernel_initializer=kernel_init,
            depthwise_initializer=kernel_init,
            kernel_regularizer=kernel_regularizer)(input_layer)
    # different scenarios
    if multiplier == 0.0:
        return x
    if multiplier == 1.0:
        return input_layer + x
    return (multiplier * input_layer) + x


# ---------------------------------------------------------------------


def mean_variance_local(
        input_layer,
        kernel_size: Tuple[int, int] = (5, 5)):
    """
    calculate window mean per channel and window variance per channel

    :param input_layer: the layer to operate on
    :param kernel_size: size of the kernel (window)
    :return: mean, variance tensors
    """
    # --- argument checking
    if input_layer is None:
        raise ValueError("input_layer cannot be empty")
    shape = tf.keras.backend.int_shape(input_layer)
    if len(shape) != 4:
        raise ValueError("input_layer must be a 4d tensor")
    if not isinstance(kernel_size, tuple):
        raise ValueError("kernel_size must be a tuple")

    # ---
    local_mean = \
        tf.keras.layers.AveragePooling2D(
            strides=(1, 1),
            padding="same",
            pool_size=kernel_size)(input_layer)
    local_diff = \
        tf.keras.layers.Subtract()(
            [input_layer, local_mean])
    local_diff = tf.keras.backend.square(local_diff)
    local_variance = \
        tf.keras.layers.AveragePooling2D(
            strides=(1, 1),
            padding="same",
            pool_size=kernel_size)(local_diff)

    return local_mean, local_variance

# ---------------------------------------------------------------------


def mean_sigma_local(
        input_layer,
        kernel_size: Tuple[int, int] = (5, 5),
        epsilon: float = EPSILON_DEFAULT):
    """
    calculate window mean per channel and window sigma per channel

    :param input_layer: the layer to operate on
    :param kernel_size: size of the kernel (window)
    :param epsilon: small number for robust sigma calculation
    :return: mean, sigma tensors
    """
    mean, variance = \
        mean_variance_local(
            input_layer=input_layer,
            kernel_size=kernel_size)

    sigma = tf.sqrt(tf.abs(variance) + epsilon)

    return mean, sigma


# ---------------------------------------------------------------------


def mean_sigma_global(
        input_layer,
        axis: List[int] = [1, 2, 3],
        sigma_epsilon: float = EPSILON_DEFAULT):
    """
    Create a global mean sigma per channel

    :param input_layer:
    :param axis:
    :param sigma_epsilon: small number to add for robust sigma calculation
    :return:
    """
    # --- argument checking
    if input_layer is None:
        raise ValueError("input_layer cannot be empty")
    shape = tf.keras.backend.int_shape(input_layer)
    if len(shape) != 4:
        raise ValueError("input_layer must be a 4d tensor")

    # --- compute mean and sigma
    def func(x):
        mean = tf.keras.backend.mean(x, axis=axis, keepdims=True)
        diff_2 = tf.keras.backend.square(x - mean)
        variance = tf.keras.backend.mean(diff_2, axis=axis, keepdims=True)
        sigma = tf.keras.backend.sqrt(variance + sigma_epsilon)
        return mean, sigma

    return \
        tf.keras.layers.Lambda(
            function=func,
            trainable=False)(input_layer)

# ---------------------------------------------------------------------


def sparse_block(
        input_layer,
        bn_params: Dict = None,
        threshold_sigma: float = 1.0,
        max_value: float = None,
        symmetric: bool = True,
        per_channel_sparsity: bool = False):
    """
    Create sparsity in an input layer

    :param input_layer:
    :param bn_params: batch norm parameters
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
    if max_value is not None and threshold_sigma > max_value:
        raise ValueError("threshold_sigma must be <= max_value")
    use_bn = bn_params is not None

    # --- function building
    def func_norm(args):
        x, mean_x, sigma_x = args
        return (x - mean_x) / sigma_x

    def func_abs_sign(args):
        y = args
        y_abs = tf.abs(y)
        y_sign = tf.sign(y)
        return y_abs, y_sign

    # --- computation is relu((mean-x)/sigma) with custom threshold
    # compute axis to perform mean/sigma calculation on
    if use_bn:
        # learnable model
        x_bn = \
            tf.keras.layers.BatchNormalization(
                **bn_params)(input_layer)
    else:
        # not learnable model
        shape = tf.keras.backend.int_shape(input_layer)
        int_shape = len(shape)
        k = 1
        if per_channel_sparsity:
            k = 2
        axis = list([i + 1 for i in range(max(int_shape - k, 1))])
        mean, sigma = \
            mean_sigma_global(
                input_layer,
                axis=axis)
        x_bn = \
            tf.keras.layers.Lambda(func_norm, trainable=False)([
                input_layer, mean, sigma])

    # ---
    if symmetric:
        x_abs, x_sign = \
            tf.keras.layers.Lambda(func_abs_sign, trainable=False)(x_bn)
        x_relu = \
            differentiable_relu(
                input_layer=x_abs,
                threshold=threshold_sigma,
                max_value=max_value)
        x_result = \
            tf.keras.layers.Multiply()([
                x_relu,
                x_sign,
            ])
    else:
        x_result = \
            differentiable_relu(
                input_layer=x_bn,
                threshold=threshold_sigma,
                max_value=max_value)

    return x_result

# ---------------------------------------------------------------------


def layer_denormalize(args):
    """
    Convert input [-0.5, +0.5] to [v0, v1] range
    """
    y, v_min, v_max = args
    y_clip = \
        tf.clip_by_value(
            t=y,
            clip_value_min=-0.5,
            clip_value_max=0.5)
    return (y_clip + 0.5) * (v_max - v_min) + v_min


# ---------------------------------------------------------------------


def layer_normalize(args):
    """
    Convert input from [v0, v1] to [-0.5, +0.5] range
    """
    y, v_min, v_max = args
    y_clip = \
        tf.clip_by_value(
            t=y,
            clip_value_min=v_min,
            clip_value_max=v_max)
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
    :return: normalization model
    """
    model_input = tf.keras.Input(shape=input_dims)

    # --- normalize input
    # from [min_value, max_value] to [-0.5, +0.5]
    model_output = \
        tf.keras.layers.Lambda(
            function=layer_normalize,
            trainable=False)([model_input,
                              float(min_value),
                              float(max_value)])

    # --- wrap model
    return tf.keras.Model(
        name=name,
        trainable=False,
        inputs=model_input,
        outputs=model_output)


# ---------------------------------------------------------------------


def build_denormalize_model(
        input_dims,
        min_value: float = 0.0,
        max_value: float = 255.0,
        name: str = "denormalize") -> tf.keras.Model:
    """
    Wrap a denormalize layer in a model

    :param input_dims: Models input dimensions
    :param min_value: Minimum value
    :param max_value: Maximum value
    :param name: name of the model
    :return: denormalization model
    """
    model_input = tf.keras.Input(shape=input_dims)

    # --- normalize input
    # from [min_value, max_value] to [-0.5, +0.5]
    model_output = \
        tf.keras.layers.Lambda(
            function=layer_denormalize,
            trainable=False)([model_input,
                              float(min_value),
                              float(max_value)])

    # --- wrap model
    return \
        tf.keras.Model(
            name=name,
            trainable=False,
            inputs=model_input,
            outputs=model_output)


# ---------------------------------------------------------------------
