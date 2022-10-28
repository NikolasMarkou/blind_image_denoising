import os
import json
import itertools
from enum import Enum

import numpy as np
import tensorflow as tf
from pathlib import Path
from tensorflow import keras
from typing import List, Tuple, Union, Dict, Iterable

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .constants import *
from .custom_logger import logger
from .custom_layers import \
    Multiplier, \
    ChannelwiseMultiplier

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
        x = layer_normalize((x, 0.0, 255.0))
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


def probabilistic_drop_off(
        iterator: Iterable,
        probability: float = 0.5):
    """
    randomly zero out an element of the iterator

    :param iterator:
    :param probability: probability of an element not being affected
    :return:
    """
    for value in iterator:
        if np.random.uniform(low=0, high=1.0, size=None) > probability:
            yield value * 0.0
        else:
            yield value

# ---------------------------------------------------------------------


def gaussian_kernel(
        size: Tuple[int, int],
        nsig: Tuple[float, float],
        dtype: np.float64) -> np.ndarray:
    """
    builds a 2D Gaussian kernel array

    :param size: size of of the grid
    :param nsig: max value out of the gaussian on the xy axis
    :param dtype: number type
    :return: 2d gaussian grid
    """
    assert len(nsig) == 2
    assert len(size) == 2
    kern1d = [
        np.linspace(
            start=-np.abs(nsig[i]),
            stop=np.abs(nsig[i]),
            num=size[i],
            endpoint=True,
            dtype=dtype)
        for i in range(2)
    ]
    x, y = np.meshgrid(kern1d[0], kern1d[1])
    d = np.sqrt(x * x + y * y)
    sigma, mu = 1.0, 0.0
    g = np.exp(-((d - mu) ** 2 / (2.0 * (sigma ** 2))))
    return g / g.sum()

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


def random_choice(
        x: tf.Tensor,
        size: int = 1,
        axis: int = 0) -> tf.Tensor:
    """
    Randomly select size options from x
    """
    dim_x = tf.cast(tf.shape(x)[axis], tf.int64)
    indices = tf.range(0, dim_x, dtype=tf.int64)
    sample_index = tf.random.shuffle(indices)[:size]
    return tf.gather(x, sample_index, axis=axis)

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


class ConvType(Enum):
    CONV2D = 0

    CONV2D_DEPTHWISE = 1

    CONV2D_TRANSPOSE = 2

    @staticmethod
    def from_string(type_str: str) -> "ConvType":
        # --- argument checking
        if type_str is None:
            raise ValueError("type_str must not be null")
        if not isinstance(type_str, str):
            raise ValueError("type_str must be string")
        type_str = type_str.strip().upper()
        if len(type_str) <= 0:
            raise ValueError("stripped type_str must not be empty")

        # --- clean string and get
        return ConvType[type_str]

    def to_string(self) -> str:
        return self.name


def conv2d_wrapper(
        input_layer,
        conv_params: Dict,
        bn_params: Dict = None,
        pre_activation: str = None,
        channelwise_scaling: bool = False,
        multiplier_scaling: bool = False,
        conv_type: Union[ConvType, str] = ConvType.CONV2D):
    """
    wraps a conv2d with a preceding normalizer

    :param input_layer: the layer to operate on
    :param conv_params: conv2d parameters
    :param bn_params: batchnorm parameters, None to disable bn
    :param pre_activation: activation after the batchnorm, None to disable
    :param conv_type: if true use depthwise convolution,
    :param channelwise_scaling: if True add a learnable channel wise scaling at the end
    :param multiplier_scaling: if True add a learnable single scale at the end

    :return: transformed input
    """
    # --- argument checking
    if input_layer is None:
        raise ValueError("input_layer cannot be None")
    if conv_params is None:
        raise ValueError("conv_params cannot be None")

    # --- prepare arguments
    use_bn = bn_params is not None
    use_pre_activation = pre_activation is not None
    # TODO restructure this
    if isinstance(conv_type, str):
        conv_type = ConvType.from_string(conv_type)
    if "depth_multiplier" in conv_params:
        if conv_type != ConvType.CONV2D_DEPTHWISE:
            logger.info("Changing conv_type to CONV2D_DEPTHWISE because it contains depth_multiplier argument "
                        f"[conv_params[\'depth_multiplier\']={conv_params['depth_multiplier']}]")
        conv_type = ConvType.CONV2D_DEPTHWISE
    if "dilation_rate" in conv_params:
        if conv_type != ConvType.CONV2D_TRANSPOSE:
            logger.info("Changing conv_type to CONV2D_TRANSPOSE because it contains dilation argument "
                        f"[conv_params[\'dilation_rate\']={conv_params['dilation_rate']}]")
        conv_type = ConvType.CONV2D_TRANSPOSE

    # --- perform batchnorm and preactivation
    x = input_layer

    if use_bn:
        x = tf.keras.layers.BatchNormalization(**bn_params)(x)
    if use_pre_activation:
        x = tf.keras.layers.Activation(pre_activation)(x)

    # --- convolution
    if conv_type == ConvType.CONV2D:
        x = tf.keras.layers.Conv2D(**conv_params)(x)
    elif conv_type == ConvType.CONV2D_DEPTHWISE:
        x = tf.keras.layers.DepthwiseConv2D(**conv_params)(x)
    elif conv_type == ConvType.CONV2D_TRANSPOSE:
        x = tf.keras.layers.Conv2DTranspose(**conv_params)(x)
    else:
        raise ValueError(f"don't know how to handle this [{conv_type}]")

    # --- learn the proper scale of the previous layer
    if channelwise_scaling:
        x = \
            ChannelwiseMultiplier(
                multiplier=1.0,
                regularizer=keras.regularizers.L1(DEFAULT_CHANNELWISE_MULTIPLIER_L1),
                trainable=True,
                activation="relu")(x)
    if multiplier_scaling:
        x = \
            Multiplier(
                multiplier=1.0,
                regularizer=keras.regularizers.L1(DEFAULT_MULTIPLIER_L1),
                trainable=True,
                activation="relu")(x)
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
        epsilon: float = DEFAULT_EPSILON):
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

    sigma = tf.sqrt(variance + epsilon)

    return mean, sigma


# ---------------------------------------------------------------------


def mean_sigma_global(
        input_layer,
        axis: List[int] = [1, 2, 3],
        epsilon: float = DEFAULT_EPSILON):
    """
    Create a global mean sigma per channel

    :param input_layer:
    :param axis:
    :param epsilon: small number to add for robust sigma calculation
    :return:
    """
    # --- argument checking
    if input_layer is None:
        raise ValueError("input_layer cannot be empty")
    shape = tf.keras.backend.int_shape(input_layer)
    if len(shape) != 4:
        raise ValueError("input_layer must be a 4d tensor")

    # --- build block
    x = input_layer
    mean = tf.reduce_mean(x, axis=axis, keepdims=True)
    diff_2 = tf.square(x - mean)
    variance = tf.reduce_mean(diff_2, axis=axis, keepdims=True)
    sigma = tf.sqrt(variance + epsilon)
    return mean, sigma


# ---------------------------------------------------------------------


def sparse_block(
        input_layer: tf.Tensor,
        bn_params: Dict = None,
        threshold_sigma: float = 1.0,
        symmetrical: bool = False,
        reverse: bool = False,
        soft_sparse: bool = False) -> tf.Tensor:
    """
    create sparsity in an input layer (keeps only positive)

    :param input_layer:
    :param bn_params: batch norm parameters, leave None for disabling
    :param threshold_sigma: sparsity of the results (assuming negative values in input)
    -3 -> 0.1% sparsity
    -2 -> 2.3% sparsity
    -1 -> 15.9% sparsity
    0  -> 50% sparsity
    +1 -> 84.1% sparsity
    +2 -> 97.7% sparsity
    +3 -> 99.9% sparsity
    :param symmetrical: if True use abs values, if False cutoff negatives
    :param reverse: if True cutoff large values, if False cutoff small values
    :param soft_sparse: if True use sigmoid, if False use relu

    :return: sparse results
    """
    # --- argument checking
    if threshold_sigma < 0:
        raise ValueError("threshold_sigma must be >= 0")

    # --- set variables
    use_bn = bn_params is not None

    # --- build sparse block
    x = input_layer

    # normalize
    if use_bn:
        x_bn = \
            tf.keras.layers.BatchNormalization(
                **bn_params)(x)
    else:
        mean, sigma = mean_sigma_global(input_layer=x)
        x_bn = (x - mean) / (sigma + DEFAULT_EPSILON)

    if symmetrical:
        x_bn = tf.abs(x_bn)

    # threshold based on normalization
    # keep only positive above threshold
    if soft_sparse:
        x_binary = \
            tf.nn.sigmoid(x_bn - threshold_sigma)
    else:
        x_binary = \
            tf.nn.relu(tf.sign(x_bn - threshold_sigma))

    # focus on small values
    if reverse:
        x_binary = 1.0 - x_binary

    # zero out values below the threshold
    return \
        tf.keras.layers.Multiply()([
            x_binary,
            x,
        ])

# ---------------------------------------------------------------------


def stats_2d_block(
        input_layer: tf.Tensor) -> tf.Tensor:
    """
    compute the basic stats of a tensor per channel
    """
    x = input_layer
    x_max = tf.reduce_max(x, axis=[1, 2], keepdims=False)
    x_min = tf.reduce_min(x, axis=[1, 2], keepdims=False)
    x_mean = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
    x_variance = \
        tf.reduce_mean(
            tf.square(x - x_mean), axis=[1, 2], keepdims=False)
    x_sigma = tf.sqrt(x_variance + DEFAULT_EPSILON)
    x_mean = tf.squeeze(x_mean, axis=[1, 2])
    return \
        tf.keras.layers.Concatenate(axis=-1)([
            x_max,
            x_min,
            x_mean,
            x_sigma
        ])
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
    Convert input from [v0, v1] to [-1.0, +1.0] range
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
    # from [-0.5, +0.5] to [v0, v1] range
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
