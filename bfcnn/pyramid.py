r"""
blocks and builders for pyramid handling
"""

# ---------------------------------------------------------------------

__author__ = "Nikolas Markou"
__version__ = "1.0.0"
__license__ = "MIT"

# ---------------------------------------------------------------------

import numpy as np
from enum import Enum
import tensorflow as tf
from tensorflow import keras
from typing import Dict, Tuple, Union, List

# ---------------------------------------------------------------------

from .custom_logger import logger

# ---------------------------------------------------------------------


DEFAULT_XY_MAX = (2.0, 2.0)
DEFAULT_KERNEL_SIZE = (5, 5)
DEFAULT_UPSCALE_XY_MAX = (1.0, 1.0)
DEFAULT_UPSCALE_KERNEL_SIZE = (5, 5)


# ---------------------------------------------------------------------


def gaussian_filter_layer(
        kernel_size: Tuple[int, int] = DEFAULT_KERNEL_SIZE,
        strides: Tuple[int, int] = (1, 1),
        dilation_rate: Tuple[int, int] = (1, 1),
        padding: str = "same",
        xy_max: Tuple[float, float] = DEFAULT_XY_MAX,
        activation: str = "linear",
        trainable: bool = False,
        use_bias: bool = False,
        name: str = None):
    """
    build a gaussian filter layer

    :param kernel_size: kernel size tuple
    :param strides: strides, leave to (1,1) to get the same size
    :param dilation_rate: dilation rate, leave to (1,1) to get the same size
    :param padding: padding of the layer
    :param activation: activation after filter application
    :param trainable: whether trainable or not
    :param use_bias: whether to use bias or not
    :param xy_max: how far the gaussian are we going
        (symmetrically) on the 2 axis
    :param name: name of the convolutional layer

    :return: convolutional layer
    """

    # define the 2d gaussian initialization kernel
    def gaussian_kernel(
            size: Tuple[int, int],
            nsig: Tuple[float, float]) -> np.ndarray:
        """
        builds a 2D Gaussian kernel array

        :param size: size of of the grid
        :param nsig: max value out of the gaussian on the xy axis
        :return: 2d gaussian grid
        """
        assert len(nsig) == 2
        assert len(size) == 2
        kern1d = [
            np.linspace(
                start=-np.abs(nsig[i]),
                stop=np.abs(nsig[i]),
                num=size[i],
                endpoint=True)
            for i in range(2)
        ]
        x, y = np.meshgrid(kern1d[0], kern1d[1])
        d = np.sqrt(x * x + y * y)
        sigma, mu = 1.0, 0.0
        g = np.exp(-((d - mu) ** 2 / (2.0 * (sigma ** 2))))
        return g / g.sum()

    # initialise to set kernel to required value
    def kernel_init(shape, dtype):
        logger.info(f"building gaussian kernel with size: {shape}")
        kernel = np.zeros(shape)
        kernel_channel = \
            gaussian_kernel(
                size=(shape[0], shape[1]),
                nsig=xy_max)
        for i in range(shape[2]):
            kernel[:, :, i, 0] = kernel_channel
        return kernel

    return \
        keras.layers.DepthwiseConv2D(
            name=name,
            strides=strides,
            padding=padding,
            use_bias=use_bias,
            depth_multiplier=1,
            trainable=trainable,
            activation=activation,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            kernel_initializer=kernel_init,
            depthwise_initializer=kernel_init)


# ---------------------------------------------------------------------


def gaussian_filter_block(
        input_layer,
        kernel_size: Tuple[int, int] = DEFAULT_KERNEL_SIZE,
        strides: Tuple[int, int] = (1, 1),
        dilation_rate: Tuple[int, int] = (1, 1),
        padding: str = "same",
        xy_max: Tuple[float, float] = DEFAULT_XY_MAX,
        activation: str = "linear",
        trainable: bool = False,
        use_bias: bool = False,
        name: str = None):
    """
    Build a gaussian filter block
    apply it to input_layer and return result

    :param input_layer: the layer to apply the filter to
    :param kernel_size: kernel size tuple
    :param strides: strides, leave to (1,1) to get the same size
    :param dilation_rate: dilation rate, leave to (1,1) to get the same size
    :param padding: padding of the layer
    :param activation: activation after filter application
    :param trainable: whether trainable or not
    :param use_bias: whether to use bias or not
    :param xy_max: how far the gaussian are we going
        (symmetrically) on the 2 axis
    :param name: name of the convolutional layer

    :return: filtered input
    """
    gaussian_filter = \
        gaussian_filter_layer(
            name=name,
            xy_max=xy_max,
            strides=strides,
            padding=padding,
            use_bias=use_bias,
            trainable=trainable,
            activation=activation,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate
        )

    return gaussian_filter(input_layer)

# ---------------------------------------------------------------------


def downsample_2x2_block(
        input_layer,
        kernel_size: Tuple[int, int] = DEFAULT_KERNEL_SIZE,
        xy_max: Tuple[float, float] = DEFAULT_XY_MAX,
        trainable: bool = False):
    """
    creates an downsample block and filters the input_layer

    :param input_layer: the input layer
    :param kernel_size: kernel size tuple
    :param xy_max: how far the gaussian are we going
        (symmetrically) on the 2 axis
    :param trainable: is the pyramid trainable (default False)
    :return: filtered input layer
    """
    x = input_layer
    x = \
        gaussian_filter_block(
            input_layer=x,
            xy_max=xy_max,
            padding="same",
            strides=(2, 2),
            trainable=trainable,
            kernel_size=kernel_size)
    return x


# ---------------------------------------------------------------------


class MixType(Enum):
    EQUAL = 1
    EXPONENTIAL = 2

    @staticmethod
    def from_string(type_str: str) -> "MixType":
        # --- argument checking
        if type_str is None:
            raise ValueError("type_str must not be null")
        if not isinstance(type_str, str):
            raise ValueError("type_str must be string")
        if len(type_str.strip()) <= 0:
            raise ValueError("stripped type_str must not be empty")

        # --- clean string and get
        return MixType[type_str.strip().upper()]

    def to_string(self) -> str:
        return self.name


# ---------------------------------------------------------------------


class PyramidType(Enum):
    NONE = 1
    GAUSSIAN = 2
    LAPLACIAN = 3

    @staticmethod
    def from_string(type_str: str) -> "PyramidType":
        # --- argument checking
        if type_str is None:
            raise ValueError("type_str must not be null")
        if not isinstance(type_str, str):
            raise ValueError("type_str must be string")
        if len(type_str.strip()) <= 0:
            raise ValueError("stripped type_str must not be empty")

        # --- clean string and get
        return PyramidType[type_str.strip().upper()]

    def to_string(self) -> str:
        return self.name


# ---------------------------------------------------------------------


def build_gaussian_pyramid_model(
        input_dims: Union[Tuple, List],
        levels: int,
        kernel_size: Tuple[int, int] = DEFAULT_KERNEL_SIZE,
        xy_max: Tuple[float, float] = DEFAULT_XY_MAX,
        trainable: bool = False,
        name: str = "gaussian_pyramid") -> keras.Model:
    """
    Build a gaussian pyramid model

    :param input_dims: input dimensions
    :param levels: how many levels to go down the pyramid (level 0 is the original input)
    :param kernel_size: kernel size tuple
    :param xy_max: how far the gaussian are we going
        (symmetrically) on the 2 axis
    :param trainable: is the pyramid trainable (default False)
    :param name: name of the model
    :return: gaussian pyramid keras model
    """
    # --- prepare input
    input_layer = \
        keras.Input(
            name="input_tensor",
            shape=input_dims)

    # --- split input in levels
    level_x = \
        keras.layers.Layer(name="level_0")(input_layer)
    multiscale_layers = [level_x]
    for level in range(1, levels):
        level_x = \
            downsample_2x2_block(
                input_layer=level_x,
                xy_max=xy_max,
                trainable=trainable,
                kernel_size=kernel_size)
        level_x = \
            keras.layers.Layer(name=f"level_{level}")(level_x)
        multiscale_layers.append(level_x)

    return \
        keras.Model(
            name=name,
            trainable=trainable,
            inputs=input_layer,
            outputs=multiscale_layers)


# ---------------------------------------------------------------------


def build_inverse_gaussian_pyramid_model(
        input_dims: Union[Tuple, List],
        levels: int,
        trainable: bool = False,
        mix_type: MixType = MixType.EQUAL,
        name: str = "inverse_gaussian_pyramid") -> keras.Model:
    """
    Build a gaussian pyramid model

    :param input_dims: input dimensions
    :param levels: how many levels to go down the pyramid
    :param kernel_size: kernel size tuple
    :param xy_max: how far the gaussian are we going
        (symmetrically) on the 2 axis
    :param trainable: is the pyramid trainable (default False)
    :param mix_type: how to mix the different layers
    :param name: name of the model
    :return: inverse gaussian pyramid keras model
    """
    # --- prepare input
    input_layers = [
        keras.Input(name=f"input_tensor_{i}", shape=input_dims)
        for i in range(0, levels)
    ]

    # --- compute layer weights
    if mix_type == MixType.EQUAL:
        layer_weights = np.array([1.0 / levels] * levels)
    elif mix_type == MixType.EXPONENTIAL:
        layer_weights = np.array(range(levels))
        layer_weights = 2 ** layer_weights
        layer_weights = layer_weights / np.sum(layer_weights)
    else:
        raise ValueError("don't know how to use mix_type [{0}]".format(
            mix_type))

    # --- merge different levels (from smallest to biggest)
    output_layer = input_layers[-1] * layer_weights[-1]

    for i in range(levels - 2, -1, -1):
        level_up_x = \
            keras.layers.UpSampling2D(
                size=(2, 2),
                interpolation="bilinear")(output_layer)
        output_layer = \
            keras.layers.Add()([
                level_up_x,
                input_layers[i] * layer_weights[i]])
    output_layer = \
        keras.layers.Layer(name="output_tensor")(output_layer)

    return \
        keras.Model(
            name=name,
            trainable=trainable,
            inputs=input_layers,
            outputs=output_layer)


# ---------------------------------------------------------------------


def build_laplacian_pyramid_model(
        input_dims: Union[Tuple, List],
        levels: int,
        kernel_size: Tuple[int, int] = DEFAULT_KERNEL_SIZE,
        trainable: bool = False,
        name: str = "laplacian_pyramid") -> keras.Model:
    """
    build a laplacian pyramid model

    :param input_dims: input dimensions
    :param levels: how many levels to go down the pyramid
    :param kernel_size: kernel size tuple
    :param trainable: is the pyramid trainable (default False)
    :param name: name of the model
    :return: laplacian pyramid keras model
    """
    # ---
    logger.info(f"building laplacian pyramid model with: {levels} levels")

    # --- prepare input
    input_layer = \
        keras.Input(
            name="input_tensor",
            shape=input_dims)

    # --- split input in levels
    level_x = input_layer
    multiscale_layers = []

    for level in range(0, levels - 1):
        level_x_down = \
            keras.layers.AveragePooling2D(
                pool_size=kernel_size,
                strides=(2, 2),
                padding="same")(level_x)
        level_x_smoothed = \
            keras.layers.UpSampling2D(
                size=(2, 2),
                interpolation="bilinear")(level_x_down)
        level_x_diff = level_x - level_x_smoothed
        level_x = level_x_down
        multiscale_layers.append(level_x_diff)
    level_x = \
        keras.layers.Layer(name=f"level_{levels - 1}")(level_x)
    multiscale_layers.append(level_x)

    return \
        keras.Model(
            name=name,
            trainable=trainable,
            inputs=input_layer,
            outputs=multiscale_layers)


# ---------------------------------------------------------------------


def build_inverse_laplacian_pyramid_model(
        input_dims: Union[Tuple, List],
        levels: int,
        trainable: bool = False,
        name: str = "inverse_laplacian_pyramid") -> keras.Model:
    """
    Build an inverse laplacian pyramid model

    :param input_dims: input dimensions
    :param levels: how many levels to go down the pyramid
    :param trainable: is the pyramid trainable (default False)
    :param name: name of the model
    :return: inverse laplacian pyramid keras model
    """
    # ---- logging
    logger.info(f"building inverse laplacian pyramid model with: {levels} levels")

    # --- prepare input
    input_layers = [
        keras.Input(name=f"input_tensor_{i}", shape=input_dims)
        for i in range(0, levels)
    ]

    # --- merge different levels (from smallest to biggest)
    output_layer = None
    for level_x in reversed(input_layers):
        if output_layer is None:
            output_layer = level_x
        else:
            level_up_x = \
                keras.layers.UpSampling2D(
                    size=(2, 2),
                    interpolation="bilinear")(output_layer)
            output_layer = level_up_x + level_x
    output_layer = \
        keras.layers.Layer(name="output_tensor")(output_layer)
    return \
        keras.Model(
            name=name,
            trainable=trainable,
            inputs=input_layers,
            outputs=output_layer)


# ---------------------------------------------------------------------


def build_pyramid_model(
        input_dims: Union[Tuple, List],
        config: Dict) -> keras.Model:
    """
    Builds a multiscale pyramid model

    :param input_dims: input dimensions
    :param config: pyramid configuration
    :return: pyramid model
    """
    if config is None:
        no_levels = 1
        xy_max = DEFAULT_XY_MAX
        kernel_size = DEFAULT_KERNEL_SIZE
        pyramid_type = PyramidType.from_string("NONE")
    else:
        no_levels = config.get("levels", 1)
        xy_max = config.get("xy_max", DEFAULT_XY_MAX)
        kernel_size = config.get("kernel_size", DEFAULT_KERNEL_SIZE)
        pyramid_type = PyramidType.from_string(config.get("type", "NONE"))

    if pyramid_type == PyramidType.GAUSSIAN:
        pyramid_model = \
            build_gaussian_pyramid_model(
                input_dims=input_dims,
                levels=no_levels,
                xy_max=xy_max,
                kernel_size=kernel_size)
    elif pyramid_type == PyramidType.LAPLACIAN:
        pyramid_model = \
            build_laplacian_pyramid_model(
                input_dims=input_dims,
                levels=no_levels,
                kernel_size=kernel_size)
    elif pyramid_type == PyramidType.NONE:
        pyramid_model = \
            build_gaussian_pyramid_model(
                input_dims=input_dims,
                levels=no_levels,
                xy_max=xy_max,
                kernel_size=kernel_size)
    else:
        raise ValueError(
            "don't know how to build pyramid type [{0}]".format(pyramid_type))
    return pyramid_model


# ---------------------------------------------------------------------


def build_inverse_pyramid_model(
        input_dims: Union[Tuple, List],
        config: Dict) -> keras.Model:
    """
    Builds an inverse multiscale pyramid model

    :param input_dims: input dimensions
    :param config: pyramid configuration
    :return: inverse pyramid model
    """
    if config is None:
        no_levels = 1
        pyramid_type = PyramidType.from_string("NONE")
        mix_type = MixType.from_string(MixType.EQUAL.to_string())
    else:
        no_levels = config.get("levels", 1)
        pyramid_type = PyramidType.from_string(config.get("type", "NONE"))
        mix_type = MixType.from_string(config.get("mix_type", MixType.EQUAL.to_string()))

    if pyramid_type == PyramidType.GAUSSIAN:
        pyramid_model = \
            build_inverse_gaussian_pyramid_model(
                input_dims=input_dims,
                levels=no_levels,
                mix_type=mix_type)
    elif pyramid_type == PyramidType.LAPLACIAN:
        pyramid_model = \
            build_inverse_laplacian_pyramid_model(
                input_dims=input_dims,
                levels=no_levels)
    elif pyramid_type == PyramidType.NONE:
        pyramid_model = \
            build_inverse_gaussian_pyramid_model(
                input_dims=input_dims,
                levels=no_levels,
                mix_type=mix_type)
    else:
        raise ValueError(
            "don't know how to build pyramid type [{0}]".format(pyramid_type))
    return pyramid_model

# ---------------------------------------------------------------------
