import numpy as np
from typing import List
import tensorflow as tf
from tensorflow import keras

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .custom_logger import logger
from .constants import DEFAULT_EPSILON

# ---------------------------------------------------------------------

DELTA_KERNELS = {
    2: [[+1.0, -1.0],
        [+1.0, -1.0]],
    3: [[+1.0, +0.0, -1.0],
        [+2.0, +0.0, -2.0],
        [+1.0, +0.0, -1.0]],
    4: [[+3.0, +1.0, -1.0, -3.0],
        [+3.0, +1.0, -1.0, -3.0],
        [+3.0, +1.0, -1.0, -3.0],
        [+3.0, +1.0, -1.0, -3.0]],
    5: [[+2.0, +1.0, +0.0, -1.0, -2.0],
        [+2.0, +1.0, +0.0, -1.0, -2.0],
        [+2.0, +1.0, +0.0, -1.0, -2.0],
        [+2.0, +1.0, +0.0, -1.0, -2.0],
        [+2.0, +1.0, +0.0, -1.0, -2.0]]
}


# ---------------------------------------------------------------------


def delta_layer(
        kernel_size: int = 3,
        transpose: bool = False,
        trainable: bool = False):
    """
    create a delta layer

    :param kernel_size: 2,3,4,5
    :param transpose: whether to transpose x-y in kernel
    :param trainable: whether this layer is trainable or not
    :return: DepthwiseConv2D layer
    """
    # --- argument checking
    if kernel_size not in DELTA_KERNELS:
        raise ValueError("kernel_size [{0}] not found".format(kernel_size))

    # --- initialise to set kernel to required value
    def kernel_init(shape, dtype):
        kernel = np.zeros(shape)
        delta_kernel = DELTA_KERNELS[kernel_size]
        for i in range(shape[2]):
            kernel[:, :, i, 0] = delta_kernel
        if transpose:
            kernel = np.transpose(kernel, axes=[1, 0, 2, 3])
        return kernel

    return \
        keras.layers.DepthwiseConv2D(
            strides=(1, 1),
            padding="same",
            use_bias=False,
            depth_multiplier=1,
            activation="linear",
            trainable=trainable,
            kernel_initializer=kernel_init,
            depthwise_initializer=kernel_init,
            kernel_size=(kernel_size, kernel_size))

# ---------------------------------------------------------------------
# initialize like this so we can wrap them in tf.function later on
# ---------------------------------------------------------------------


DELTA_X_LAYERS = {
    k: delta_layer(k, transpose=False, trainable=False)
    for k in DELTA_KERNELS.keys()
}

DELTA_Y_LAYERS = {
    k: delta_layer(k, transpose=True, trainable=False)
    for k in DELTA_KERNELS.keys()
}


# ---------------------------------------------------------------------


def delta(
        input_layer: tf.Tensor,
        kernel_size: int = 3,
        transpose: bool = False) -> tf.Tensor:
    """
    Compute delta x for each channel layer

    :param input_layer: input layer to be filtered
    :param kernel_size: 2,3,4,5
    :param transpose: whether to transpose x-y in kernel
    :return: filtered input_layer
    """
    if transpose:
        return DELTA_Y_LAYERS[kernel_size](input_layer)
    return DELTA_X_LAYERS[kernel_size](input_layer)


# ---------------------------------------------------------------------


def delta_x(
        input_layer: tf.Tensor,
        kernel_size: int = 3) -> tf.Tensor:
    """
    Compute delta x for each channel layer

    :param input_layer: input layer to be filtered
    :param kernel_size: 2,3,4,5

    :return: filtered input_layer
    """
    return DELTA_X_LAYERS[kernel_size](input_layer)


# ---------------------------------------------------------------------


def delta_y(
        input_layer: tf.Tensor,
        kernel_size: int = 3) -> tf.Tensor:
    """
    Compute delta y for each channel layer

    :param input_layer: input layer to be filtered
    :param kernel_size: 2,3,4,5

    :return: filtered input_layer
    """
    return DELTA_Y_LAYERS[kernel_size](input_layer)


# ---------------------------------------------------------------------

def delta_xy_magnitude(
        input_layer: tf.Tensor,
        kernel_size: int = 3,
        alpha: float = 1.0,
        beta: float = 1.0,
        eps: float = DEFAULT_EPSILON) -> tf.Tensor:
    """
    Computes the delta loss of a layer
    (alpha * (dI/dx)^2 + beta * (dI/dy)^2) ^ 0.5

    :param input_layer:
    :param kernel_size: how big the delta kernel should be
    :param alpha: multiplier of dx
    :param beta: multiplier of dy
    :param eps: small value to add for stability

    :return: delta magnitude on both axis
    """
    dx = delta_x(input_layer, kernel_size=kernel_size)
    dy = delta_y(input_layer, kernel_size=kernel_size)
    return \
        tf.sqrt(
            tf.square(dx) * alpha +
            tf.square(dy) * beta +
            eps)

# ---------------------------------------------------------------------
