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
from .custom_layers import TrainableMultiplier, RandomOnOff

# ---------------------------------------------------------------------


def differentiable_relu_layer(
        threshold: float = 0.0,
        max_value: float = 6.0,
        multiplier: float = 10.0):
    """
    Creates a differentiable relu layer

    :param threshold: lower bound value before zeroing
    :param max_value: max allowed value
    :param multiplier: controls steepness
    :result: activation layer
    """
    # --- arguments check
    if threshold is None:
        raise ValueError("threshold must not be empty")
    if max_value is not None:
        if threshold > max_value:
            raise ValueError(
                f"max_value [{max_value}] must be > threshold [{threshold}")

    # --- function building
    def func_diff_relu_0(args):
        x = args
        step_threshold = tf.math.sigmoid(multiplier * (x - threshold))
        step_max_value = tf.math.sigmoid(multiplier * (x - max_value))
        result = \
            ((step_max_value * max_value) + ((1.0 - step_max_value) * x)) * \
            step_threshold
        return result

    def func_diff_relu_1(args):
        x = args
        step_threshold = tf.math.sigmoid(multiplier * (x - threshold))
        result = step_threshold * x
        return result

    fn = func_diff_relu_0
    if max_value is None:
        fn = func_diff_relu_1

    return \
        keras.layers.Lambda(
            function=fn,
            trainable=False)

# ---------------------------------------------------------------------


def differentiable_relu(
        input_layer,
        **kwargs):
    """
    Creates a differentiable relu layer and filters input_layer

    :param input_layer: input tensor to operate on
    :result: filtered input layer
    """
    # --- arguments check
    if input_layer is None:
        raise ValueError("input layer cannot be None")

    # --- create layer
    layer = differentiable_relu_layer(**kwargs)

    # --- filter input layer
    return layer(input_layer)

# ---------------------------------------------------------------------


def builder():
    # TODO
    pass

# ---------------------------------------------------------------------

