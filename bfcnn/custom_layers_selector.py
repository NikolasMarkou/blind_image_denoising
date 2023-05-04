import copy
from enum import Enum
import tensorflow as tf
from tensorflow import keras
from typing import List, Tuple, Union, Dict, Iterable

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .custom_logger import logger
from .constants import *
from .custom_layers import \
    DifferentiableGateLayer
from .utilities import \
    dense_wrapper, \
    conv2d_wrapper, \
    min_max_mean_sigma_block


# ---------------------------------------------------------------------


class SelectorType(Enum):
    PIXEL = 0

    CHANNEL = 1

    BATCH = 2

    @staticmethod
    def from_string(type_str: str) -> "SelectorType":
        # --- argument checking
        if type_str is None:
            raise ValueError("type_str must not be null")
        if not isinstance(type_str, str):
            raise ValueError("type_str must be string")
        type_str = type_str.strip().upper()
        if len(type_str) <= 0:
            raise ValueError("stripped type_str must not be empty")

        # --- clean string and get
        return SelectorType[type_str]

    def to_string(self) -> str:
        return self.name

# ---------------------------------------------------------------------


class ActivationType(Enum):
    SOFT = 0

    HARD = 1

    @staticmethod
    def from_string(type_str: str) -> "ActivationType":
        # --- argument checking
        if type_str is None:
            raise ValueError("type_str must not be null")
        if not isinstance(type_str, str):
            raise ValueError("type_str must be string")
        type_str = type_str.strip().upper()
        if len(type_str) <= 0:
            raise ValueError("stripped type_str must not be empty")

        # --- clean string and get
        return ActivationType[type_str]

    def to_string(self) -> str:
        return self.name


# ---------------------------------------------------------------------


def selector_block(
        input_1_layer,
        input_2_layer,
        selector_layer,
        selector_type: SelectorType,
        activation_type: ActivationType,
        filters_compress: int,
        kernel_regularizer: str = "l1",
        kernel_initializer: str = "glorot_normal",
        **kwargs):
    """
    from 2 input layers,
    select a combination of the 2 with bias on the first one

    :return: filtered input_layer
    """
    # --- argument checking
    # TODO

    # --- setup network
    x = selector_layer
    filters_target = \
        tf.keras.backend.int_shape(input_1_layer)[-1]

    if selector_type == SelectorType.PIXEL:
        # ---
        selector_conv_0_params = dict(
            filters=filters_compress,
            kernel_size=1,
            use_bias=False,
            activation="relu",
            kernel_regularizer=kernel_regularizer,
            kernel_initializer=kernel_initializer)

        selector_conv_1_params = dict(
            filters=filters_target,
            kernel_size=1,
            use_bias=False,
            activation="relu",
            kernel_regularizer=kernel_regularizer,
            kernel_initializer=kernel_initializer)

        x = conv2d_wrapper(
            input_layer=x,
            conv_params=selector_conv_0_params,
            bn_params=None)

        x = conv2d_wrapper(
            input_layer=x,
            conv_params=selector_conv_1_params,
            bn_params=None)
    elif selector_type == SelectorType.CHANNEL or \
            selector_type == SelectorType.BATCH:
        # ---
        # out squeeze and excite gating does not use global avg
        # followed by dense layer, because we are using this on large images
        # global averaging looses too much information
        selector_dense_0_params = dict(
            units=filters_compress,
            use_bias=False,
            activation="relu",
            kernel_regularizer=kernel_regularizer,
            kernel_initializer=kernel_initializer)

        selector_dense_1_params = dict(
            units=filters_target,
            use_bias=False,
            activation="relu",
            kernel_regularizer=kernel_regularizer,
            kernel_initializer=kernel_initializer)

        # transformation
        x = tf.reduce_mean(x, axis=[1, 2], keepdims=False)

        x = dense_wrapper(
            input_layer=x,
            dense_params=selector_dense_0_params,
            bn_params=None)

        x = dense_wrapper(
            input_layer=x,
            dense_params=selector_dense_1_params,
            bn_params=None)

        x = tf.expand_dims(x, axis=1)
        x = tf.expand_dims(x, axis=2)
    else:
        raise ValueError(f"don't know how to handle this [{selector_type}]")

    # --- attach activation head
    if activation_type == ActivationType.SOFT:
        x = tf.keras.layers.Activation("sigmoid")(2.5 - x)
    elif activation_type == ActivationType.HARD:
        x = tf.keras.layers.Activation("hard_sigmoid")(2.5 - x)
    else:
        raise ValueError(f"dont know how to handle this [{activation_type}]")

    return \
        tf.keras.layers.Multiply()([input_1_layer, x]) + \
        tf.keras.layers.Multiply()([input_2_layer, 1.0 - x])

# ---------------------------------------------------------------------

