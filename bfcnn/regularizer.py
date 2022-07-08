r"""
blocks and builders for custom regularizers
"""

# ---------------------------------------------------------------------

import numpy as np
from enum import Enum
import tensorflow as tf
from tensorflow import keras
from typing import Dict, Tuple, Union, List, Any

# ---------------------------------------------------------------------

from .custom_logger import logger
from .constants import CONFIG_STR, TYPE_STR


# ---------------------------------------------------------------------

def reshape_to_2d(x):
    # --- argument checking
    if x is None:
        raise ValueError("input cannot be empty")

    # --- reshape to 2d matrix
    if len(x.shape) == 2:
        x_reshaped = x
    elif len(x.shape) == 4:
        # reshape x which is 4d to 2d
        x_transpose = tf.transpose(x, perm=(3, 0, 1, 2))
        x_reshaped = tf.reshape(x_transpose, shape=(tf.shape(x_transpose)[0], -1))
    else:
        raise ValueError(f"don't know how to handle shape [{x.shape}]")
    return x_reshaped


# ---------------------------------------------------------------------


class SoftOrthogonalConstraintRegularizer(keras.regularizers.Regularizer):
    """
    Implements the soft orthogonality constraint as described in
    Can We Gain More from Orthogonality Regularizations in Training Deep CNNs?
    https://arxiv.org/abs/1810.09102
    """

    def __init__(self,
                 lambda_coefficient: float = 1.0,
                 l1_coefficient: float = 0.001):
        self._lambda_coefficient = lambda_coefficient
        self._l1_coefficient = l1_coefficient

    def __call__(self, x):
        # --- reshape
        x = reshape_to_2d(x)

        # --- compute (Wt * W) - I
        wt_w = \
            tf.linalg.matmul(
                tf.transpose(x, perm=(1, 0)),
                x)

        # frobenius norm
        return \
            self._lambda_coefficient * \
            tf.square(
                tf.norm(wt_w - tf.eye(tf.shape(wt_w)[0]),
                        ord="fro",
                        axis=(0, 1),
                        keepdims=False)) + \
            self._l1_coefficient * \
            tf.reduce_sum(tf.abs(wt_w), axis=None, keepdims=False)

    def get_config(self):
        return {
            "l1_coefficient": self._l1_coefficient,
            "lambda_coefficient": self._lambda_coefficient
        }


# ---------------------------------------------------------------------


class SoftOrthogonalCustomConstraintRegularizer(keras.regularizers.Regularizer):
    """
    Implements the soft orthogonality constraint as described in
    Can We Gain More from Orthogonality Regularizations in Training Deep CNNs?
    https://arxiv.org/abs/1810.09102

    I changed the original a bit so the diagonal doesnt need to be necessarily ones.
    """

    def __init__(self,
                 lambda_coefficient: float = 1.0,
                 l1_coefficient: float = 0.001):
        self._lambda_coefficient = lambda_coefficient
        self._l1_coefficient = l1_coefficient

    def __call__(self, x):
        # --- reshape
        x = reshape_to_2d(x)

        # --- compute (Wt * W) - I
        wt_w = \
            tf.linalg.matmul(
                tf.transpose(x, perm=(1, 0)),
                x)
        wt_w_mask = tf.ones(tf.shape(wt_w)[0]) - tf.eye(tf.shape(wt_w)[0])
        wt_w_masked = tf.math.multiply(wt_w, wt_w_mask)

        # frobenius norm
        return \
            self._lambda_coefficient * \
            tf.square(
                tf.norm(wt_w_masked,
                        ord="fro",
                        axis=(0, 1),
                        keepdims=False)) + \
            self._l1_coefficient * \
            tf.reduce_sum(tf.abs(wt_w), axis=None, keepdims=False)

    def get_config(self):
        return {
            "l1_coefficient": self._l1_coefficient,
            "lambda_coefficient": self._lambda_coefficient
        }


# ---------------------------------------------------------------------


class RegularizerMixer(keras.regularizers.Regularizer):
    """
    Combines regularizers
    """

    def __init__(self,
                 regularizers: List[keras.regularizers.Regularizer]):
        self._regularizers = regularizers

    def __call__(self, x):
        result_regularizers = None

        for regularizer in self._regularizers:
            r = regularizer(x)
            if result_regularizers is None:
                result_regularizers = r
            else:
                result_regularizers += r
        return result_regularizers

    def get_config(self):
        return {
            "regularizers": [
                regularizer.get_config() for
                regularizer in self._regularizers
            ]
        }


# ---------------------------------------------------------------------


def builder_helper(
        config: Union[str, Dict]) -> keras.regularizers.Regularizer:
    """
    build a single regularizing function

    :param config:
    :return:
    """
    # --- argument checking
    if config is None:
        raise ValueError("config cannot be None")

    # --- prepare variables
    if isinstance(config, str):
        regularizer_type = config.lower()
        regularizer_params = {}
    elif isinstance(config, Dict):
        regularizer_type = config.get(TYPE_STR, None).lower()
        regularizer_params = config.get(CONFIG_STR, {})
    else:
        raise ValueError("don't know how to handle config")

    # --- select correct regularizer
    regularizer = None
    if regularizer_type == "l1":
        regularizer = keras.regularizers.L1(**regularizer_params)
    elif regularizer_type == "l2":
        regularizer = keras.regularizers.L2(**regularizer_params)
    elif regularizer_type == "l1l2":
        regularizer = keras.regularizers.L1L2(**regularizer_params)
    elif regularizer_type == "soft_orthogonal":
        regularizer = SoftOrthogonalConstraintRegularizer(**regularizer_params)
    elif regularizer_type == "soft_orthogonal_custom":
        regularizer = SoftOrthogonalCustomConstraintRegularizer(**regularizer_params)
    else:
        raise ValueError(f"don't know how to handle [{regularizer_type}]")
    return regularizer


# ---------------------------------------------------------------------


def builder(
        config: Union[str, Dict, List]) -> Any:
    """
    build a single or mixed regularization function

    :param config:
    :return:
    """
    # --- argument checking
    if config is None:
        raise ValueError("config cannot be None")

    # --- prepare variables
    if isinstance(config, List):
        regularizers = [
            builder_helper(config=r) for r in config
        ]
    else:
        return builder_helper(config=config)

    # --- mixes all the regularizes together
    return RegularizerMixer(regularizers=regularizers)

# ---------------------------------------------------------------------
