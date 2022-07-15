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

# define file constants
REGULARIZERS_STR = "regularizers"
L1_COEFFICIENT_STR = "l1_coefficient"
LAMBDA_COEFFICIENT_STR = "lambda_coefficient"


# ---------------------------------------------------------------------


class RegularizationType(Enum):
    L1 = 0

    L2 = 1

    L1L2 = 2

    SOFT_ORTHONORMAL = 3

    SOFT_ORTHOGONAL = 4

    @staticmethod
    def from_string(type_str: str) -> "RegularizationType":
        # --- argument checking
        if type_str is None:
            raise ValueError("type_str must not be null")
        if not isinstance(type_str, str):
            raise ValueError("type_str must be string")
        type_str = type_str.strip().upper()
        if len(type_str) <= 0:
            raise ValueError("stripped type_str must not be empty")

        # --- clean string and get
        return RegularizationType[type_str]

    def to_string(self) -> str:
        return self.name


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
        x_reshaped = \
            tf.reshape(
                x_transpose,
                shape=(tf.shape(x_transpose)[0], -1))
    else:
        raise ValueError(f"don't know how to handle shape [{x.shape}]")
    return x_reshaped


# ---------------------------------------------------------------------


class SoftOrthonormalConstraintRegularizer(keras.regularizers.Regularizer):
    """
    Implements the soft orthogonality constraint as described in
    Can We Gain More from Orthogonality Regularizations in Training Deep CNNs?
    https://arxiv.org/abs/1810.09102

    This constraint forces the kernels
    to be orthogonal and have a l2 norm of 1 (orthonormal)
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
            L1_COEFFICIENT_STR: self._l1_coefficient,
            LAMBDA_COEFFICIENT_STR: self._lambda_coefficient
        }


# ---------------------------------------------------------------------


class SoftOrthogonalConstraintRegularizer(keras.regularizers.Regularizer):
    """
    Implements the soft orthogonality constraint as described in
    Can We Gain More from Orthogonality Regularizations in Training Deep CNNs?
    https://arxiv.org/abs/1810.09102

    This constraint forces the kernels
    to be orthogonal and have a l2 norm of whatever they want (orthogonal)
    but subject to l1 constraint
    """

    def __init__(self,
                 lambda_coefficient: float = 1.0,
                 l1_coefficient: float = 0.001):
        self._lambda_coefficient = lambda_coefficient
        self._l1_coefficient = l1_coefficient

    def __call__(self, x):
        # --- reshape
        x = reshape_to_2d(x)

        # --- compute (Wt * W)
        wt_w = \
            tf.linalg.matmul(
                tf.transpose(x, perm=(1, 0)),
                x)
        # mask diagonal
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
            tf.reduce_sum(tf.abs(wt_w_masked), axis=None, keepdims=False)

    def get_config(self):
        return {
            L1_COEFFICIENT_STR: self._l1_coefficient,
            LAMBDA_COEFFICIENT_STR: self._lambda_coefficient
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
            REGULARIZERS_STR: [
                regularizer.get_config() for
                regularizer in self._regularizers
            ]
        }


# ---------------------------------------------------------------------


def builder_helper(
        config: Union[str, Dict],
        verbose: bool = False) -> keras.regularizers.Regularizer:
    """
    build a single regularizing function

    :param config:
    :param verbose: if True show extra messages
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

    # --- logging
    if verbose:
        logger.info(f"building configuration with config [{config}")

    # --- select correct regularizer
    regularizer = None
    regularizer_type = RegularizationType.from_string(regularizer_type)
    if regularizer_type == RegularizationType.L1:
        regularizer = keras.regularizers.L1(**regularizer_params)
    elif regularizer_type == RegularizationType.L2:
        regularizer = keras.regularizers.L2(**regularizer_params)
    elif regularizer_type == RegularizationType.L1L2:
        regularizer = keras.regularizers.L1L2(**regularizer_params)
    elif regularizer_type == RegularizationType.SOFT_ORTHONORMAL:
        regularizer = SoftOrthonormalConstraintRegularizer(**regularizer_params)
    elif regularizer_type == RegularizationType.SOFT_ORTHOGONAL:
        regularizer = SoftOrthogonalConstraintRegularizer(**regularizer_params)
    else:
        raise ValueError(f"don't know how to handle [{regularizer_type}]")
    return regularizer


# ---------------------------------------------------------------------


def builder(
        config: Union[str, Dict, List]) -> keras.regularizers.Regularizer:
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
