r"""
blocks and builders for custom regularizers
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
from typing import Dict, Tuple, Union, List, Any

# ---------------------------------------------------------------------

from .custom_logger import logger


# ---------------------------------------------------------------------


class SoftOrthogonalConstraintRegularizer(keras.regularizers.Regularizer):
    def __init__(self,
                 lambda_coefficient: float = 1.0,
                 l1_coefficient: float = 0.01):
        self._lambda_coefficient = lambda_coefficient
        self._l1_coefficient = l1_coefficient

    def __call__(self, x):
        # --- argument checking
        if x is None:
            raise ValueError("input cannot be empty")

        # reshape to 2d matrix
        if len(x.shape) == 2:
            x_reshaped = x
        elif len(x.shape) == 4:
            # reshape x which is 4d to 2d
            x_transpose = tf.transpose(x, axes=(3, 0, 1, 2))
            x_reshaped = tf.reshape(x_transpose, shape=(tf.shape(x_transpose)[0], -1))
        else:
            logger.info(f"don't know how to handle shape [{x.shape}]")
            return 0.0

        # compute (Wt * W) - I
        wt_w = \
            tf.linalg.matmul(
                tf.transpose(x_reshaped, axes=(1, 0)),
                x_reshaped)
        # frobenius norm
        return \
            self._lambda_coefficient * \
            tf.norm(wt_w - tf.eye(tf.shape(wt_w)), ord="fro", axis=(0, 1), keepdims=False) + \
            self._l1_coefficient * tf.reduce_sum(tf.abs(wt_w), axis=None, keepdims=False)

    def get_config(self):
        return {
            "l1_coefficient": self._l1_coefficient,
            "lambda_coefficient": self._lambda_coefficient
        }


# ---------------------------------------------------------------------


def builder(
        config: Union[str, Dict]) -> Any:
    """
    build a regularizing function

    :param config:
    :return:
    """
    # --- argument checking
    if config is None:
        raise ValueError("config cannot be None")

    # --- prepare variables
    if isinstance(config, str):
        regularizer_type = config.lower()
        regularizer_parameters = {}
    else:
        regularizer_type = config.get("type", None).lower()
        regularizer_parameters = config.get("parameters", {})

    # --- select correct regularizer
    if regularizer_type == "l1":
        return keras.regularizers.L1(**regularizer_parameters)
    if regularizer_type == "l2":
        return keras.regularizers.L2(**regularizer_parameters)
    if regularizer_type == "l1l2":
        return keras.regularizers.L1L2(**regularizer_parameters)
    if regularizer_type == "soft_orthogonal":
        return SoftOrthogonalConstraintRegularizer(**regularizer_parameters)
    raise ValueError(f"don't know how to handle [{regularizer_type}]")

# ---------------------------------------------------------------------
