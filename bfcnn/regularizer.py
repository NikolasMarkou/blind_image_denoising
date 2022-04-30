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
    def __init__(self, lambda_coefficient: float = 1.0):
        self._lambda_coefficient = lambda_coefficient

    def __call__(self, x):
        # --- argument checking
        if x is None:
            raise ValueError("input cannot be empty")

        # reshape to 2d matrix
        if x.shape == 2:
            x_reshaped = x
        elif x.shape == 4:
            # reshape x which is 4d to 2d
            x_transpose = np.transpose(x, axes=(3, 0, 1, 2))
            x_transpose_shape = x_transpose.shape
            x_reshaped = \
                np.reshape(
                    x_transpose,
                    newshape=(
                        x_transpose_shape[0],
                        np.prod(x_transpose_shape[1:])))
        else:
            logger.info(f"don't know how to handle shape [{x.shape}]")
            return 0.0

        # compute (Wt * W) - I
        wt_w = \
            np.matmul(
                np.transpose(x_reshaped, axes=(1, 0)),
                x_reshaped)
        # frobenius norm
        return \
            self._lambda_coefficient * \
            np.linalg.cond(wt_w - np.identity(wt_w.shape[0]), "fro")

    def get_config(self):
        return {
            "lambda_coefficient": self._lambda_coefficient
        }

# ---------------------------------------------------------------------


def builder(config: Dict) -> Any:
    """
    build a regularizing function

    :param config:
    :return:
    """
    if config is None:
        return None
    regularizer_type = config.get("type", None).lower()
    regularizer_parameters = config.get("parameters", {})
    if regularizer_type is None:
        return None
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

