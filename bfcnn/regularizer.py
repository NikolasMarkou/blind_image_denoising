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
from typing import Dict, Tuple, Union, List

# ---------------------------------------------------------------------

from .custom_logger import logger

# ---------------------------------------------------------------------


class SoftOrthogonalConstraintRegularizer(keras.regularizers.Regularizer):
    def __init__(self, lambda_coefficient: float = 1.0):
        self._lambda_coefficient = lambda_coefficient

    def __call__(self, x):
        assert len(x.shape) == 4
        # reshape x which is 4d to 2d
        x_transpose = np.transpose(x, axes=(3, 0, 1, 2))
        x_transpose_shape = x_transpose.shape
        x_reshaped = \
            np.reshape(
                x_transpose,
                newshape=(
                    x_transpose_shape[0],
                    np.prod(x_transpose_shape[1:])))
        # compute Wt * W - I
        Wt_W = \
            np.transpose(x_reshaped, axes=(1, 0)) * x_reshaped
        I = np.identity(Wt_W.shape[0])
        # frobenius norm
        return self._lambda_coefficient * np.linalg.cond(Wt_W - I, "fro")

    def get_config(self):
        return {
            "lambda_coefficient": self._lambda_coefficient
        }

# ---------------------------------------------------------------------

