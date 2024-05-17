r"""
blocks and builders for custom regularizers
"""

# ---------------------------------------------------------------------

import tensorflow as tf
from typing import Dict, Tuple, Union, List, Any

# ---------------------------------------------------------------------

from .constants import *


# ---------------------------------------------------------------------


@tf.function(
    input_signature=[
        tf.TensorSpec(shape=(None, None),
                      dtype=tf.float32)])
def reshape_2d_to_2d(w: tf.Tensor):
    return tf.transpose(w, perm=(1, 0))


@tf.function(
    input_signature=[
        tf.TensorSpec(shape=(None, None, None, None),
                      dtype=tf.float32)])
def reshape_4d_to_2d(w: tf.Tensor) -> tf.Tensor:
    w_t = \
        tf.transpose(
            w, perm=(3, 0, 1, 2))
    return \
        tf.reshape(
            w_t,
            shape=(tf.shape(w_t)[0], -1))


def reshape_to_2d(weights: tf.Tensor) -> tf.Tensor:
    rank = len(weights.shape)
    if rank == 2:
        return reshape_2d_to_2d(weights)
    if rank == 4:
        return reshape_4d_to_2d(weights)
    return weights


# ---------------------------------------------------------------------


def wt_x_w(weights: tf.Tensor) -> tf.Tensor:
    # --- reshape
    wt = reshape_to_2d(weights)

    # --- compute (Wt * W)
    wt_w = \
        tf.linalg.matmul(
            wt,
            tf.transpose(wt, perm=(1, 0)))

    return wt_w


# ---------------------------------------------------------------------

@tf.keras.utils.register_keras_serializable()
class SoftOrthogonalConstraintRegularizer(tf.keras.regularizers.Regularizer):
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
                 l1_coefficient: float = 0.01,
                 l2_coefficient: float = 0.00,
                 **kwargs):
        super().__init__(**kwargs)
        self._lambda_coefficient = lambda_coefficient
        self._l1_coefficient = l1_coefficient
        self._l2_coefficient = l2_coefficient
        self._call_fn = None

    def generic_fn(self, x):
        # --- compute (Wt * W)
        wt_w = wt_x_w(x)

        # --- mask diagonal
        wt_w_masked = \
            tf.math.multiply(wt_w, 1.0 - tf.eye(tf.shape(wt_w)[0]))

        # --- init result
        result = tf.constant(0.0, dtype=tf.float32)

        # --- frobenius norm
        if self._lambda_coefficient > 0.0:
            result += \
                self._lambda_coefficient * \
                tf.square(
                    tf.norm(wt_w_masked,
                            ord="fro",
                            axis=(0, 1),
                            keepdims=False))

        # --- l1 on Wt_W
        if self._l1_coefficient > 0.0:
            result += \
                self._l1_coefficient * \
                tf.reduce_sum(tf.abs(wt_w_masked), axis=None, keepdims=False)

        # --- l2 on Wt_W
        if self._l2_coefficient > 0.0:
            result += \
                self._l2_coefficient * \
                tf.reduce_sum(tf.pow(wt_w_masked, 2.0), axis=None, keepdims=False)

        return result

    def __call__(self, x):
        if self._call_fn is None:
            self._call_fn = (
                tf.function(func=self.generic_fn,
                            reduce_retracing=True).get_concrete_function(x))
        return self._call_fn(x)


# ---------------------------------------------------------------------

@tf.keras.utils.register_keras_serializable()
class SoftOrthonormalConstraintRegularizer(tf.keras.regularizers.Regularizer):
    """
    Implements the soft orthogonality constraint as described in
    Can We Gain More from Orthogonality Regularizations in Training Deep CNNs?
    https://arxiv.org/abs/1810.09102

    This constraint forces the kernels
    to be orthogonal and have a l2 norm of 1 (orthonormal)
    """

    def __init__(self,
                 lambda_coefficient: float = 1.0,
                 l1_coefficient: float = 0.001,
                 l2_coefficient: float = 0.00,
                 **kwargs):
        super().__init__(**kwargs)
        self._lambda_coefficient = lambda_coefficient
        self._l1_coefficient = l1_coefficient
        self._l2_coefficient = l2_coefficient

    def __call__(self, x):
        # --- compute (Wt * W)
        wt_w = wt_x_w(x)
        w_shape = tf.shape(wt_w)
        i = tf.eye(num_rows=w_shape[0])

        # --- init result
        result = tf.constant(0.0, dtype=tf.float32)

        # --- frobenius norm
        if self._lambda_coefficient > 0.0:
            result += \
                self._lambda_coefficient * \
                tf.square(
                    tf.norm(wt_w - i,
                            ord="fro",
                            axis=(0, 1),
                            keepdims=False))

        # --- l1 on Wt_W
        if self._l1_coefficient > 0.0:
            result += \
                self._l1_coefficient * \
                tf.reduce_sum(tf.abs(wt_w), axis=None, keepdims=False)

        # --- l2 on Wt_W
        if self._l2_coefficient > 0.0:
            result += \
                self._l2_coefficient * \
                tf.reduce_sum(tf.pow(wt_w, 2.0), axis=None, keepdims=False)

        return result


# ---------------------------------------------------------------------

@tf.keras.utils.register_keras_serializable()
class L1StickyRegularizer(tf.keras.regularizers.Regularizer):
    """
    https://www.desmos.com/calculator/ft2si1rv8z
    """
    def __init__(self,
                 l1: float = 0.1,
                 **kwargs):
        self.regularizer = tf.keras.regularizers.l1(l1=l1)

    def __call__(self, x):
        x_composite = tf.math.abs(-tf.math.abs(x+0.5) + 0.5)
        return self.regularizer(x_composite)

# ---------------------------------------------------------------------
