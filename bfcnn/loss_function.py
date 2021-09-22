r"""Constructs the loss function of the blind image denoising"""

# ---------------------------------------------------------------------

__author__ = "Nikolas Markou"
__version__ = "0.1.0"
__license__ = "None"

# ---------------------------------------------------------------------

from enum import Enum
from typing import List
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend as K

# ---------------------------------------------------------------------


class Loss(Enum):
    MeanAbsoluteError = 1
    MeanSquareError = 2

# ---------------------------------------------------------------------


def mae_loss(y_true, y_pred):
    tmp_pixels = K.abs(y_true[0] - y_pred[0])
    return K.mean(tmp_pixels)

# ---------------------------------------------------------------------


def mse_loss(y_true, y_pred):
    tmp_pixels = K.pow(y_true[0] - y_pred[0], 2.0)
    return K.mean(tmp_pixels)

# ---------------------------------------------------------------------


def build_loss_fn(loss: List[Loss]):
    # --- argument checking
    if len(loss) <= 0:
        raise ValueError("loss list cannot be empty")

    # --- define loss functions
    loss_fn = []

    if Loss.MeanSquareError in loss:
        loss_fn.append(mse_loss)

    if Loss.MeanAbsoluteError in loss:
        loss_fn.append(mae_loss)

    # --- total loss
    def total_loss(y_true, y_pred):
        result = 0.0
        for fn in loss_fn:
            result = result + fn(y_true, y_pred)
        return result

    return total_loss, loss_fn

# ---------------------------------------------------------------------


def loss_function_builder(config):
    """
    Constructs the loss function of the depth prediction model
    """
    # controls how we discount each level
    mae_multiplier = config.get("mae_multiplier", 1.0)
    regularization_multiplier = config.get("regularization", 1.0)

    def loss_function(
            ground_truth,
            prediction,
            model_losses=None):
        """
        The loss function of the depth prediction model

        Args:
            ground_truth: ground truth
            prediction: prediction
            model_losses: weight/regularization losses
        Returns:
        """
        # ---
        mean_absolute_error_loss = 0.0
        if ground_truth is not None and prediction is not None:
            diff = tf.abs(ground_truth - prediction)
            diff_sum = tf.reduce_sum(diff, axis=[1, 2, 3])
            mean_absolute_error_loss = tf.reduce_mean(diff_sum, axis=[0])
        # ---
        regularization_loss = 0.0
        if model_losses is not None:
            regularization_loss = tf.add_n(model_losses)

        # ---
        mean_total_loss = \
            mean_absolute_error_loss * mae_multiplier + \
            regularization_loss * regularization_multiplier

        return {
            "mean_total_loss": mean_total_loss,
            "regularization_loss": regularization_loss,
            "mean_absolute_error_loss": mean_absolute_error_loss,
        }

    return loss_function

# ---------------------------------------------------------------------
