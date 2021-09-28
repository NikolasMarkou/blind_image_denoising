r"""Constructs the loss function of the blind image denoising"""

# ---------------------------------------------------------------------

__author__ = "Nikolas Markou"
__version__ = "0.1.0"
__license__ = "None"

# ---------------------------------------------------------------------

import tensorflow as tf
from typing import List, Dict, Callable

# ---------------------------------------------------------------------


def loss_function_builder(
        config: Dict) -> Callable:
    """
    Constructs the loss function of the depth prediction model
    """
    # controls how we discount each level
    mae_multiplier = config.get("mae_multiplier", 1.0)
    mre_multiplier = config.get("mre_multiplier", 1.0)
    regularization_multiplier = config.get("regularization", 1.0)

    def loss_function(
            input_batch,
            prediction_batch,
            noisy_batch=None,
            model_losses=None) -> Dict:
        """
        The loss function of the depth prediction model

        :param: input_batch: ground truth
        :param: prediction_batch: prediction
        :param: model_losses: weight/regularization losses
        :return loss
        """
        # --- mean absolute error from prediction
        mean_absolute_error_loss = 0.0
        if input_batch is not None and prediction_batch is not None:
            diff = tf.abs(input_batch - prediction_batch)
            diff = tf.reduce_mean(diff, axis=[1, 2, 3])
            mean_absolute_error_loss = tf.reduce_mean(diff, axis=[0])

        # --- mean relative error from prediction
        mean_relative_error_loss = 0.0
        if input_batch is not None and prediction_batch is not None:
            diff = tf.abs(input_batch - prediction_batch)
            diff_min = tf.reduce_min(diff, axis=[1, 2], keepdims=True)
            diff_mean = tf.reduce_mean(diff, axis=[1, 2], keepdims=True)
            diff_relative = 255 * ((diff - diff_min) / (diff_mean + 0.00001))
            diff_relative = \
                tf.reduce_mean(diff_relative, axis=[1, 2, 3])
            mean_relative_error_loss = \
                tf.reduce_mean(diff_relative, axis=[0])

        # --- mean absolute error from noisy
        mean_absolute_error_noise = 0.0
        if input_batch is not None and noisy_batch is not None:
            diff = tf.abs(input_batch - noisy_batch)
            diff = tf.reduce_mean(diff, axis=[1, 2, 3])
            mean_absolute_error_noise = tf.reduce_mean(diff, axis=[0])

        mean_absolute_error_improvement = \
            mean_absolute_error_noise - mean_absolute_error_loss

        # --- regularization error
        regularization_loss = 0.0
        if model_losses is not None:
            regularization_loss = tf.add_n(model_losses)

        # --- add up loss
        mean_total_loss = \
            mean_absolute_error_loss * mae_multiplier + \
            regularization_loss * regularization_multiplier + \
            mean_relative_error_loss * mre_multiplier

        return {
            "mean_total_loss": mean_total_loss,
            "mae_loss": mean_absolute_error_loss,
            "mre_loss": mean_relative_error_loss,
            "mae_noise": mean_absolute_error_noise,
            "regularization_loss": regularization_loss,
            "mae_improvement": mean_absolute_error_improvement
        }

    return loss_function

# ---------------------------------------------------------------------
