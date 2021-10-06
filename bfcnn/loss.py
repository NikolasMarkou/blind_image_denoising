r"""Constructs the loss function of the blind image denoising"""

# ---------------------------------------------------------------------

__author__ = "Nikolas Markou"
__version__ = "0.1.0"
__license__ = "None"

# ---------------------------------------------------------------------

import tensorflow as tf
from tensorflow import keras
from typing import List, Dict, Callable

# ---------------------------------------------------------------------


def loss_function_builder(
        config: Dict) -> Callable:
    """
    Constructs the loss function of the depth prediction model
    """
    # controls how we discount each level
    hinge = config.get("hinge", 0.0)
    mae_multiplier = config.get("mae_multiplier", 1.0)
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
        mae_prediction_loss = 0.0
        if input_batch is not None and prediction_batch is not None:
            diff = tf.abs(input_batch - prediction_batch)
            diff = keras.layers.ReLU(threshold=hinge)(diff)
            diff = tf.reduce_mean(diff, axis=[1, 2, 3])
            mae_prediction_loss = tf.reduce_mean(diff, axis=[0])

        # --- mean absolute error from noisy
        mae_noisy = 0.0
        if input_batch is not None and noisy_batch is not None:
            diff = tf.abs(input_batch - noisy_batch)
            diff = keras.layers.ReLU(threshold=hinge)(diff)
            diff = tf.reduce_mean(diff, axis=[1, 2, 3])
            mae_noisy = tf.reduce_mean(diff, axis=[0])

        mae_improvement = \
            mae_noisy - mae_prediction_loss

        # --- regularization error
        regularization_loss = 0.0
        if model_losses is not None:
            regularization_loss = tf.add_n(model_losses)

        # --- add up loss
        mean_total_loss = \
            mae_prediction_loss * mae_multiplier + \
            regularization_loss * regularization_multiplier

        return {
            "mae_noise": mae_noisy,
            "mae_loss": mae_prediction_loss,
            "mean_total_loss": mean_total_loss,
            "regularization_loss": regularization_loss,
            "mae_improvement": mae_improvement
        }

    return loss_function

# ---------------------------------------------------------------------
