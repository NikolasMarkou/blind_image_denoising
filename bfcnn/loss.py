r"""Constructs the loss function of the blind image denoising"""

# ---------------------------------------------------------------------

__author__ = "Nikolas Markou"
__version__ = "1.0.0"
__license__ = "MIT"

# ---------------------------------------------------------------------

import tensorflow as tf
from tensorflow import keras
from typing import List, Dict, Callable

# ---------------------------------------------------------------------

DEFAULT_EPSILON = 0.0001

# ---------------------------------------------------------------------


def snr(
        original,
        prediction,
        multiplier: float = 10.0,
        base: float = 10.0):
    """
    Signal-to-noise ratio expressed in dB
    """
    d_2 = tf.square(original - prediction)
    # sum over all dims
    d_2 = tf.reduce_sum(d_2, axis=[1, 2, 3])
    d_prediction = tf.reduce_sum(prediction, axis=[1, 2, 3])
    # mean over batch
    result = \
        (tf.reduce_mean(d_prediction, axis=[0]) + DEFAULT_EPSILON) / \
        (tf.reduce_mean(d_2, axis=[0]) + DEFAULT_EPSILON)
    return multiplier * tf.math.log(result) / tf.math.log(base)


# ---------------------------------------------------------------------


def mae(
        original,
        prediction,
        hinge: float = 0):
    """
    Mean Absolute Error (mean over channels and batches)

    :param original:
    :param prediction:
    :param hinge: hinge value
    """
    d = tf.abs(original - prediction)
    if hinge != 0.0:
        d = keras.layers.ReLU(threshold=hinge)(d)
    # sum over all dims
    d = tf.reduce_mean(d, axis=[1, 2, 3])
    # mean over batch
    loss = tf.reduce_mean(d, axis=[0])
    return loss


# ---------------------------------------------------------------------


def nae(
        original,
        prediction,
        hinge: float = 0):
    """
    Normalized Absolute Error
    (sum over width, height, channel and mean over batches)

    :param original:
    :param prediction:
    :param hinge: hinge value
    """
    d = tf.abs(original - prediction)
    if hinge != 0.0:
        d = keras.layers.ReLU(threshold=hinge)(d)
    # sum over all dims
    d = tf.reduce_sum(d, axis=[1, 2, 3])
    d_x = tf.reduce_sum(original, axis=[1, 2, 3])
    # mean over batch
    loss = \
        tf.reduce_mean(d, axis=[0]) / \
        (tf.reduce_mean(d_x, axis=[0]) + DEFAULT_EPSILON)
    return loss


# ---------------------------------------------------------------------


def loss_function_builder(
        config: Dict) -> Callable:
    """
    Constructs the loss function of the depth prediction model
    """
    # controls how we discount each level
    hinge = config.get("hinge", 0.0)
    nae_multiplier = config.get("nae_multiplier", 0.0)
    mae_multiplier = config.get("mae_multiplier", 1.0)
    regularization_multiplier = config.get("regularization", 1.0)

    def loss_function(
            input_batch,
            prediction_batch,
            noisy_batch=None,
            model_losses=None,
            difficulty: float = -1.0) -> Dict:
        """
        The loss function of the depth prediction model

        :param: input_batch: ground truth
        :param: prediction_batch: prediction
        :param: model_losses: weight/regularization losses
        :param: difficulty:
            if >= 0 then it is an indication how corrupted the noisy batch is
        :return loss
        """

        # --- mean absolute error from prediction
        mae_prediction_loss = \
            mae(input_batch, prediction_batch, hinge)

        # ---
        nae_prediction = \
            nae(input_batch, prediction_batch, hinge)

        nae_noise = \
            nae(input_batch, noisy_batch, hinge)

        nae_improvement = nae_noise - nae_prediction

        # --- regularization error
        regularization_loss = 0.0
        if model_losses is not None:
            regularization_loss = tf.add_n(model_losses)

        # --- snr
        signal_to_noise_ratio = \
            snr(input_batch, prediction_batch)

        # --- difficulty computation
        if difficulty >= 0:
            # TODO
            pass

        # --- add up loss
        mean_total_loss = \
            nae_prediction * nae_multiplier + \
            mae_prediction_loss * mae_multiplier + \
            regularization_loss * regularization_multiplier

        return {
            "nae_noise": nae_noise,
            "snr": signal_to_noise_ratio,
            "mae_loss": mae_prediction_loss,
            "nae_prediction": nae_prediction,
            "mean_total_loss": mean_total_loss,
            "nae_improvement": nae_improvement,
            "regularization_loss": regularization_loss
        }

    return loss_function

# ---------------------------------------------------------------------
