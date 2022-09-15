r"""Constructs the loss function of the blind image denoising"""

# ---------------------------------------------------------------------

import tensorflow as tf
from tensorflow import keras
from typing import List, Dict, Callable

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .constants import *
from .custom_logger import logger
from .delta import delta_xy_magnitude


# ---------------------------------------------------------------------


def snr(
        original,
        prediction,
        multiplier: float = 10.0,
        base: float = 10.0):
    """
    Signal-to-noise ratio expressed in dB

    :param original: original image batch
    :param prediction: denoised image batch
    :param multiplier:
    :param base: logarithm base
    """
    # mse of prediction
    d_2 = tf.square(original - prediction)
    # sum over all dims
    d_2 = tf.reduce_sum(d_2, axis=[1, 2, 3])
    d_prediction = tf.reduce_sum(prediction, axis=[1, 2, 3])
    # mean over batch
    result = \
        (tf.reduce_mean(d_prediction, axis=[0]) + EPSILON_DEFAULT) / \
        (tf.reduce_mean(d_2, axis=[0]) + EPSILON_DEFAULT)
    return multiplier * tf.math.log(result) / tf.math.log(base)


# ---------------------------------------------------------------------


def mae_weighted_delta(
        original,
        prediction,
        hinge: float = 0.0):
    """
    Mean Absolute Error (mean over channels and batches) with weights

    :param original: original image batch
    :param prediction: denoised image batch
    :param hinge: hinge value
    """
    original_delta = \
        delta_xy_magnitude(
            input_layer=original,
            kernel_size=5,
            alpha=1.0,
            beta=1.0,
            eps=EPSILON_DEFAULT)

    d_weight = \
        original_delta / \
        (tf.abs(tf.reduce_max(
            input_tensor=original_delta,
            axis=[1, 2],
            keepdims=True)) +
         EPSILON_DEFAULT)
    d_weight = tf.abs(d_weight)

    # --- calculate hinged absolute diff
    d = tf.abs(original - prediction)
    d = keras.layers.ReLU(threshold=hinge)(d)

    # --- multiply diff and weight
    d = keras.layers.Multiply()([d, d_weight])

    # --- mean over all dims
    d = tf.reduce_mean(d, axis=[1, 2, 3])

    # --- mean over batch
    loss = tf.reduce_mean(d, axis=[0])

    return loss


# ---------------------------------------------------------------------


def mae_diff(
        error,
        hinge: float = 0.0,
        cutoff: float = 255.0):
    """
    Mean Absolute Error (mean over channels and batches)

    :param error: original image batch
    :param hinge: hinge value
    :param cutoff: max value
    """
    d = tf.abs(error)
    d = keras.layers.ReLU(threshold=hinge, max_value=cutoff)(d)
    # --- mean over all dims
    d = tf.reduce_mean(d, axis=[1, 2, 3])
    # mean over batch
    return tf.reduce_mean(d, axis=[0])


# ---------------------------------------------------------------------


def mae(
        original,
        prediction,
        hinge: float = 0.0,
        cutoff: float = 255.0):
    """
    Mean Absolute Error (mean over channels and batches)

    :param original: original image batch
    :param prediction: denoised image batch
    :param hinge: hinge value
    :param cutoff: max value
    """
    error = original - prediction
    return mae_diff(error, hinge=hinge, cutoff=cutoff)


# ---------------------------------------------------------------------


def mse_diff(
        error,
        hinge: float = 0,
        cutoff: float = 255.0):
    """
    Mean Square Error (mean over channels and batches)

    :param error:
    :param hinge: hinge value
    :param cutoff: max value
    """
    d = tf.square(error)
    d = keras.layers.ReLU(threshold=hinge, max_value=cutoff)(d)
    # mean over all dims
    d = tf.reduce_mean(d, axis=[1, 2, 3])
    # mean over batch
    return tf.reduce_mean(d, axis=[0])


# ---------------------------------------------------------------------


def mse(
        original,
        prediction,
        hinge: float = 0,
        cutoff: float = (255.0 * 255.0)):
    """
    Mean Square Error (mean over channels and batches)

    :param original: original image batch
    :param prediction: denoised image batch
    :param hinge: hinge value
    :param cutoff: max value
    """
    error = tf.square(original - prediction)
    return mse_diff(error=error, hinge=hinge, cutoff=cutoff)


# ---------------------------------------------------------------------


def nae(
        original,
        prediction,
        hinge: float = 0):
    """
    Normalized Absolute Error
    (sum over width, height, channel and mean over batches)

    :param original: original image batch
    :param prediction: denoised image batch
    :param hinge: hinge value
    """
    d = tf.abs(original - prediction)
    d = keras.layers.ReLU(threshold=hinge)(d)
    # sum over all dims
    d = tf.reduce_sum(d, axis=[1, 2, 3])
    d_x = tf.reduce_sum(original, axis=[1, 2, 3])
    # mean over batch
    loss = \
        tf.reduce_mean(d, axis=[0]) / \
        (tf.reduce_mean(d_x, axis=[0]) + EPSILON_DEFAULT)
    return loss


# ---------------------------------------------------------------------


def loss_function_builder(
        config: Dict) -> Callable:
    """
    Constructs the loss function of the depth prediction model

    :param config: configuration dictionary
    :return: callable loss function
    """
    logger.info("building loss_function with config [{0}]".format(config))

    # ---
    hinge = tf.constant(config.get("hinge", 0.0))
    cutoff = tf.constant(config.get("cutoff", 255.0))

    # --- mae
    mae_multiplier = tf.constant(config.get("mae_multiplier", 1.0))

    # --- regularization
    regularization_multiplier = tf.constant(config.get("regularization", 1.0))

    def loss_function(
            input_batch,
            prediction_batch,
            noisy_batch,
            model_losses) -> Dict:
        """
        The loss function of the depth prediction model

        :param input_batch: ground truth
        :param prediction_batch: prediction
        :param noisy_batch: noisy batch
        :param model_losses: weight/regularization losses
        :return: dictionary of losses
        """

        # --- mean absolute error from prediction
        error = input_batch - prediction_batch
        mae_actual = \
            mae_diff(
                error=error,
                hinge=0)

        # --- loss prediction on mae
        mae_prediction_loss = \
            mae_diff(error=error, hinge=hinge, cutoff=cutoff)

        # ---
        nae_noise = nae(input_batch, noisy_batch)
        nae_prediction = nae(input_batch, prediction_batch)
        nae_improvement = \
            (nae_noise + EPSILON_DEFAULT) / (nae_prediction + EPSILON_DEFAULT)

        # --- regularization error
        regularization_loss = tf.add_n(model_losses)

        # --- snr
        signal_to_noise_ratio = \
            snr(input_batch, prediction_batch)

        # --- add up loss
        mean_total_loss = \
            mae_prediction_loss * mae_multiplier + \
            regularization_loss * regularization_multiplier

        return {
            NAE_NOISE_STR: nae_noise,
            MAE_LOSS_STR: mae_actual,
            SNR_STR: signal_to_noise_ratio,
            NAE_PREDICTION_STR: nae_prediction,
            MEAN_TOTAL_LOSS_STR: mean_total_loss,
            NAE_IMPROVEMENT_STR: nae_improvement,
            REGULARIZATION_LOSS_STR: regularization_loss
        }

    return loss_function

# ---------------------------------------------------------------------
