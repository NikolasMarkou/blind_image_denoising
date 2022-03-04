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
    """
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


def mae_weighted(
        original,
        noisy,
        prediction,
        hinge: float = 0):
    """
    Mean Absolute Error (mean over channels and batches) with weights

    :param original:
    :param noisy:
    :param prediction:
    :param hinge: hinge value
    """
    # --- calculate the weight per pixel based on how noisy it is
    d_weight = tf.pow(original - noisy, 2) + EPSILON_DEFAULT
    if hinge != 0.0:
        d_weight = keras.layers.ReLU(threshold=hinge)(d_weight)
    d_weight = keras.layers.Softmax(axis=[1, 2])(d_weight)
    d_weight = 1.0 - d_weight

    # --- calculate hinged absolute diff
    d = tf.abs(original - prediction)
    if hinge != 0.0:
        d = keras.layers.ReLU(threshold=hinge)(d)

    # --- multiply diff and weight
    d = keras.layers.Multiply()([d, d_weight])

    # --- sum over all dims
    d = tf.reduce_mean(d, axis=[1, 2, 3])

    # --- mean over batch
    loss = tf.reduce_mean(d, axis=[0])

    return loss

# ---------------------------------------------------------------------


def mae_weighted_delta(
        original,
        prediction,
        hinge: float = 0):
    """
    Mean Absolute Error (mean over channels and batches) with weights

    :param original:
    :param prediction:
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
        keras.layers.Softmax(axis=[1, 2])(original_delta)

    # --- calculate hinged absolute diff
    d = tf.abs(original - prediction)
    if hinge != 0.0:
        d = keras.layers.ReLU(threshold=hinge)(d)

    # --- multiply diff and weight
    d = keras.layers.Multiply()([d, d_weight])

    # --- sum over all dims
    d = tf.reduce_mean(d, axis=[1, 2, 3])

    # --- mean over batch
    loss = tf.reduce_mean(d, axis=[0])

    return loss

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

    # controls how we discount each level
    hinge = config.get("hinge", 0.0)
    nae_multiplier = config.get("nae_multiplier", 0.0)
    mae_multiplier = config.get("mae_multiplier", 1.0)
    mae_delta_enabled = config.get("mae_delta", False)
    mae_weighted_enabled = config.get("mae_weighted", False)
    regularization_multiplier = config.get("regularization", 1.0)

    def loss_function(
            input_batch,
            prediction_batch,
            noisy_batch=None,
            model_losses=None,
            discriminate_batch=None,
            discriminate_ground_truth=None,
            difficulty: float = -1.0) -> Dict:
        """
        The loss function of the depth prediction model

        :param: input_batch: ground truth
        :param: prediction_batch: prediction
        :param: model_losses: weight/regularization losses
        :param: difficulty:
            if >= 0 then it is an indication how corrupted the noisy batch is
        :return: dictionary of losses
        """

        # --- mean absolute error from prediction
        mae_weighted_delta_loss = 0.0
        mae_weighted_prediction_loss = 0.0
        mae_prediction_loss = \
            mae(
                original=input_batch,
                prediction=prediction_batch,
                hinge=hinge)
        if mae_delta_enabled:
            mae_weighted_delta_loss = \
                mae_weighted_delta(
                    original=input_batch,
                    prediction=prediction_batch,
                    hinge=hinge)
        if mae_weighted_enabled:
            mae_weighted_prediction_loss = \
                mae_weighted(
                    original=input_batch,
                    noisy=noisy_batch,
                    prediction=prediction_batch,
                    hinge=hinge)
        # ---
        nae_prediction = \
            nae(input_batch, prediction_batch, hinge)

        nae_noise = \
            nae(input_batch, noisy_batch, hinge)

        nae_improvement = nae_noise - nae_prediction

        # ---
        discriminate_loss = 0
        if discriminate_batch is not None and \
                discriminate_ground_truth is not None:
            discriminate_loss = \
                keras.losses.sparse_categorical_crossentropy(
                    y_true=discriminate_ground_truth,
                    y_pred=discriminate_batch,
                    axis=-1)
            discriminate_loss = \
                tf.reduce_mean(discriminate_loss, axis=[1, 2], keepdims=False)
            discriminate_loss = \
                tf.reduce_mean(discriminate_loss, axis=[0])

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
            (mae_prediction_loss +
             mae_weighted_delta_loss +
             mae_weighted_prediction_loss) * mae_multiplier + \
            regularization_loss * regularization_multiplier

        return {
            "nae_noise": nae_noise,
            "snr": signal_to_noise_ratio,
            "mae_loss": mae_prediction_loss,
            "nae_prediction": nae_prediction,
            MEAN_TOTAL_LOSS_STR: mean_total_loss,
            "nae_improvement": nae_improvement,
            DISCRIMINATE_LOSS_STR: discriminate_loss,
            REGULARIZATION_LOSS_STR: regularization_loss
        }

    return loss_function

# ---------------------------------------------------------------------
