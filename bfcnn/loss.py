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
from .pyramid import build_pyramid_model
from .utilities import input_shape_fixer

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


def mae(
        original,
        prediction,
        hinge: float = 0.0):
    """
    Mean Absolute Error (mean over channels and batches)

    :param original: original image batch
    :param prediction: denoised image batch
    :param hinge: hinge value
    """
    d = tf.abs(original - prediction)
    d = keras.layers.ReLU(threshold=hinge)(d)
    # mean over all dims
    d = tf.reduce_mean(d, axis=[1, 2, 3])
    # mean over batch
    return tf.reduce_mean(d, axis=[0])

# ---------------------------------------------------------------------


def mse(
        original,
        prediction,
        hinge: float = 0):
    """
    Mean Square Error (mean over channels and batches)

    :param original: original image batch
    :param prediction: denoised image batch
    :param hinge: hinge value
    """
    d = tf.square(original - prediction)
    d = keras.layers.ReLU(threshold=hinge)(d)
    # mean over all dims
    d = tf.reduce_mean(d, axis=[1, 2, 3])
    # mean over batch
    return tf.reduce_mean(d, axis=[0])


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

    # controls how we discount each level
    hinge = config.get("hinge", 0.0)
    nae_multiplier = tf.constant(config.get("nae_multiplier", 0.0))
    mae_multiplier = tf.constant(config.get("mae_multiplier", 1.0))
    mae_decomposition_multiplier = tf.constant(config.get("mae_decomposition_multiplier", 1.0))
    mae_delta_enabled = tf.constant(config.get("mae_delta", False))
    regularization_multiplier = config.get("regularization", 1.0)
    levels_bn = [
        keras.layers.BatchNormalization()
        for _ in range(10)
    ]

    def loss_function(
            input_batch,
            prediction_batch,
            noisy_batch,
            model_losses,
            input_batch_decomposition = [],
            prediction_batch_decomposition = []) -> Dict:
        """
        The loss function of the depth prediction model

        :param input_batch: ground truth
        :param prediction_batch: prediction
        :param noisy_batch: noisy batch
        :param model_losses: weight/regularization losses
        :return: dictionary of losses
        """

        # --- mean absolute error from prediction
        mae_prediction_loss = tf.constant(0.0)
        mae_weighted_delta_loss = tf.constant(0.0)

        mae_prediction_loss += \
            mae(
                original=input_batch,
                prediction=prediction_batch,
                hinge=hinge)
        if mae_delta_enabled:
            mae_weighted_delta_loss += \
                mae_weighted_delta(
                    original=input_batch,
                    prediction=prediction_batch,
                    hinge=hinge)

        mae_actual = \
            mae(
                original=input_batch,
                prediction=prediction_batch,
                hinge=0)

        # --- loss prediction on decomposition
        mae_decomposition_loss = tf.constant(0.0)
        for i in range(len(prediction_batch_decomposition)):
            input_x_i = levels_bn[i](input_batch_decomposition[i])
            prediction_x_i = levels_bn[i](prediction_batch_decomposition[i])
            mae_decomposition_loss += \
                mae(
                    original=input_x_i,
                    prediction=prediction_x_i,
                    hinge=0)

        # ---
        nae_prediction = \
            nae(input_batch, prediction_batch, hinge)

        nae_noise = \
            nae(input_batch, noisy_batch, hinge)

        nae_improvement = nae_noise - nae_prediction

        # --- regularization error
        regularization_loss = tf.add_n(model_losses)

        # --- snr
        signal_to_noise_ratio = \
            snr(input_batch, prediction_batch)

        # --- add up loss
        mean_total_loss = \
            nae_prediction * nae_multiplier + \
            regularization_loss * regularization_multiplier + \
            (mae_prediction_loss + mae_weighted_delta_loss + mae_decomposition_loss) * mae_multiplier

        return {
            "nae_noise": nae_noise,
            MAE_LOSS_STR: mae_actual,
            "snr": signal_to_noise_ratio,
            "nae_prediction": nae_prediction,
            MEAN_TOTAL_LOSS_STR: mean_total_loss,
            "nae_improvement": nae_improvement,
            REGULARIZATION_LOSS_STR: regularization_loss,
            MAE_DECOMPOSITION_LOSS_STR: mae_decomposition_loss,
        }

    return loss_function

# ---------------------------------------------------------------------
