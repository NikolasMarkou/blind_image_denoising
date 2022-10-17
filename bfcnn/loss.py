r"""Constructs the loss function of the blind image denoising"""

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


def delta(
        x: tf.Tensor,
        mask: tf.Tensor = None,
        kernel_size: int = 3,
        alpha: float = 1.0,
        beta: float = 1.0,
        eps: float = DEFAULT_EPSILON,
        axis: List[int] = [1, 2, 3]):
    """
    Computes the delta loss of a layer
    (alpha * (dI/dx)^2 + beta * (dI/dy)^2) ^ 0.5

    :param x:
    :param mask: pixels to ignore
    :param kernel_size: how big the delta kernel should be
    :param alpha: multiplier of dx
    :param beta: multiplier of dy
    :param eps: small value to add for stability
    :param axis: list of axis to sum against
    :return: delta loss
    """
    dd = \
        delta_xy_magnitude(
            input_layer=x,
            kernel_size=kernel_size,
            alpha=alpha,
            beta=beta,
            eps=eps)
    if mask is None:
        return tf.reduce_mean(dd, axis=axis, keepdims=False)
    dd = dd * mask
    valid_pixels = tf.reduce_sum(mask, axis=axis, keepdims=False) + eps
    return tf.reduce_sum(dd, axis=axis, keepdims=False) / valid_pixels

# ---------------------------------------------------------------------


def snr(
        original: tf.Tensor,
        prediction: tf.Tensor,
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
    d_2 = tf.reduce_sum(tf.square(original - prediction), axis=[1, 2, 3])
    d_prediction = tf.reduce_sum(prediction, axis=[1, 2, 3])
    # mean over batch
    result = d_prediction / (d_2 + DEFAULT_EPSILON)
    return \
        tf.reduce_mean(
             tf.math.log(result) * (multiplier / tf.math.log(base)),
            axis=[0])


# ---------------------------------------------------------------------


def mae_weighted_delta(
        original: tf.Tensor,
        prediction: tf.Tensor,
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
            eps=DEFAULT_EPSILON)

    d_weight = \
        original_delta / \
        (tf.abs(tf.reduce_max(
            input_tensor=original_delta,
            axis=[1, 2],
            keepdims=True)) +
         DEFAULT_EPSILON)
    d_weight = tf.abs(d_weight)

    # --- calculate hinged absolute diff
    d = \
        tf.keras.activations.relu(
            x=tf.abs(original - prediction),
            alpha=0.0,
            max_value=None,
            threshold=hinge)

    # --- multiply diff and weight
    d = tf.math.multiply(d, d_weight)

    # --- mean over all dims
    d = tf.reduce_mean(d, axis=[1, 2, 3])

    # --- mean over batch
    loss = tf.reduce_mean(d, axis=[0])

    return loss


# ---------------------------------------------------------------------


def mae_diff(
        error: tf.Tensor,
        hinge: float = 0.0,
        cutoff: float = 255.0):
    """
    Mean Absolute Error (mean over channels and batches)

    :param error: original image batch
    :param hinge: hinge value
    :param cutoff: max value
    """
    # --- mean over all dims
    d = tf.reduce_mean(
        tf.keras.activations.relu(
            x=tf.abs(error),
            threshold=hinge,
            max_value=cutoff),
        axis=[1, 2, 3])
    # mean over batch
    return tf.reduce_mean(d, axis=[0])


# ---------------------------------------------------------------------


def mae(
        original: tf.Tensor,
        prediction: tf.Tensor,
        **kwargs):
    """
    Mean Absolute Error (mean over channels and batches)

    :param original: original image batch
    :param prediction: denoised image batch
    :param hinge: hinge value
    :param cutoff: max value
    """
    return \
        mae_diff(
            error=(original - prediction),
            **kwargs)


# ---------------------------------------------------------------------


def mse_diff(
        error: tf.Tensor,
        hinge: float = 0,
        cutoff: float = (255.0 * 255.0)):
    """
    Mean Square Error (mean over channels and batches)

    :param error:
    :param hinge: hinge value
    :param cutoff: max value
    """
    d = \
        tf.keras.activations.relu(
            x=tf.square(error),
            threshold=hinge,
            max_value=cutoff)(d)
    # mean over all dims
    d = tf.reduce_mean(d, axis=[1, 2, 3])
    # mean over batch
    return tf.reduce_mean(d, axis=[0])


# ---------------------------------------------------------------------


def mse(
        original: tf.Tensor,
        prediction: tf.Tensor,
        **kwargs):
    """
    Mean Square Error (mean over channels and batches)

    :param original: original image batch
    :param prediction: denoised image batch
    :param hinge: hinge value
    :param cutoff: max value
    """
    return mse_diff(
        error=tf.square(original - prediction),
        **kwargs)


# ---------------------------------------------------------------------


def nae(
        original: tf.Tensor,
        prediction: tf.Tensor,
        hinge: float = 0):
    """
    Normalized Absolute Error
    (sum over width, height, channel and mean over batches)

    :param original: original image batch
    :param prediction: denoised image batch
    :param hinge: hinge value
    """
    d = tf.keras.activations.relu(x=tf.abs(original - prediction), threshold=hinge)
    # sum over all dims
    d = tf.reduce_sum(d, axis=[1, 2, 3])
    d_x = tf.reduce_sum(original, axis=[1, 2, 3])
    # mean over batch
    return \
        tf.reduce_mean(d, axis=[0]) / \
        (tf.reduce_mean(d_x, axis=[0]) + DEFAULT_EPSILON)


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
            (nae_noise + DEFAULT_EPSILON) / (nae_prediction + DEFAULT_EPSILON)

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
