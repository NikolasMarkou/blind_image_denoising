r"""Constructs the loss function of the blind image denoising"""

import tensorflow as tf
from typing import List, Dict, Callable, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .constants import *
from .custom_logger import logger

# ---------------------------------------------------------------------


def gar_loss(
        x: tf.Tensor,
        alpha: float = 1.0,
        c: float = 1.0) -> tf.Tensor:
    """
    General and Adaptive Robust loss as described in
    A General and Adaptive Robust Loss Function,
    Jonathan T. Barron,
    Google Research,
    2019

    :param x: error tensor
    :param alpha: shape parameter that controls the robustness of the loss
    :param c: c > 0 is a scale parameter that controls the size of the loss’s quadratic bowl near x = 0
    :return loss
    """
    a_2 = tf.abs(alpha - 2.0)
    return \
        (a_2 / alpha) * \
        (tf.pow((tf.square(x/c) / a_2) + 1.0, alpha / 2.0) - 1.0)

# ---------------------------------------------------------------------


def mae_diff(
        error: tf.Tensor,
        hinge: float = 0.0,
        cutoff: float = 255.0) -> tf.Tensor:
    """
    Mean Absolute Error (mean over channels and batches)

    :param error: diff between prediction and ground truth
    :param hinge: hinge value
    :param cutoff: max value

    :return: mean absolute error
    """
    d = \
        tf.keras.activations.relu(
            x=tf.abs(error),
            threshold=hinge,
            max_value=cutoff),

    # --- mean over all dims
    d = tf.reduce_mean(d, axis=[1, 2, 3])

    # --- mean over batch
    d = tf.reduce_mean(d)

    return d


# ---------------------------------------------------------------------


def mae(
        original: tf.Tensor,
        prediction: tf.Tensor,
        **kwargs) -> Tuple[tf.Tensor, tf.Tensor]:
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


def rmse_diff(
        error: tf.Tensor,
        hinge: float = 0,
        cutoff: float = (255.0 * 255.0)) -> tf.Tensor:
    """
    Root Mean Square Error (mean over channels and batches)

    :param error:
    :param hinge: hinge value
    :param cutoff: max value
    """
    d = \
        tf.keras.activations.relu(
            x=error,
            threshold=hinge,
            max_value=cutoff)
    d = tf.square(d)
    # mean over all dims
    d = tf.reduce_mean(d, axis=[1, 2, 3])
    d = tf.sqrt(d + DEFAULT_EPSILON)
    # mean over batch
    return tf.reduce_mean(d, axis=[0])


# ---------------------------------------------------------------------


def rmse(
        original: tf.Tensor,
        prediction: tf.Tensor,
        **kwargs) -> tf.Tensor:
    """
    Root Mean Square Error (mean over channels and batches)

    :param original: original image batch
    :param prediction: denoised image batch
    """
    return rmse_diff(
        error=(original - prediction),
        **kwargs)


# ---------------------------------------------------------------------

def improvement(
        original: tf.Tensor,
        noisy: tf.Tensor,
        denoised: tf.Tensor) -> tf.Tensor:
    """
    starts negative,
    goes to zero meaning no improvement
    then goes to positive meaning actual improvement
    """
    original_noisy = mae(original, noisy)
    original_denoised = mae(original, denoised)
    return original_noisy - original_denoised

# ---------------------------------------------------------------------


def loss_function_builder(
        config: Dict) -> Dict[str, Callable]:
    """
    Constructs the loss function of the depth prediction model

    :param config: configuration dictionary
    :return: callable loss function
    """
    logger.info("building loss_function with config [{0}]".format(config))

    # ---
    hinge = config.get("hinge", 0.0)
    cutoff = config.get("cutoff", 255.0)

    # --- mae
    mae_multiplier = config.get("mae_multiplier", 1.0)
    use_mae = mae_multiplier > 0.0

    # --- ssim
    ssim_multiplier = config.get("ssim_multiplier", 1.0)
    use_ssim = ssim_multiplier > 0.0

    # --- mse
    mse_multiplier = config.get("mse_multiplier", 0.0)
    use_mse = mse_multiplier > 0.0

    # --- regularization
    regularization_multiplier = config.get("regularization", 1.0)

    def model_loss(model):
        regularization_loss = \
            tf.add_n(model.losses)
        return {
            REGULARIZATION_LOSS_STR: regularization_loss,
            TOTAL_LOSS_STR: regularization_loss * regularization_multiplier
        }

    # ---
    def denoiser_loss(
            gt_batch: tf.Tensor,
            predicted_batch: tf.Tensor) -> Dict[str, tf.Tensor]:
        # actual mean absolute error (no hinge or cutoff)
        mae_actual = \
            mae(original=gt_batch,
                prediction=predicted_batch,
                hinge=0.0,
                cutoff=255.0)
        # actual mean square error (no hinge or cutoff)
        mse_actual = \
            rmse(original=gt_batch,
                prediction=predicted_batch,
                hinge=0.0,
                cutoff=255.0)

        # loss prediction on mae
        mae_prediction_loss = \
            tf.constant(0.0, dtype=tf.float32)
        if use_mae:
            mae_prediction_loss += \
                mae(original=gt_batch,
                    prediction=predicted_batch,
                    hinge=hinge,
                    cutoff=cutoff)

        # loss ssim
        ssim_loss = tf.constant(0.0, dtype=tf.float32)
        if use_ssim:
            ssim_loss = \
                tf.reduce_mean(
                    tf.image.ssim(
                        img1=gt_batch,
                        img2=predicted_batch,
                        filter_size=7,
                        max_val=255.0))
            ssim_loss = 1.0 - ssim_loss

        # loss prediction on mse
        mse_prediction_loss = \
            tf.constant(0.0, dtype=tf.float32)
        if use_mse:
            mse_prediction_loss += \
                rmse(original=gt_batch,
                     prediction=predicted_batch,
                     hinge=hinge,
                     cutoff=(cutoff * cutoff))

        return {
            TOTAL_LOSS_STR:
                mae_prediction_loss * mae_multiplier +
                mse_prediction_loss * mse_multiplier +
                ssim_loss * ssim_multiplier,
            MSE_LOSS_STR: mse_actual,
            MAE_LOSS_STR: mae_actual,
            SSIM_LOSS_STR: ssim_loss
        }

    # ----
    return {
        MODEL_LOSS_FN_STR: model_loss,
        DENOISER_LOSS_FN_STR: denoiser_loss
    }

# ---------------------------------------------------------------------
