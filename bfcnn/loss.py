r"""Constructs the loss function of the blind image denoising"""

import tensorflow as tf
import tensorflow_addons as tfa
from typing import List, Dict, Callable

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .constants import *
from .custom_logger import logger
from .delta import delta_xy_magnitude
from .pyramid import build_pyramid_model

MODEL_LOSS_FN_STR = "model"
INPAINT_LOSS_FN_STR = "inpaint"
DENOISER_LOSS_FN_STR = "denoiser"
SUPERRES_LOSS_FN_STR = "superres"
DENOISER_UQ_LOSS_FN_STR = "denoiser_uq"

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


def delta(
        x: tf.Tensor,
        mask: tf.Tensor = tf.constant(1.0, tf.float32),
        kernel_size: int = 3,
        alpha: float = 1.0,
        beta: float = 1.0,
        eps: float = DEFAULT_EPSILON,
        axis: List[int] = [1, 2, 3]) -> tf.Tensor:
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
    valid_pixels = tf.reduce_sum(mask, axis=axis, keepdims=False) + 1
    return tf.reduce_sum(dd, axis=axis, keepdims=False) / valid_pixels


# ---------------------------------------------------------------------


def psnr(
        original: tf.Tensor,
        prediction: tf.Tensor,
        max_val: float = 255.0) -> tf.Tensor:
    """
    Peak signal-to-noise ratio expressed in dB

    :param original: original image batch
    :param prediction: denoised image batch
    :param max_val:
    """
    # psnr of prediction per image pair
    psnr_batch = tf.image.psnr(original, prediction, max_val=max_val)

    return tf.reduce_mean(psnr_batch, axis=[0])


# ---------------------------------------------------------------------


def mae_weighted_delta(
        original: tf.Tensor,
        prediction: tf.Tensor,
        mask: tf.Tensor = tf.constant(1.0, tf.float32),
        bias: float = 1.0,
        hinge: float = 0.0,
        cutoff: float = 255.0) -> tf.Tensor:
    """
    Mean Absolute Error (mean over channels and batches) with weights

    :param original: original image batch
    :param prediction: denoised image batch
    :param mask: where to focus the loss
    :param bias: add this to the whole xy magnitude
    :param hinge: hinge value
    :param cutoff: max value

    :return: mean absolute error weighted by delta
    """
    d_weight = \
        delta_xy_magnitude(
            input_layer=original,
            kernel_size=5,
            alpha=1.0,
            beta=1.0,
            eps=DEFAULT_EPSILON)
    d_weight = d_weight + bias
    d_weight = \
        d_weight / \
        (tf.reduce_max(
            input_tensor=d_weight,
            axis=[1, 2],
            keepdims=True) + DEFAULT_EPSILON)
    d_weight = tf.abs(d_weight)

    # --- calculate hinged absolute diff
    d = \
        tf.keras.activations.relu(
            x=tf.abs(original - prediction),
            alpha=0.0,
            max_value=cutoff,
            threshold=hinge)

    # --- multiply diff and weight
    d = tf.math.multiply(d, d_weight)
    d = tf.math.multiply(d, mask)

    # --- mean over all dims
    d = tf.reduce_mean(d, axis=[1, 2, 3])

    # --- mean over batch
    return tf.reduce_mean(d, axis=[0])


# ---------------------------------------------------------------------


def mae_diff(
        error: tf.Tensor,
        mask: tf.Tensor = tf.constant(1.0, tf.float32),
        count_non_zero_mean: bool = False,
        hinge: float = 0.0,
        cutoff: float = 255.0) -> tf.Tensor:
    """
    Mean Absolute Error (mean over channels and batches)

    :param error: diff between prediction and ground truth
    :param mask:
    :param count_non_zero_mean: if True, calculate mean on non zero
    :param hinge: hinge value
    :param cutoff: max value

    :return: mean absolute error
    """
    axis = [1, 2, 3]
    d = \
        tf.multiply(
            tf.keras.activations.relu(
                x=tf.abs(error),
                threshold=hinge,
                max_value=cutoff),
            mask)

    # --- mean over all dims
    if count_non_zero_mean:
        # mean over non zero
        d = \
            (tf.reduce_sum(
                input_tensor=d,
                axis=axis,
                keepdims=False)) / \
            (DEFAULT_EPSILON +
             tf.math.count_nonzero(
                 input=d,
                 axis=axis,
                 keepdims=False,
                 dtype=tf.float32))

    # --- mean over batch
    return tf.reduce_mean(d)


# ---------------------------------------------------------------------


def mae(
        original: tf.Tensor,
        prediction: tf.Tensor,
        **kwargs) -> tf.Tensor:
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
    d = tf.sqrt(tf.nn.relu(d))
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


def nae(
        original: tf.Tensor,
        prediction: tf.Tensor,
        hinge: float = 0) -> tf.Tensor:
    """
    Normalized Absolute Error
    (sum over width, height, channel and mean over batches)

    :param original: original image batch
    :param prediction: denoised image batch
    :param hinge: hinge value
    """
    d = tf.keras.activations.relu(
        x=tf.abs(original - prediction),
        threshold=hinge)

    # sum over all dims
    d = tf.reduce_sum(d, axis=[1, 2, 3])
    d_x = tf.reduce_sum(original, axis=[1, 2, 3])

    # mean over batch
    return \
        tf.reduce_mean(d, axis=[0]) / \
        (tf.reduce_mean(d_x, axis=[0]) + DEFAULT_EPSILON)


# ---------------------------------------------------------------------


def soft_orthogonal(
        feature_map: tf.Tensor) -> tf.Tensor:
    """

    :return: soft orthogonal loss
    """
    # move channels
    shape = tf.shape(feature_map)
    f = tf.transpose(feature_map, perm=(0, 3, 1, 2))
    ft = tf.reshape(f, shape=(shape[0], shape[3], -1))
    ft_x_f = \
        tf.linalg.matmul(
            ft,
            tf.transpose(ft, perm=(0, 2, 1)))

    # identity matrix
    i = tf.eye(
        num_rows=shape[3],
        num_columns=shape[3])
    x = \
        tf.square(
            tf.norm(ft_x_f - i,
                    ord="fro",
                    axis=(1, 2),
                    keepdims=False))
    return tf.reduce_mean(x, axis=[0])


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

    # --- uq
    # forces the uncertainty to have a small variance
    uq_sigma_min = config.get("uq_min", 0.01)
    uq_sigma_max = config.get("uq_max", 1.0)
    uq_sigma_multiplier = config.get("uq_sigma_multiplier", 1.0)
    uq_entropy_multiplier = config.get("uq_entropy_multiplier", 1.0)

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
    def uq_loss(
            sigma_batch: tf.Tensor = None,
            entropy_batch: tf.Tensor = None) -> tf.Tensor:
        sigma_loss = tf.constant(0.0, dtype=tf.float32)
        if sigma_batch is not None:
            sigma_batch = \
                tf.clip_by_value(
                    sigma_batch,
                    clip_value_min=uq_sigma_min,
                    clip_value_max=uq_sigma_max)
            sigma_loss += tf.reduce_mean(sigma_batch)

        entropy_loss = tf.constant(0.0, dtype=tf.float32)
        if entropy_batch is not None:
            entropy_loss += tf.reduce_mean(entropy_loss)

        return {
            TOTAL_LOSS_STR:
                sigma_loss * uq_sigma_multiplier +
                entropy_loss * uq_entropy_multiplier,
            SIGMA_LOSS_STR: sigma_loss,
            ENTROPY_LOSS_STR: entropy_loss
        }

    # ---
    def denoiser_loss(
            input_batch: tf.Tensor,
            predicted_batch: tf.Tensor,
            mask: tf.Tensor = tf.constant(1.0, tf.float32)) -> Dict[str, tf.Tensor]:
        # --- actual mean absolute error (no hinge or cutoff)
        mae_actual = \
            mae(original=input_batch,
                prediction=predicted_batch,
                hinge=0.0,
                cutoff=255.0)

        # --- loss prediction on mae
        mae_prediction_loss = \
            tf.constant(0.0, dtype=tf.float32)
        if use_mae:
            mae_prediction_loss += \
                mae(original=input_batch,
                    prediction=predicted_batch,
                    mask=mask,
                    hinge=hinge,
                    cutoff=cutoff)

        # --- loss ssim
        ssim_loss = tf.constant(0.0, dtype=tf.float32)
        if use_ssim:
            ssim_loss = \
                tf.reduce_mean(tf.image.ssim(input_batch, predicted_batch, 255.0))

        # --- loss prediction on mse
        mse_prediction_loss = \
            tf.constant(0.0, dtype=tf.float32)
        if use_mse:
            mse_prediction_loss += \
                rmse(original=input_batch,
                     prediction=predicted_batch,
                     hinge=hinge,
                     cutoff=(cutoff * cutoff))

        # --- snr
        peak_signal_to_noise_ratio = \
            psnr(input_batch, predicted_batch)

        return {
            TOTAL_LOSS_STR:
                mae_prediction_loss * mae_multiplier +
                mse_prediction_loss * mse_multiplier +
                ssim_loss * ssim_multiplier,
            MAE_LOSS_STR: mae_actual,
            SSIM_LOSS_STR: ssim_loss,
            PSNR_STR: peak_signal_to_noise_ratio
        }

    # ----
    return {
        MODEL_LOSS_FN_STR: model_loss,
        DENOISER_LOSS_FN_STR: denoiser_loss,
        SUPERRES_LOSS_FN_STR: denoiser_loss,
        DENOISER_UQ_LOSS_FN_STR: uq_loss
    }

# ---------------------------------------------------------------------
