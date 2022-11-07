r"""Constructs the loss function of the blind image denoising"""

import tensorflow as tf
from typing import List, Dict, Callable

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .constants import *
from .custom_logger import logger
from .delta import delta_xy_magnitude
from .pyramid import build_pyramid_model

# ---------------------------------------------------------------------


def delta(
        x: tf.Tensor,
        mask: tf.Tensor = None,
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


def snr(
        original: tf.Tensor,
        prediction: tf.Tensor,
        multiplier: float = 10.0,
        base: float = 10.0) -> tf.Tensor:
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
    # --- mean over all dims
    if count_non_zero_mean:
        d = \
            tf.keras.activations.relu(
                x=tf.abs(error),
                threshold=hinge,
                max_value=cutoff)
        d_count = \
            tf.math.count_nonzero(
                input=d,
                axis=[1, 2, 3],
                keepdims=False,
                dtype=tf.uint32)
        d_sum = \
            tf.reduce_sum(
                input_tensor=d,
                axis=[1, 2, 3],
                keepdims=False)
        d = d_sum / (d_count + 1.0)
    else:
        d = \
            tf.reduce_mean(
                tf.math.multiply(
                    tf.keras.activations.relu(
                        x=tf.abs(error),
                        threshold=hinge,
                        max_value=cutoff),
                    mask),
                axis=[1, 2, 3])
    # mean over batch
    return tf.reduce_mean(d, axis=[0])


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


def mse_diff(
        error: tf.Tensor,
        hinge: float = 0,
        cutoff: float = (255.0 * 255.0)) -> tf.Tensor:
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
            max_value=cutoff)
    # mean over all dims
    d = tf.reduce_mean(d, axis=[1, 2, 3])
    # mean over batch
    return tf.reduce_mean(d, axis=[0])


# ---------------------------------------------------------------------


def mse(
        original: tf.Tensor,
        prediction: tf.Tensor,
        **kwargs) -> tf.Tensor:
    """
    Mean Square Error (mean over channels and batches)

    :param original: original image batch
    :param prediction: denoised image batch
    """
    return mse_diff(
        error=tf.square(original - prediction),
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
    hinge = tf.constant(config.get("hinge", 0.0))
    cutoff = tf.constant(config.get("cutoff", 255.0))

    # --- mae
    mae_multiplier = tf.constant(config.get("mae_multiplier", 1.0))

    # --- count non zero
    count_non_zero_mean = tf.constant(config.get("count_non_zero_mean", False))

    # --- delta
    use_delta = tf.constant(config.get("use_delta", False))

    # --- regularization
    regularization_multiplier = tf.constant(config.get("regularization", 1.0))

    # --- features
    features_multiplier = tf.constant(config.get("features_multiplier", 0.0))
    use_features = features_multiplier > 0.0

    # --- multiscale mae
    use_multiscale = tf.constant(config.get("use_multiscale", False))
    laplacian_config = {
        "levels": 3,
        "type": "laplacian",
        "xy_max": (1.0, 1.0),
        "kernel_size": (5, 5)
    }
    pyramid_model = \
        build_pyramid_model(
            input_dims=(None, None, 3),
            config=laplacian_config)
    pyramid_levels = \
        tf.constant(laplacian_config["levels"])

    def loss_function(
            input_batch: tf.Tensor,
            prediction_batch: tf.Tensor,
            noisy_batch: tf.Tensor,
            model_losses: tf.Tensor,
            feature_map_batch: tf.Tensor = None,
            mask_batch: tf.Tensor = tf.constant(1.0, dtype=tf.float32)) -> Dict[str, tf.Tensor]:
        """
        The loss function of the depth prediction model

        :param input_batch: ground truth
        :param prediction_batch: prediction
        :param noisy_batch: noisy batch
        :param model_losses: weight/regularization losses
        :param feature_map_batch: features batch
        :param mask_batch: mask to focus on

        :return: dictionary of losses
        """
        # --- actual mean absolute error (no hinge or cutoff)
        mae_actual = \
            mae(original=input_batch,
                prediction=prediction_batch,
                hinge=0.0,
                cutoff=255.0)

        # --- loss prediction on mae
        mae_prediction_loss = \
            tf.constant(0.0, dtype=tf.float32)
        if use_multiscale:
            input_batch_multiscale = \
                pyramid_model(input_batch, training=False)
            prediction_batch_multiscale = \
                pyramid_model(prediction_batch, training=False)
            for i in range(pyramid_levels):
                mae_prediction_loss += \
                    mae(original=input_batch_multiscale[i],
                        prediction=prediction_batch_multiscale[i],
                        mask=mask_batch,
                        hinge=hinge,
                        cutoff=cutoff,
                        count_non_zero_mean=count_non_zero_mean)
        elif use_delta:
            mae_prediction_loss += \
                mae_weighted_delta(
                    original=input_batch,
                    prediction=prediction_batch,
                    mask=mask_batch,
                    hinge=hinge,
                    cutoff=cutoff)
        else:
            mae_prediction_loss += \
                mae(original=input_batch,
                    prediction=prediction_batch,
                    mask=mask_batch,
                    hinge=hinge,
                    cutoff=cutoff,
                    count_non_zero_mean=count_non_zero_mean)

        # --- regularization on features map
        feature_map_regularization_loss = tf.constant(0.0)
        if use_features:
            feature_map_regularization_loss = soft_orthogonal(feature_map_batch)

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
            regularization_loss * regularization_multiplier + \
            feature_map_regularization_loss * features_multiplier

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
