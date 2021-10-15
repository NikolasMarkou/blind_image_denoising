# ---------------------------------------------------------------------

__author__ = "Nikolas Markou"
__version__ = "0.1.0"
__license__ = "None"

# ---------------------------------------------------------------------

import tensorflow as tf
from tensorflow import keras
from typing import List, Tuple, Union, Dict

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .utilities import *
from .custom_logger import logger

# ---------------------------------------------------------------------


def model_builder(
        config: Dict) -> Tuple[keras.Model, keras.Model, keras.Model]:
    """
    Reads a configuration and returns 3 models,

    :param config: configuration dictionary
    :return: denoiser model, normalize model, denormalize model
    """
    logger.info("building model with config [{0}]".format(config))

    # --- argument parsing
    model_type = config["type"]
    levels = config.get("levels", 1)
    filters = config.get("filters", 32)
    no_layers = config.get("no_layers", 5)
    min_value = config.get("min_value", 0)
    max_value = config.get("max_value", 255)
    batchnorm = config.get("batchnorm", True)
    kernel_size = config.get("kernel_size", 3)
    stop_grads = config.get("stop_grads", False)
    input_shape = config.get("input_shape", (None, None, 3))
    output_multiplier = config.get("output_multiplier", 1.0)
    local_normalization = config.get("local_normalization", -1)
    final_activation = config.get("final_activation", "linear")
    kernel_regularizer = config.get("kernel_regularizer", "l1")
    kernel_initializer = config.get("kernel_initializer", "glorot_normal")
    use_local_normalization = local_normalization > 0
    local_normalization_kernel = [local_normalization, local_normalization]

    for i in range(len(input_shape)):
        if input_shape[i] == "?" or \
                input_shape[i] == "" or \
                input_shape[i] == "-1":
            input_shape[i] = None

    # --- build normalize denormalize models
    model_normalize = \
        build_normalize_model(
            input_dims=input_shape,
            min_value=min_value,
            max_value=max_value)

    model_denormalize = \
        build_denormalize_model(
            input_dims=input_shape,
            min_value=min_value,
            max_value=max_value)

    # --- build denoise model
    model_params = dict(
        add_gates=False,
        filters=filters,
        use_bn=batchnorm,
        add_sparsity=False,
        no_layers=no_layers,
        input_dims=input_shape,
        kernel_size=kernel_size,
        final_activation=final_activation,
        kernel_regularizer=kernel_regularizer,
        kernel_initializer=kernel_initializer
    )

    if model_type == "resnet":
        pass
    elif model_type == "sparse_resnet":
        model_params["add_sparsity"] = True
    elif model_type == "gatenet":
        model_params["add_gates"] = True
    else:
        raise ValueError(
            "don't know how to build model [{0}]".format(model_type))

    model_pyramid = \
        build_gaussian_pyramid_model(
            input_dims=input_shape,
            levels=levels)

    def func_sigma_norm(args):
        y, mean_y, sigma_y = args
        return (y - mean_y) / sigma_y

    def func_sigma_denorm(args):
        y, mean_y, sigma_y = args
        return (y * sigma_y) + mean_y

    # --- connect the parts of the model
    # setup input
    input_layer = \
        keras.Input(
            shape=input_shape,
            name="input_tensor")
    x = input_layer

    # local normalization cap
    if use_local_normalization:
        mean, sigma = \
            mean_sigma_local(
                x,
                kernel_size=local_normalization_kernel)
        x = \
            keras.layers.Lambda(
                function=func_sigma_norm,
                trainable=False)([x, mean, sigma])

    x_levels = model_pyramid(x)
    x_previous_result = None
    level = 0

    for x_level in x_levels[::-1]:
        if x_previous_result is not None:
            x_previous_result = \
                keras.layers.UpSampling2D(
                    size=(2, 2),
                    interpolation="nearest")(x_previous_result)
            x_level = \
                keras.layers.Add()([x_level, -x_previous_result])

        # stop gradient to the upper level (makes learning faster)
        if stop_grads:
            x_level = keras.backend.stop_gradient(x_level)

        # denoise image
        x_level = \
            build_resnet_model(
                name=f"level_{level}",
                **model_params)(x_level)

        # uplift a bit because of tanh saturation
        if output_multiplier != 1.0:
            x_level = x_level * output_multiplier

        if x_previous_result is None:
            x_previous_result = x_level
        else:
            x_previous_result = \
                keras.layers.Add()([x_previous_result, x_level])
        level = level + 1

    # local denormalization cap
    if use_local_normalization:
        x_previous_result = \
            keras.layers.Lambda(
                function=func_sigma_denorm,
                trainable=False)([x_previous_result, mean, sigma])

    # name output
    output_layer = \
        keras.layers.Layer(name="output_tensor")(x_previous_result)

    # --- wrap and name model
    model_denoise = \
        keras.Model(
            inputs=input_layer,
            outputs=output_layer,
            name=f"{model_type}_denoiser")

    return \
        model_denoise, \
        model_normalize, \
        model_denormalize

# ---------------------------------------------------------------------


class DenoisingInferenceModule(tf.Module):
    """denoising inference module."""

    def __init__(
            self,
            model_denoise: keras.Model,
            model_normalize: keras.Model = None,
            model_denormalize: keras.Model = None,
            iterations: int = 1,
            cast_to_uint8: bool = True):
        """
        Initializes a module for detection.

        :param model_denoise: denoising model to use for inference.
        :param model_normalize:
        :param model_denormalize:
        :param iterations: how many times to run the model
        :param cast_to_uint8: cast output to uint8
        """
        # --- argument checking
        if model_denoise is None:
            raise ValueError("model_denoise should not be None")
        if iterations <= 0:
            raise ValueError("iterations should be > 0")

        # --- setup instance variables
        self._model_denoise = model_denoise
        self._model_normalize = model_normalize
        self._model_denormalize = model_denormalize
        self._iterations = iterations
        self._cast_to_uint8 = cast_to_uint8

    def _run_inference_on_images(self, image):
        """
        Cast image to float and run inference.

        :param image: uint8 Tensor of shape [1, None, None, 3]
        :return: denoised image: uint8 Tensor of shape [1, None, None, 3]
        """
        x = tf.cast(image, dtype=tf.float32)

        # --- normalize
        if self._model_normalize is not None:
            x = self._model_normalize(x)

        # --- run denoise model as many times as required
        for i in range(self._iterations):
            x = self._model_denoise(x)
            x = keras.backend.clip(x, min_value=-0.5, max_value=+0.5)

        # --- denormalize
        if self._model_denormalize is not None:
            x = self._model_denormalize(x)

        # --- cast to uint8
        if self._cast_to_uint8:
            x = tf.round(x)
            x = tf.cast(x, dtype=tf.uint8)
        
        return x

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=[1, None, None, 3], dtype=tf.uint8)])
    def __call__(self, input_tensor):
        return self._run_inference_on_images(input_tensor)

# ---------------------------------------------------------------------
