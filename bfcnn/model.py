import tensorflow as tf
from tensorflow import keras
from typing import List, Tuple, Union, Dict

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
    input_shape = config["input_shape"]
    filters = config.get("filters", 32)
    no_layers = config.get("no_layers", 5)
    min_value = config.get("min_value", 0)
    max_value = config.get("max_value", 255)
    batchnorm = config.get("batchnorm", True)
    kernel_size = config.get("kernel_size", 3)
    output_multiplier = config.get("output_multiplier", 1.0)
    final_activation = config.get("final_activation", "linear")
    kernel_regularizer = config.get("kernel_regularizer", "l1")
    normalize_denormalize = config.get("normalize_denormalize", False)
    kernel_initializer = config.get("kernel_initializer", "glorot_normal")

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
    if model_type == "resnet":
        model_denoise = \
            build_resnet_model(
                use_bn=batchnorm,
                filters=filters,
                no_layers=no_layers,
                input_dims=input_shape,
                kernel_size=kernel_size,
                final_activation=final_activation,
                kernel_regularizer=kernel_regularizer,
                kernel_initializer=kernel_initializer)
    elif model_type == "sparse_resnet":
        model_denoise = \
            build_sparse_resnet_model(
                filters=filters,
                no_layers=no_layers,
                input_dims=input_shape,
                kernel_size=kernel_size,
                final_activation=final_activation,
                kernel_regularizer=kernel_regularizer,
                kernel_initializer=kernel_initializer)
    elif model_type == "sparse_resnet_mean_sigma":
        model_denoise = \
            build_sparse_resnet_mean_sigma_model(
                filters=filters,
                no_layers=no_layers,
                input_dims=input_shape,
                kernel_size=kernel_size,
                final_activation=final_activation,
                kernel_regularizer=kernel_regularizer,
                kernel_initializer=kernel_initializer)
    elif model_type == "gatenet":
        model_denoise = \
            build_gatenet_model(
                use_bn=batchnorm,
                filters=filters,
                no_layers=no_layers,
                input_dims=input_shape,
                kernel_size=kernel_size,
                final_activation=final_activation,
                kernel_regularizer=kernel_regularizer,
                kernel_initializer=kernel_initializer)
    else:
        raise ValueError(
            "don't know how to build model [{0}]".format(model_type))

    # --- connect the parts of the model
    # setup input
    model_input = \
        keras.Input(
            shape=input_shape,
            name="input_tensor")
    x = model_input
    # add normalize cap
    if normalize_denormalize:
        x = model_normalize(x)

    mean, sigma = mean_sigma_global(x)
    x = (x - mean) / sigma

    # denoise image
    x = model_denoise(x)

    # uplift a bit because of tanh saturation
    if output_multiplier != 1.0:
        x = x * output_multiplier

    x = (x * sigma) + mean
    x = \
        keras.backend.clip(
            x,
            min_value=min_value,
            max_value=max_value)

    # add denormalize cap
    if normalize_denormalize:
        x = model_denormalize(x)

    # --- wrap model
    model_denoise = \
        keras.Model(
            inputs=model_input,
            outputs=x)

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
