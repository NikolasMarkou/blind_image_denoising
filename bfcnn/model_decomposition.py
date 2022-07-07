# ---------------------------------------------------------------------

__author__ = "Nikolas Markou"
__version__ = "1.0.0"
__license__ = "MIT"

# ---------------------------------------------------------------------

import abc
import tensorflow as tf
from tensorflow import keras
from collections import namedtuple
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .custom_logger import logger

# ---------------------------------------------------------------------


class DecompositionInferenceModule(tf.Module, abc.ABC):
    """decomposition inference module."""

    def __init__(
            self,
            model_decomposition: keras.Model,
            model_normalize: keras.Model,
            model_denormalize: keras.Model,
            training_channels: int = 1,
            cast_to_uint8: bool = True):
        """
        Initializes a module for denoising.

        :param model_decomposition: denoising model to use for inference.
        :param model_normalize: model that normalizes the input
        :param model_denormalize: model that denormalizes the output
        :param training_channels: how many color channels were used in training
        :param cast_to_uint8: cast output to uint8

        """
        # --- argument checking
        if model_decomposition is None:
            raise ValueError("model_denoise_decomposition should not be None")
        if model_normalize is None:
            raise ValueError("model_normalize should not be None")
        if model_denormalize is None:
            raise ValueError("model_denormalize should not be None")
        if training_channels <= 0:
            raise ValueError("training channels should be > 0")

        # --- setup instance variables
        self._cast_to_uint8 = cast_to_uint8
        self._model_normalize = model_normalize
        self._model_denormalize = model_denormalize
        self._training_channels = training_channels
        self._model_decomposition = model_decomposition

    def _run_inference_on_images(self, image):
        """
        Cast image to float and run inference.

        :param image: uint8 Tensor of shape
        :return: denoised image: uint8 Tensor of shape if the input
        """
        x = tf.cast(image, dtype=tf.float32)

        # --- normalize
        x = self._model_normalize(x)

        # --- run denoise model
        x = self._model_denoise_decomposition(x)

        # --- denormalize
        x = self._model_denormalize(x)

        # --- cast to uint8
        if self._cast_to_uint8:
            x = tf.round(x)
            x = tf.cast(x, dtype=tf.uint8)

        return x

    @abc.abstractmethod
    def __call__(self, input_tensor):
        pass

# ---------------------------------------------------------------------


class DecompositionInferenceModule1Channel(DecompositionInferenceModule):
    def __init__(
            self,
            model_decomposition: keras.Model = None,
            model_normalize: keras.Model = None,
            model_denormalize: keras.Model = None,
            cast_to_uint8: bool = True):
        super().__init__(
            model_decomposition=model_decomposition,
            model_normalize=model_normalize,
            model_denormalize=model_denormalize,
            training_channels=1,
            cast_to_uint8=cast_to_uint8)

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=[None, None, None, 1], dtype=tf.uint8)])
    def __call__(self, input_tensor):
        return self._run_inference_on_images(input_tensor)


# ---------------------------------------------------------------------


class DecompositionInferenceModule3Channel(DecompositionInferenceModule):
    def __init__(
            self,
            model_decomposition: keras.Model = None,
            model_normalize: keras.Model = None,
            model_denormalize: keras.Model = None,
            cast_to_uint8: bool = True):
        super().__init__(
            model_decomposition=model_decomposition,
            model_normalize=model_normalize,
            model_denormalize=model_denormalize,
            training_channels=3,
            cast_to_uint8=cast_to_uint8)

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=[None, None, None, 3], dtype=tf.uint8)])
    def __call__(self, input_tensor):
        return self._run_inference_on_images(input_tensor)


# ---------------------------------------------------------------------


def module_decomposition_builder(
        model_decomposition: keras.Model = None,
        model_normalize: keras.Model = None,
        model_denormalize: keras.Model = None,
        training_channels: int = 1,
        cast_to_uint8: bool = True) -> DecompositionInferenceModule:
    """
    builds a module for decomposition.

    :param model_decomposition: denoising model to use for inference.
    :param model_normalize: model that normalizes the input
    :param model_denormalize: model that denormalizes the output
    :param training_channels: how many color channels were used in training
    :param cast_to_uint8: cast output to uint8

    :return: decomposition module
    """
    logger.info(
        f"building decomposition module with "
        f"training_channels:{training_channels}, "
        f"cast_to_uint8:{cast_to_uint8}")

    if training_channels == 1:
        return \
            DecompositionInferenceModule1Channel(
                model_decomposition=model_decomposition,
                model_normalize=model_normalize,
                model_denormalize=model_denormalize,
                cast_to_uint8=cast_to_uint8)
    elif training_channels == 3:
        return \
            DecompositionInferenceModule3Channel(
                model_decomposition=model_decomposition,
                model_normalize=model_normalize,
                model_denormalize=model_denormalize,
                cast_to_uint8=cast_to_uint8)
    else:
        raise ValueError(
            "don't know how to handle training_channels:{0}".format(training_channels))

# ---------------------------------------------------------------------
