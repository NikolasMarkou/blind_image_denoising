import os
import tensorflow as tf
from tensorflow import keras

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .constants import *
from .custom_logger import logger

# ---------------------------------------------------------------------

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# ---------------------------------------------------------------------


class InpaintModule(tf.Module):
    """inpaint inference module."""

    def __init__(
            self,
            model_backbone: keras.Model,
            model_inpaint: keras.Model,
            model_normalizer: keras.Model,
            model_denormalizer: keras.Model,
            cast_to_uint8: bool = True):
        """
        Initializes a module for super resolution.

        :param model_backbone: backbone model to use for inference
        :param model_inpaint: inpaint model to use for inference.
        :param model_normalizer: model that normalizes the input
        :param model_denormalizer: model that denormalizes the output
        :param cast_to_uint8: cast output to uint8

        """
        super().__init__(name=INPAINT_STR)

        # --- argument checking
        if model_backbone is None:
            raise ValueError("model_backbone should not be None")
        if model_inpaint is None:
            raise ValueError("model_inpaint should not be None")
        if model_normalizer is None:
            raise ValueError("model_normalizer should not be None")
        if model_denormalizer is None:
            raise ValueError("model_denormalizer should not be None")

        training_channels = \
            model_backbone.input_shape[-1]

        # --- setup instance variables
        self._cast_to_uint8 = cast_to_uint8
        self._model_backbone = model_backbone
        self._model_inpaint = model_inpaint
        self._model_normalizer = model_normalizer
        self._model_denormalizer = model_denormalizer
        self._training_channels = training_channels

    @tf.function
    def __call__(self, image, mask):
        """
        Cast image to float and run inference.

        :param image: uint8 Tensor of shape
        :return: inpaint image: uint8 Tensor of shape if the input
        """
        x = tf.cast(image, dtype=tf.float32)
        m = tf.clip_by_value(mask, clip_value_min=0, clip_value_max=1)
        m = tf.cast(m, dtype=tf.float32)

        # --- normalize
        x = self._model_normalizer(x)

        x_m = \
            tf.multiply(
                tf.ones_like(
                    input=x),
                tf.reduce_mean(
                    input_tensor=x,
                    keepdims=True,
                    axis=[1, 2]))

        x_i = \
            tf.multiply(x, m) + \
            tf.multiply(x_m, 1.0 - m)

        # --- run backbone
        x = self._model_backbone(x_i)

        # --- run inpaint model
        x = self._model_inpaint([x, x_i, m])

        # --- denormalize
        x = self._model_denormalizer(x)

        # --- cast to uint8
        if self._cast_to_uint8:
            x = tf.round(x)
            x = tf.cast(x, dtype=tf.uint8)

        return x

# ---------------------------------------------------------------------
