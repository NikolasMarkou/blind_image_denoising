import tensorflow as tf

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .constants import *

# ---------------------------------------------------------------------


class SuperresModule(tf.Module):
    """superres inference module."""

    def __init__(
            self,
            model_backbone: tf.keras.Model,
            model_superres: tf.keras.Model,
            model_normalizer: tf.keras.Model,
            model_denormalizer: tf.keras.Model,
            cast_to_uint8: bool = True):
        """
        Initializes a module for super resolution.

        :param model_backbone: backbone model to use for inference
        :param model_superres: super resolution model to use for inference.
        :param model_normalizer: model that normalizes the input
        :param model_denormalizer: model that denormalizes the output
        :param cast_to_uint8: cast output to uint8

        """
        super().__init__(name=SUPERRES_STR)

        # --- argument checking
        if model_backbone is None:
            raise ValueError("model_backbone should not be None")
        if model_superres is None:
            raise ValueError("model_superres should not be None")
        if model_normalizer is None:
            raise ValueError("model_normalizer should not be None")
        if model_denormalizer is None:
            raise ValueError("model_denormalizer should not be None")

        # --- setup instance variables
        self._cast_to_uint8 = cast_to_uint8
        self._model_backbone = model_backbone
        self._model_superres = model_superres
        self._model_normalizer = model_normalizer
        self._model_denormalizer = model_denormalizer

    @tf.function
    def __call__(self, image):
        """
        Cast image to float and run inference.

        :param image: uint8 Tensor of shape

        :return: super resolution image: uint8 Tensor of shape if the input
        """
        x = tf.cast(image, dtype=tf.float32)

        # --- normalize
        x = self._model_normalizer(x)

        # --- run backbone
        x = self._model_backbone(x)

        # --- run superres model
        x, _, _ = self._model_superres(x)

        # --- denormalize
        x = self._model_denormalizer(x)

        # --- cast to uint8
        if self._cast_to_uint8:
            x = tf.round(x)
            x = tf.cast(x, dtype=tf.uint8)

        return x

# ---------------------------------------------------------------------
