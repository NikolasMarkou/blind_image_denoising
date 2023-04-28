import tensorflow as tf

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .constants import *

# ---------------------------------------------------------------------


class DenoiserModule(tf.Module):
    """denoising inference module."""

    def __init__(
            self,
            model_hydra: tf.keras.Model,
            cast_to_uint8: bool = True):
        """
        Initializes a module for denoising.

        :param model_hydra:
        :param cast_to_uint8: cast output to uint8

        """
        super().__init__(name=DENOISER_STR)

        # --- argument checking
        if model_hydra is None:
            raise ValueError("model_denoise should not be None")

        # --- setup instance variables
        self._cast_to_uint8 = cast_to_uint8
        self._model_hydra = model_hydra

    @tf.function
    def __call__(self, image):
        """
        Cast image to float and run inference.

        :param image: uint8 Tensor of shape
        :return: denoised image: uint8 Tensor of shape if the input
        """
        x = tf.cast(image, dtype=tf.float32)

        x = self._model_hydra(x)[0]

        # --- cast to uint8
        if self._cast_to_uint8:
            x = tf.round(x)
            x = tf.cast(x, dtype=tf.uint8)

        return x

# ---------------------------------------------------------------------
