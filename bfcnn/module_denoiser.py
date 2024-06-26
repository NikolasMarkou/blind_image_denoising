from abc import ABC
import tensorflow as tf

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .constants import *
from .utilities import (
    pad_to_power_of_2,
    remove_padding)
# ---------------------------------------------------------------------


class DenoiserModule(tf.Module, ABC):
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

    @tf.function(
        reduce_retracing=True,
        jit_compile=True,
        autograph=False,
        input_signature=[
            tf.TensorSpec(shape=[None, None, None, None], dtype=tf.uint8)
        ])
    def __call__(self, image):
        """
        Cast image to float and run inference.

        :param image: uint8 Tensor of shape
        :return: denoised image: uint8 Tensor of shape if the input
        """
        x = tf.cast(image, dtype=tf.float32)

        # add paddings
        x_padded, paddings = pad_to_power_of_2(x);

        # denoise
        x_padded = self._model_hydra(x_padded)

        # get only one input
        if len(self._model_hydra.outputs) > 1:
            x_padded = x_padded[0]
        else:
            pass

        # remove paddings
        x = remove_padding(x_padded, paddings)

        # --- cast to uint8
        if self._cast_to_uint8:
            x = tf.round(x)
            x = tf.cast(x, dtype=tf.uint8)

        return x

# ---------------------------------------------------------------------
