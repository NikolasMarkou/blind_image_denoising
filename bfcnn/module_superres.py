import os
import tensorflow as tf
from tensorflow import keras

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .constants import *
from .custom_logger import logger
from .module_interface import ModuleInterface

# ---------------------------------------------------------------------

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# ---------------------------------------------------------------------


class SuperresModule(tf.Module, ModuleInterface):
    """superres inference module."""

    def __init__(
            self,
            model_backbone: keras.Model,
            model_superres: keras.Model,
            model_normalizer: keras.Model,
            model_denormalizer: keras.Model,
            training_channels: int = 1,
            cast_to_uint8: bool = True):
        """
        Initializes a module for super resolution.

        :param model_backbone: backbone model to use for inference
        :param model_superres: super resolution model to use for inference.
        :param model_normalizer: model that normalizes the input
        :param model_denormalizer: model that denormalizes the output
        :param training_channels: how many color channels were used in training
        :param cast_to_uint8: cast output to uint8

        """
        # --- argument checking
        if model_backbone is None:
            raise ValueError("model_backbone should not be None")
        if model_superres is None:
            raise ValueError("model_superres should not be None")
        if model_normalizer is None:
            raise ValueError("model_normalizer should not be None")
        if model_denormalizer is None:
            raise ValueError("model_denormalizer should not be None")
        if training_channels <= 0:
            raise ValueError("training channels should be > 0")

        # --- setup instance variables
        self._cast_to_uint8 = cast_to_uint8
        self._model_backbone = model_backbone
        self._model_superres = model_superres
        self._model_normalizer = model_normalizer
        self._model_denormalizer = model_denormalizer
        self._training_channels = training_channels

    def _run_inference_on_images(self, image):
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
        x = self._model_superres(x)

        # --- denormalize
        x = self._model_denormalizer(x)

        # --- cast to uint8
        if self._cast_to_uint8:
            x = tf.round(x)
            x = tf.cast(x, dtype=tf.uint8)

        return x

    def __call__(self, input_tensor):
        return self._run_inference_on_images(input_tensor)

    def concrete_tensor_spec(self):
        return \
            tf.TensorSpec(
                shape=[None, None, None] + [self._training_channels],
                dtype=tf.uint8,
                name="input")

    def description(self) -> str:
        return \
            "BiasFree CNN, Super Resolution module,\n" \
            "takes in an image and doubles its resolution.\n" \
            f"This module was trained for [{self._training_channels}] channels,\n" \
            f"This module expects tf.uint8 input and products tf.uint8 output by default."

    def test(self, output_directory: str):
        # --- argument checking
        if not output_directory or len(output_directory) <= 0:
            raise ValueError("output_directory must not be empty")

        # ---
        concrete_input_shape = [1, 128, 128, self._training_channels]
        logger.info("testing modes with shape [{0}]".format(concrete_input_shape))
        output_log = \
            os.path.join(output_directory, "trace_log")
        writer = \
            tf.summary.create_file_writer(
                output_log)

        # sample data for your function.
        input_tensor = \
            tf.random.uniform(
                shape=concrete_input_shape,
                minval=0,
                maxval=255,
                dtype=tf.int32)
        input_tensor = \
            tf.cast(
                input_tensor,
                dtype=tf.uint8)
        # Bracket the function call with
        tf.summary.trace_on(graph=True, profiler=False)
        # Call only one tf.function when tracing.
        _ = self.concrete_function()(input_tensor)
        with writer.as_default():
            tf.summary.trace_export(
                step=0,
                name="superres_module",
                profiler_outdir=output_log)

# ---------------------------------------------------------------------


def module_builder_superres(
        model_backbone: keras.Model = None,
        model_superres: keras.Model = None,
        model_normalizer: keras.Model = None,
        model_denormalizer: keras.Model = None,
        cast_to_uint8: bool = True) -> SuperresModule:
    """
    builds a module for denoising.

    :param model_backbone: backbone model
    :param model_superres: superres model to use for inference.
    :param model_normalizer: model that normalizes the input
    :param model_denormalizer: model that denormalizes the output
    :param cast_to_uint8: cast output to uint8

    :return: superres module
    """
    logger.info(
        f"building superres module with "
        f"cast_to_uint8:{cast_to_uint8}")

    # --- argument checking
    # TODO

    training_channels = \
        model_backbone.input_shape[-1]

    return \
        SuperresModule(
            model_backbone=model_backbone,
            model_superres=model_superres,
            model_normalizer=model_normalizer,
            model_denormalizer=model_denormalizer,
            cast_to_uint8=cast_to_uint8,
            training_channels=training_channels)

# ---------------------------------------------------------------------
