import os
import json
from abc import ABC
import tensorflow as tf
from pathlib import Path
from typing import List, Union, Tuple, Dict

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .constants import *
from .custom_logger import logger
from .utilities import load_config
from .model_hydra import model_builder

# ---------------------------------------------------------------------

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


# ---------------------------------------------------------------------


def export_model(
        pipeline_config: Union[str, Dict, Path],
        checkpoint_directory: Union[str, Path],
        output_directory: Union[str, Path],
        to_tflite: bool = True,
        test_model: bool = True):
    """
    build and export a denoising model

    :param pipeline_config: path or dictionary of a configuration
    :param checkpoint_directory: path to the checkpoint directory
    :param output_directory: path to the output directory
    :param to_tflite: if true convert to tflite
    :param test_model: if true run model in test mode
    :return:
    """
    # --- argument checking
    if pipeline_config is None:
        raise ValueError("Pipeline configuration [{0}] is not valid".format(
            pipeline_config))
    if checkpoint_directory is None or \
            not os.path.isdir(str(checkpoint_directory)):
        raise ValueError("Checkpoint directory [{0}] is not valid".format(
            checkpoint_directory))
    if not os.path.isdir(output_directory):
        # if path does not exist attempt to make it
        Path(output_directory).mkdir(parents=True, exist_ok=True)
        if not os.path.isdir(output_directory):
            raise ValueError("Output directory [{0}] is not valid".format(
                output_directory))

    # --- setup variables
    output_directory = str(output_directory)
    output_saved_model = os.path.join(output_directory, "saved_model")

    # --- load and export denoising model
    logger.info("building denoising model")
    pipeline_config = load_config(pipeline_config)
    models = model_builder(pipeline_config[MODEL_STR])
    # get each model
    hydra = models.hydra
    inpaint = models.hydra
    superres = models.superres
    backbone = models.backbone
    denoiser = models.denoiser
    normalizer = models.normalizer
    denormalizer = models.denormalizer

    # --- create the help variables
    global_step = \
        tf.Variable(
            initial_value=0,
            trainable=False,
            dtype=tf.dtypes.int64,
            name="global_step")
    global_epoch = \
        tf.Variable(
            initial_value=0,
            trainable=False,
            dtype=tf.dtypes.int64,
            name="global_epoch")

    # ---
    logger.info("saving configuration pipeline")
    pipeline_config_path = \
        os.path.join(
            output_directory,
            "pipeline.json")
    with open(pipeline_config_path, "w") as f:
        f.write(json.dumps(pipeline_config, indent=4))
    logger.info(f"restoring checkpoint weights from [{checkpoint_directory}]")

    # checkpoint managing
    checkpoint = \
        tf.train.Checkpoint(
            step=global_step,
            epoch=global_epoch,
            model_hydra=hydra,
            model_backbone=backbone,
            model_denoiser=denoiser,
            model_inpaint=inpaint,
            model_superres=superres,
            model_normalizer=normalizer,
            model_denormalizer=denormalizer)
    manager = \
        tf.train.CheckpointManager(
            checkpoint=checkpoint,
            directory=checkpoint_directory,
            max_to_keep=1)
    if checkpoint:
        logger.info("!!! Found checkpoint to restore !!!")
        checkpoint \
            .restore(manager.latest_checkpoint) \
            .expect_partial() \
            .assert_existing_objects_matched()
    else:
        logger.info("!!! Did NOT find checkpoint to restore !!!")

    logger.info(f"restored checkpoint "
                f"at epoch [{int(global_epoch)}] "
                f"and step [{int(global_step)}]")

    # --- combine denoise, normalize and denormalize
    logger.info("combining backbone, denoise, normalize and denormalize model")
    input_shape = tf.keras.backend.int_shape(backbone.inputs[0])
    no_channels = input_shape[-1]
    denoising_module = \
        module_builder_denoise(
            cast_to_uint8=True,
            model_backbone=backbone,
            model_denoiser=denoiser,
            model_normalizer=normalizer,
            model_denormalizer=denormalizer)

    # getting the concrete function traces the graph
    # and forces variables to
    # be constructed, only after this can we save the
    # checkpoint and saved model.

    concrete_function = \
        tf.function(func=denoising_module.__call__).get_concrete_function(
            tf.TensorSpec(
                shape=[None, None, None] + [no_channels],
                dtype=tf.uint8,
                name="input")
        )

    # export the model as save_model format (default)
    logger.info(f"saving module: [{output_saved_model}]")
    tf.saved_model.save(
        obj=denoising_module,
        signatures=concrete_function,
        export_dir=output_saved_model,
        options=tf.saved_model.SaveOptions(save_debug_info=False))

    # --- export to tflite
    if to_tflite:
        converter = \
            tf.lite.TFLiteConverter.from_concrete_functions(
                [concrete_function])
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]
        converter.optimizations = [
            tf.lite.Optimize.DEFAULT
        ]
        tflite_model = converter.convert()
        output_tflite_model = \
            os.path.join(
                output_directory,
                "model.tflite")
        # save the model.
        with open(output_tflite_model, "wb") as f:
            f.write(tflite_model)

    # --- run graph with random input
    if test_model:
        concrete_input_shape = [1, 256, 768, no_channels]
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
        _ = concrete_function(input_tensor)
        with writer.as_default():
            tf.summary.trace_export(
                step=0,
                name="denoising_module",
                profiler_outdir=output_log)

    return concrete_function

# ---------------------------------------------------------------------


class DenoiserModule(tf.Module):
    """denoising inference module."""

    def __init__(
            self,
            model_backbone: keras.Model,
            model_denoiser: keras.Model,
            model_normalizer: keras.Model,
            model_denormalizer: keras.Model,
            training_channels: int = 1,
            cast_to_uint8: bool = True):
        """
        Initializes a module for denoising.

        :param model_backbone: backbone model to use for inference
        :param model_denoiser: denoising model to use for inference.
        :param model_normalizer: model that normalizes the input
        :param model_denormalizer: model that denormalizes the output
        :param training_channels: how many color channels were used in training
        :param cast_to_uint8: cast output to uint8

        """
        # --- argument checking
        if model_backbone is None:
            raise ValueError("model_denoise should not be None")
        if model_denoiser is None:
            raise ValueError("model_denoise should not be None")
        if model_normalizer is None:
            raise ValueError("model_normalize should not be None")
        if model_denormalizer is None:
            raise ValueError("model_denormalize should not be None")
        if training_channels <= 0:
            raise ValueError("training channels should be > 0")

        # --- setup instance variables
        self._cast_to_uint8 = cast_to_uint8
        self._model_backbone = model_backbone
        self._model_denoiser = model_denoiser
        self._model_normalizer = model_normalizer
        self._model_denormalizer = model_denormalizer
        self._training_channels = training_channels

    def _run_inference_on_images(self, image):
        """
        Cast image to float and run inference.

        :param image: uint8 Tensor of shape
        :return: denoised image: uint8 Tensor of shape if the input
        """
        x = tf.cast(image, dtype=tf.float32)

        # --- normalize
        x = self._model_normalizer(x)

        # --- run backbone
        x = self._model_backbone(x)

        # --- run denoise model
        x = self._model_denoiser(x)

        # --- denormalize
        x = self._model_denormalizer(x)

        # --- cast to uint8
        if self._cast_to_uint8:
            x = tf.round(x)
            x = tf.cast(x, dtype=tf.uint8)

        return x

    def __call__(self, input_tensor):
        return self._run_inference_on_images(input_tensor)

# ---------------------------------------------------------------------


def module_builder_denoise(
        model_backbone: keras.Model = None,
        model_denoiser: keras.Model = None,
        model_normalizer: keras.Model = None,
        model_denormalizer: keras.Model = None,
        cast_to_uint8: bool = True) -> DenoiserModule:
    """
    builds a module for denoising.

    :param model_backbone: backbone model
    :param model_denoiser: denoising model to use for inference.
    :param model_normalizer: model that normalizes the input
    :param model_denormalizer: model that denormalizes the output
    :param cast_to_uint8: cast output to uint8

    :return: denoiser module
    """
    logger.info(
        f"building denoising module with "
        f"cast_to_uint8:{cast_to_uint8}")

    # --- argument checking
    # TODO

    training_channels = \
        model_backbone.input_shape[-1]

    return \
        DenoiserModule(
            model_backbone=model_backbone,
            model_denoiser=model_denoiser,
            model_normalizer=model_normalizer,
            model_denormalizer=model_denormalizer,
            cast_to_uint8=cast_to_uint8,
            training_channels=training_channels)

# ---------------------------------------------------------------------
