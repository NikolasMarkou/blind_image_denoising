# ---------------------------------------------------------------------

__author__ = "Nikolas Markou"
__version__ = "1.0.0"
__license__ = "MIT"

# ---------------------------------------------------------------------

import os
import json
import tensorflow as tf
from pathlib import Path
from typing import List, Union, Tuple, Dict

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .constants import *
from .custom_logger import logger
from .utilities import load_config
from .model_denoise import model_builder, module_builder

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
    :param test_model:
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
    output_checkpoint = os.path.join(output_directory, "checkpoint")
    output_saved_model = os.path.join(output_directory, "saved_model")

    # --- load and export denoising model
    logger.info("building denoising model")
    pipeline_config = load_config(pipeline_config)
    models = \
        model_builder(
            pipeline_config[MODEL_DENOISE_STR])
    model_denoise = models.denoiser
    model_normalize = models.normalizer
    model_denormalize = models.denormalizer
    logger.info("saving configuration pipeline")
    pipeline_config_path = \
        os.path.join(
            output_directory,
            "pipeline.json")
    with open(pipeline_config_path, "w") as f:
        f.write(json.dumps(pipeline_config, indent=4))
    logger.info("restoring checkpoint weights")
    checkpoint = tf.train.Checkpoint(model_denoise=model_denoise)
    manager = \
        tf.train.CheckpointManager(
            checkpoint=checkpoint,
            directory=checkpoint_directory,
            max_to_keep=1)
    status = \
        checkpoint.restore(manager.latest_checkpoint).expect_partial()
    status.assert_existing_objects_matched()

    # --- combine denoise, normalize and denormalize
    logger.info("combining denoise, normalize and denormalize model")
    dataset_config = pipeline_config["dataset"]
    input_shape = dataset_config["input_shape"]
    no_channels = input_shape[2]
    denoising_module = \
        module_builder(
            iterations=1,
            cast_to_uint8=True,
            model_denoise=model_denoise,
            model_normalize=model_normalize,
            model_denormalize=model_denormalize,
            training_channels=no_channels)

    # getting the concrete function traces the graph and forces variables to
    # be constructed, only after this can we save the
    # checkpoint and saved model.
    concrete_function = \
        denoising_module.__call__.get_concrete_function(
            tf.TensorSpec(
                shape=[None, None, None] + [no_channels],
                dtype=tf.uint8,
                name="input")
        )

    # export the model as save_model format (default)
    logger.info("saving module")
    exported_checkpoint_manager = \
        tf.train.CheckpointManager(
            checkpoint=checkpoint,
            directory=output_checkpoint,
            max_to_keep=1)
    exported_checkpoint_manager.save(checkpoint_number=0)
    options = tf.saved_model.SaveOptions(save_debug_info=True)
    tf.saved_model.save(
        options=options,
        obj=denoising_module,
        signatures=concrete_function,
        export_dir=output_saved_model)

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
            tf.lite.Optimize.OPTIMIZE_FOR_LATENCY
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
        concrete_input_shape = [1] + input_shape
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

# ---------------------------------------------------------------------
