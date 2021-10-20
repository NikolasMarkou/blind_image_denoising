# ---------------------------------------------------------------------

__author__ = "Nikolas Markou"
__version__ = "0.1.0"
__license__ = "None"

# ---------------------------------------------------------------------

import os
import json
import tensorflow as tf
from pathlib import Path
from typing import List, Union, Tuple, Dict

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .custom_logger import logger
from .utilities import load_config
from .model import model_builder, DenoisingInferenceModule

# ---------------------------------------------------------------------

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# ---------------------------------------------------------------------


def export_model(
        pipeline_config: Union[str, Dict, Path],
        checkpoint_directory: Union[str, Path],
        output_directory: Union[str, Path],
        input_shape: List[int] = [1, 256, 768, 3],
        to_tflite: bool = True,
        test_model: bool = True):
    """
    build and export a denoising model

    :param pipeline_config: path or dictionary of a configuration
    :param checkpoint_directory: path to the checkpoint directory
    :param output_directory: path to the output directory
    :param input_shape:
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
    model_denoise, model_normalize, model_denormalize = \
        model_builder(
            pipeline_config["model"])
    logger.info("saving configuration pipeline")
    with open(
            os.path.join(
                output_directory,
                "pipeline.json"), "w") as f:
        f.write(json.dumps(pipeline_config, indent=4))
    logger.info("restoring checkpoint weights")
    checkpoint = tf.train.Checkpoint(model_denoise=model_denoise)
    manager = tf.train.CheckpointManager(
        checkpoint, checkpoint_directory, max_to_keep=1)
    status = checkpoint.restore(manager.latest_checkpoint).expect_partial()
    status.assert_existing_objects_matched()

    # --- combine denoise, normalize and denormalize
    logger.info("combining denoise, normalize and denormalize model")
    denoising_module = \
        DenoisingInferenceModule(
            model_denoise=model_denoise,
            model_normalize=model_normalize,
            model_denormalize=model_denormalize)

    # getting the concrete function traces the graph and forces variables to
    # be constructed, only after this can we save the
    # checkpoint and saved model.
    concrete_function = \
        denoising_module.__call__.get_concrete_function(
            tf.TensorSpec(
                shape=input_shape,
                dtype=tf.uint8,
                name="input")
        )

    logger.info("saving model")
    # export the model as save_model format (default)
    exported_checkpoint_manager = \
        tf.train.CheckpointManager(
            checkpoint,
            output_checkpoint,
            max_to_keep=1)
    exported_checkpoint_manager.save(checkpoint_number=0)
    options = tf.saved_model.SaveOptions(save_debug_info=True)
    tf.saved_model.save(
        denoising_module,
        output_saved_model,
        signatures=concrete_function,
        options=options)

    # --- export to tflite
    if to_tflite:
        converter = \
            tf.lite.TFLiteConverter.from_concrete_functions(
                [concrete_function])
        converter.target_spec.supported_ops = \
            [tf.lite.OpsSet.TFLITE_BUILTINS]
        converter.optimizations = \
            [tf.lite.Optimize.OPTIMIZE_FOR_LATENCY]
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
        logger.info("testing modes with shape [{0}]".format(input_shape))
        output_log = \
            os.path.join(output_directory, "trace_log")
        writer = \
            tf.summary.create_file_writer(
                output_log)

        # sample data for your function.
        input_tensor = \
            tf.random.uniform(
                shape=input_shape,
                minval=0,
                maxval=255,
                dtype=tf.int32)
        input_tensor = \
            tf.cast(
                input_tensor,
                dtype=tf.uint8)

        # Bracket the function call with
        # tf.summary.trace_on() and tf.summary.trace_export().
        tf.summary.trace_on(graph=True, profiler=False)
        # Call only one tf.function when tracing.
        _ = concrete_function(input_tensor)
        with writer.as_default():
            tf.summary.trace_export(
                step=0,
                profiler_outdir=output_log,
                name="denoising_module")

# ---------------------------------------------------------------------
