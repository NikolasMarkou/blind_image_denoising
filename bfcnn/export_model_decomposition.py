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
from .model_denoise import model_builder, module_denoiser_builder
from .model_decomposition import module_decomposition_builder
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
    build and export a decomposition model

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
    models = \
        model_builder(
            pipeline_config[MODEL_DENOISE_STR])
    # get each model
    denoiser = models.denoiser
    normalizer = models.normalizer
    denormalizer = models.denormalizer
    denoiser_decomposition = models.denoiser_decomposition

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
    checkpoint = \
        tf.train.Checkpoint(
            step=global_step,
            epoch=global_epoch,
            model_denoise=denoiser,
            model_normalize=normalizer,
            model_denormalize=denormalizer,
            model_denoise_decomposition=denoiser_decomposition)
    manager = \
        tf.train.CheckpointManager(
            checkpoint=checkpoint,
            directory=checkpoint_directory,
            max_to_keep=1)
    status = \
        checkpoint.restore(manager.latest_checkpoint).expect_partial()
    status.assert_existing_objects_matched()
    logger.info(f"restored checkpoint "
                f"at epoch [{int(global_epoch)}] "
                f"and step [{int(global_step)}]")

    # --- combine denoise, normalize and denormalize
    logger.info("combining denoise, normalize and denormalize model")
    dataset_config = pipeline_config["dataset"]
    input_shape = dataset_config["input_shape"]
    no_channels = input_shape[2]
    decomposition_module = \
        module_decomposition_builder(
            cast_to_uint8=True,
            model_decomposition=denoiser_decomposition,
            model_normalize=normalizer,
            model_denormalize=denormalizer,
            training_channels=no_channels)

    # getting the concrete function traces the graph
    # and forces variables to
    # be constructed, only after this can we save the
    # checkpoint and saved model.
    concrete_function = \
        decomposition_module.__call__.get_concrete_function(
            tf.TensorSpec(
                shape=[None, None, None] + [no_channels],
                dtype=tf.uint8,
                name="input")
        )

    # export the model as save_model format (default)
    logger.info(f"saving module: [{output_saved_model}]")
    tf.saved_model.save(
        obj=decomposition_module,
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
                name="decomposition_module",
                profiler_outdir=output_log)

    return concrete_function

# ---------------------------------------------------------------------
