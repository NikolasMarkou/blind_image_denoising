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
from .module_inpaint import InpaintModule
from .module_denoiser import DenoiserModule
from .module_superres import SuperresModule

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

    # --- load and export denoiser model
    logger.info("building model")
    pipeline_config = load_config(pipeline_config)
    models = model_builder(pipeline_config[MODEL_STR])
    # get each model
    hydra = models.hydra
    inpaint = models.inpaint
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
    latest_checkpoint = manager.restore_or_initialize()

    if latest_checkpoint:
        logger.info("!!! Found checkpoint to restore !!!")
        logger.info(f"latest checkpoint [{0}:{1}]".format(
            latest_checkpoint, manager.latest_checkpoint))
        checkpoint \
            .restore(manager.latest_checkpoint) \
            .expect_partial() \
            .assert_existing_objects_matched()
        logger.info(f"restored checkpoint "
                    f"at epoch [{int(global_epoch)}] "
                    f"and step [{int(global_step)}]")
    else:
        raise ValueError("!!! Did NOT find checkpoint to restore !!!")

    training_channels = backbone.input_shape[-1]

    ##################################################################################
    # combine denoiser, normalize and denormalize
    ##################################################################################
    output_saved_model_denoiser = os.path.join(output_directory, "denoiser")
    logger.info("building denoiser module")
    logger.info("combining backbone, denoise, normalize and denormalize model")
    denoiser_module = \
        DenoiserModule(
            cast_to_uint8=True,
            model_backbone=backbone,
            model_denoiser=denoiser,
            model_normalizer=normalizer,
            model_denormalizer=denormalizer)

    # getting the concrete function traces the graph
    # and forces variables to
    # be constructed, only after this can we save the
    # checkpoint and saved model.
    denoiser_concrete_function = \
        denoiser_module.__call__.get_concrete_function(
            tf.TensorSpec(shape=[1, None, None, training_channels], dtype=tf.uint8)
        )

    # export the model as save_model format (default)
    logger.info(f"saving module: [{output_saved_model_denoiser}]")
    tf.saved_model.save(
        obj=denoiser_module,
        signatures=denoiser_concrete_function,
        export_dir=output_saved_model_denoiser,
        options=tf.saved_model.SaveOptions(save_debug_info=False))

    # --- export to tflite
    if to_tflite:
        converter = \
            tf.lite.TFLiteConverter.from_concrete_functions(
                [denoiser_concrete_function])
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
                "denoiser_model.tflite")
        # save the model.
        with open(output_tflite_model, "wb") as f:
            f.write(tflite_model)

    ##################################################################################
    # combine superres, normalize and denormalize
    ##################################################################################

    output_saved_model_superres = os.path.join(output_directory, "superres")
    logger.info("building superres module")
    logger.info("combining backbone, superres, normalize and denormalize model")
    superres_module = \
        SuperresModule(
            cast_to_uint8=True,
            model_backbone=backbone,
            model_superres=superres,
            model_normalizer=normalizer,
            model_denormalizer=denormalizer)

    # getting the concrete function traces the graph
    # and forces variables to
    # be constructed, only after this can we save the
    # checkpoint and saved model.
    superres_concrete_function = \
        superres_module.__call__.get_concrete_function(
            tf.TensorSpec(shape=[1, None, None, training_channels], dtype=tf.uint8)
        )

    # export the model as save_model format (default)
    logger.info(f"saving module: [{output_saved_model_superres}]")
    tf.saved_model.save(
        obj=superres_module,
        signatures=superres_concrete_function,
        export_dir=output_saved_model_superres,
        options=tf.saved_model.SaveOptions(save_debug_info=False))

    # --- export to tflite
    if to_tflite:
        converter = \
            tf.lite.TFLiteConverter.from_concrete_functions(
                [superres_concrete_function])
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
                "superres_model.tflite")
        # save the model.
        with open(output_tflite_model, "wb") as f:
            f.write(tflite_model)

    ##################################################################################
    # combine inpaint, normalize and denormalize
    ##################################################################################

    output_saved_model_inpaint = os.path.join(output_directory, "inpaint")
    logger.info("building inpaint module")
    logger.info("combining backbone, inpaint, normalize and denormalize model")
    inpaint_module = \
        InpaintModule(
            cast_to_uint8=True,
            model_backbone=backbone,
            model_inpaint=inpaint,
            model_normalizer=normalizer,
            model_denormalizer=denormalizer)

    # getting the concrete function traces the graph
    # and forces variables to
    # be constructed, only after this can we save the
    # checkpoint and saved model.
    inpaint_concrete_function = \
        inpaint_module.__call__.get_concrete_function(
            tf.TensorSpec(shape=[1, None, None, training_channels], dtype=tf.uint8),
            tf.TensorSpec(shape=[1, None, None, 1], dtype=tf.uint8)
        )

    # export the model as save_model format (default)
    logger.info(f"saving module: [{output_saved_model_inpaint}]")
    tf.saved_model.save(
        obj=inpaint_module,
        signatures=inpaint_concrete_function,
        export_dir=output_saved_model_inpaint,
        options=tf.saved_model.SaveOptions(save_debug_info=False))

    # --- export to tflite
    if to_tflite:
        converter = \
            tf.lite.TFLiteConverter.from_concrete_functions(
                [inpaint_concrete_function])
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
                "inpaint_model.tflite")
        # save the model.
        with open(output_tflite_model, "wb") as f:
            f.write(tflite_model)

    return \
        denoiser_concrete_function, \
        superres_module, \
        inpaint_module

# ---------------------------------------------------------------------
