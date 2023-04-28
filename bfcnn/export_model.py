import os
import json
import tensorflow as tf
from pathlib import Path
from typing import List, Union, Tuple, Dict

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .constants import *
from .model import model_builder
from .custom_logger import logger
from .optimizer import optimizer_builder
from .module_denoiser import DenoiserModule
from .module_superres import SuperresModule
from .utilities import load_config, create_checkpoint

# ---------------------------------------------------------------------


def export_model(
        pipeline_config_path: Union[str, Dict, Path],
        checkpoint_directory: Union[str, Path],
        output_directory: Union[str, Path],
        to_tflite: bool = True,
        test_model: bool = True):
    """
    build and export a denoising model

    :param pipeline_config_path: path or dictionary of a configuration
    :param checkpoint_directory: path to the checkpoint directory
    :param output_directory: path to the output directory
    :param to_tflite: if true convert to tflite
    :param test_model: if true run model in test mode
    :return:
    """
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

    # --- load configuration
    config = load_config(pipeline_config_path)

    # --- setup variables
    output_directory = str(output_directory)

    # --- load and export denoiser model
    logger.info("building model")
    models = model_builder(config[MODEL_STR])

    # get each model
    hydra = models.hydra
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
        f.write(json.dumps(config, indent=4))
    logger.info(f"restoring checkpoint weights from [{checkpoint_directory}]")

    # checkpoint managing
    ckpt = \
        create_checkpoint(
            model=models.hydra, d=checkpoint_directory)
    checkpoint = \
        tf.train.Checkpoint(
            step=global_step,
            epoch=global_epoch,
            model_hydra=hydra,
            model_backbone=backbone,
            model_denoiser=denoiser,
            model_normalizer=normalizer,
            model_denormalizer=denormalizer)
    status = \
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_directory))
    status.assert_existing_objects_matched()
    #status.assert_consumed()
    logger.info("!!! Found checkpoint to restore !!!")
    logger.info(f"restored checkpoint "
                f"at epoch [{int(global_epoch)}] "
                f"and step [{int(global_step)}]")

    training_channels = backbone.input_shape[-1]

    ##################################################################################
    # save keras model
    ##################################################################################

    logger.info("saving hydra model")
    hydra.save(
        os.path.join(
            output_directory,
            MODEL_HYDRA_DEFAULT_NAME_STR))

    ##################################################################################
    # build denoiser and superres modules
    ##################################################################################

    modules = []

    for m in [(DENOISER_STR, DenoiserModule),
              (SUPERRES_STR, SuperresModule)]:
        # ---
        output_saved_model = os.path.join(output_directory, m[0])
        logger.info(f"building {m[0]} module")

        module = \
            m[1](
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
            module.__call__.get_concrete_function(
                tf.TensorSpec(shape=[1, None, None, training_channels], dtype=tf.uint8)
            )

        # export the model as save_model format (default)
        logger.info(f"saving module: [{output_saved_model}]")
        tf.saved_model.save(
            obj=module,
            signatures=concrete_function,
            export_dir=output_saved_model,
            options=tf.saved_model.SaveOptions(save_debug_info=True))

        # --- export to tflite
        if to_tflite:
            converter = \
                tf.lite.TFLiteConverter.from_concrete_functions(
                    funcs=[concrete_function],
                    trackable_obj=module)
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
                    f"{m[0]}_model.tflite")
            # save the model.
            with open(output_tflite_model, "wb") as f:
                f.write(tflite_model)

        modules.append(module)

    return modules

# ---------------------------------------------------------------------
