__author__ = "Nikolas Markou"
__version__ = "2.0.1"
__license__ = "MIT"

# ---------------------------------------------------------------------

import os
import pathlib
import tensorflow as tf

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .train_loop import train_loop
from .model_hydra import model_builder
from .utilities import logger, load_config, load_image
from .export_model import export_model, module_builder_denoise
from .pyramid import \
    build_pyramid_model, \
    build_inverse_pyramid_model
from .optimizer import \
    schedule_builder, \
    optimizer_builder
from .custom_layers import \
    GeluLayer, \
    RandomOnOff, \
    Multiplier, \
    DifferentiableReluLayer, \
    ChannelwiseMultiplier

# ---------------------------------------------------------------------

current_dir = pathlib.Path(__file__).parent.resolve()

# ---------------------------------------------------------------------

configs_dir = current_dir / "configs"

CONFIGS = [
    (os.path.basename(str(c)), load_config(str(c)))
    for c in configs_dir.glob("*.json")
]

CONFIGS_DICT = {
    os.path.splitext(os.path.basename(str(c)))[0]: load_config(str(c))
    for c in configs_dir.glob("*.json")
}

# ---------------------------------------------------------------------

pretrained_dir = current_dir / "pretrained"

pretrained_models = {}

# --- populate pretrained_models
if pretrained_dir.is_dir():
    for directory in \
            [d for d in pretrained_dir.iterdir() if d.is_dir()]:
        # ---
        model_name = str(directory.name)

        # --- define model loader function
        def load_tf():
            saved_model_path = str(directory / "saved_model")
            return tf.saved_model.load(saved_model_path)

        # --- define structure for each model
        pretrained_models[model_name] = {
            "load_tf": load_tf,
            "directory": directory,
            "tflite": str(directory / "model.tflite"),
            "configuration": str(directory / "pipeline.json"),
            "saved_model_path": str(directory / "saved_model"),
            "tf": str(directory / "saved_model" / "saved_model.pb")
        }
else:
    logger.info(
        "pretrained directory [{0}] not found".format(pretrained_dir))

# ---------------------------------------------------------------------


def load_model(model_path: str):
    # --- argument checking
    if model_path is None or len(model_path) <= 0:
        raise ValueError("model_path cannot be empty")

    # --- load from pretrained
    if model_path in pretrained_models:
        return \
            tf.saved_model.load(
                pretrained_models[model_path]["saved_model_path"])

    # --- load from any directory
    if not os.path.exists(model_path):
        raise ValueError(
            "model_path [{0}] does not exist".format(model_path))

    return tf.saved_model.load(str(model_path))

# ---------------------------------------------------------------------


__all__ = [
    CONFIGS,
    train_loop,
    load_model,
    load_image,
    export_model,
    model_builder,
    schedule_builder,
    optimizer_builder,
    pretrained_models,
    build_pyramid_model,
    module_builder_denoise,
    build_inverse_pyramid_model
]


# ---------------------------------------------------------------------
