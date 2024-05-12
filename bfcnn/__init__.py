__author__ = "Nikolas Markou"
__version__ = "3.2.0"
__license__ = "MIT"

# ---------------------------------------------------------------------

import os
import pathlib
import tensorflow as tf

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .constants import \
    DENOISER_STR, SUPERRES_STR, INPAINT_STR
from .train_loop import train_loop
from .export_model import export_model
from .model import model_builder
from .file_operations import load_image
from .utilities import logger, load_config
from .pyramid import \
    build_pyramid_model, \
    build_inverse_pyramid_model
from .optimizer import \
    schedule_builder, \
    optimizer_builder
from .custom_layers import \
    RandomOnOff, \
    Multiplier, \
    ChannelwiseMultiplier

# ---------------------------------------------------------------------

current_dir = pathlib.Path(__file__).parent.resolve()

# ---------------------------------------------------------------------

configs_dir = current_dir / "configs"

configs = [
    (os.path.basename(str(c)), load_config(str(c)))
    for c in configs_dir.glob("*.json")
]

CONFIGS_DICT = {
    os.path.splitext(os.path.basename(str(c)))[0]: load_config(str(c))
    for c in configs_dir.glob("*.json")
}

# ---------------------------------------------------------------------

pretrained_dir = current_dir / "pretrained"

models = {}

# --- populate pretrained_models
if pretrained_dir.is_dir():
    for directory in [d for d in pretrained_dir.iterdir() if d.is_dir()]:
        # ---
        model_name = str(directory.name)

        # --- define model loader function
        def load_denoiser_module():
            return tf.saved_model.load(str(directory / DENOISER_STR))

        def load_superres_module():
            return tf.saved_model.load(str(directory / SUPERRES_STR))

        # --- define structure for each model
        models[model_name] = {
            "directory": directory,
            SUPERRES_STR: load_superres_module,
            DENOISER_STR: load_denoiser_module,
            "configuration": str(directory / "pipeline.json"),
            "saved_model_path": str(directory / "saved_model"),
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
    if model_path in models:
        return \
            tf.saved_model.load(
                models[model_path]["saved_model_path"])

    # --- load from any directory
    if not os.path.exists(model_path):
        raise ValueError(
            "model_path [{0}] does not exist".format(model_path))

    return tf.saved_model.load(str(model_path))


# ---------------------------------------------------------------------


def load_denoiser_model(model_path: str):
    # --- argument checking
    if model_path is None or len(model_path) <= 0:
        raise ValueError("model_path cannot be empty")

    # --- load from pretrained
    if model_path in models:
        return models[model_path][DENOISER_STR]()

    raise ValueError("model_path [{0}] does not exist".format(model_path))


# ---------------------------------------------------------------------


def load_superres_model(model_path: str):
    # --- argument checking
    if model_path is None or len(model_path) <= 0:
        raise ValueError("model_path cannot be empty")

    # --- load from pretrained
    if model_path in models:
        return models[model_path][SUPERRES_STR]()

    raise ValueError("model_path [{0}] does not exist".format(model_path))

# ---------------------------------------------------------------------


# offer a descent pretrained model fore each
if len(models) > 0:
    load_default_denoiser = list(models.values())[0][DENOISER_STR]
    load_default_superres = list(models.values())[0][SUPERRES_STR]
else:
    load_default_denoiser = None
    load_default_superres = None

# ---------------------------------------------------------------------


__all__ = [
    models,
    configs,
    train_loop,
    load_model,
    load_image,
    export_model,
    model_builder,
    schedule_builder,
    optimizer_builder,
    load_denoiser_model,
    load_superres_model,
    build_pyramid_model,
    load_default_denoiser,
    load_default_superres,
    build_inverse_pyramid_model
]

# ---------------------------------------------------------------------
