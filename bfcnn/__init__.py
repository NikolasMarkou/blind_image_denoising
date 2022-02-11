# ---------------------------------------------------------------------

__author__ = "Nikolas Markou"
__version__ = "1.0.0"
__license__ = "MIT"

# ---------------------------------------------------------------------

import os
import pathlib
import tensorflow as tf
from .train_loop import train_loop
from .export_model import export_model
from .model_denoise import model_builder
from .pyramid import build_pyramid_model

# ---------------------------------------------------------------------

pretrained_models = {}
current_dir = pathlib.Path(__file__).parent.resolve()
pretrained_dir = os.path.join(str(current_dir), "pretrained")

if os.path.exists(pretrained_dir):
    for directory in \
            [f.path for f in os.scandir(pretrained_dir) if f.is_dir()]:
        # ---
        model_name = os.path.split(directory)[-1]

        # --- define model loader function
        def load_tf():
            return tf.keras.models.load_model(
                os.path.join(directory, "saved_model/"))

        # --- define structure for each model
        pretrained_models[model_name] = {
            "load_tf": load_tf,
            "directory": directory,
            "tflite": os.path.join(directory, "model.tflite"),
            "configuration": os.path.join(directory, "pipeline.json"),
            "tf": os.path.join(directory, "saved_model/saved_model.pb")
        }

# ---------------------------------------------------------------------


def load_model(model_path: str):
    if model_path in pretrained_models:
        return pretrained_models[str(model_path)]["load_tf"]()
    return tf.keras.models.load_model(str(model_path))

# ---------------------------------------------------------------------


__all__ = [
    train_loop,
    load_model,
    export_model,
    model_builder,
    pretrained_models,
    build_pyramid_model
]


# ---------------------------------------------------------------------
