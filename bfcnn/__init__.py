# ---------------------------------------------------------------------

__author__ = "Nikolas Markou"
__version__ = "0.1.0"
__license__ = "None"

# ---------------------------------------------------------------------

import os
import pathlib
import tensorflow as tf
from .model import model_builder
from .train_loop import train_loop
from .export_model import export_model

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

__all__ = [
    train_loop,
    export_model,
    model_builder,
    pretrained_models
]

# ---------------------------------------------------------------------
