import keras
import pytest

import os
import sys
import numpy as np

from .constants import *

sys.path.append(os.getcwd() + "/../")

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

import bfcnn


# ---------------------------------------------------------------------


def test_model_builder():
    config = {
        "filters": 16,
        "no_layers": 5,
        "min_value": 0,
        "add_var": False,
        "max_value": 255,
        "kernel_size": 5,
        "type": "resnet",
        "batchnorm": True,
        "activation": "elu",
        "clip_values": False,
        "shared_model": False,
        "output_multiplier": 1.0,
        "local_normalization": -1,
        "kernel_regularizer": "l1",
        "final_activation": "tanh",
        "add_skip_with_input": True,
        "add_residual_between_models": False,
        "input_shape": ["?", "?", 3],
        "kernel_initializer": "glorot_normal",
        "pyramid": {
            "levels": 2,
            "type": "laplacian",
            "xy_max": [1.0, 1.0],
            "kernel_size": [3, 3]
        },
        "inverse_pyramid": {
            "levels": 2,
            "type": "laplacian",
            "xy_max": [1.0, 1.0],
            "kernel_size": [3, 3]
        }
    }

    model_denoise, \
    model_normalize, \
    model_denormalize, \
    model_pyramid, \
    model_inverse_pyramid = \
        bfcnn.model_builder(config=config)

    assert isinstance(model_denoise, keras.Model)
    assert isinstance(model_normalize, keras.Model)
    assert isinstance(model_denormalize, keras.Model)
    assert isinstance(model_pyramid, keras.Model)
    assert isinstance(model_inverse_pyramid, keras.Model)

# ---------------------------------------------------------------------
