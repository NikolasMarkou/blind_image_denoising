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

@pytest.mark.parametrize(
    "config", bfcnn.configs)
def test_model_builder(config):
    models = bfcnn.model_builder(config=config["model_denoise"])

    # denoise
    assert isinstance(models[0], keras.Model)
    # normalize
    assert isinstance(models[1], keras.Model)
    # denormalize
    assert isinstance(models[2], keras.Model)
# ---------------------------------------------------------------------
