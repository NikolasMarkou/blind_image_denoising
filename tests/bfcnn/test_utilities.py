import os
import sys
import pytest
import numpy as np
import tensorflow as tf
from tensorflow import keras

import bfcnn.model
from .constants import *

sys.path.append(os.getcwd() + "/../")

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

import bfcnn

# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "input_dims", [(64, 64, 1), (256, 256, 1), (1024, 1024, 1),
                   (64, 64, 3), (256, 256, 3), (1024, 1024, 3)])
def test_build_normalize_model(input_dims):
    model = \
        bfcnn.model.build_normalize_model(
            input_dims=input_dims)
    assert model is not None
    assert isinstance(model, keras.Model)

# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "input_dims", [(64, 64, 1), (256, 256, 1), (1024, 1024, 1),
                   (64, 64, 3), (256, 256, 3), (1024, 1024, 3)])
def test_build_denormalize_model(input_dims):
    model = \
        bfcnn.model.build_denormalize_model(
            input_dims=input_dims)
    assert model is not None
    assert isinstance(model, keras.Model)


# ---------------------------------------------------------------------

