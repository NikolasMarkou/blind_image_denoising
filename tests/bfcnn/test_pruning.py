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
def test_get_conv2d_weights(config):
    models = bfcnn.model_builder(config=config["model_denoise"])
    model_weights = bfcnn.pruning.get_conv2d_weights(models.denoiser)
    assert model_weights.shape[0] > 0

# ---------------------------------------------------------------------

@pytest.mark.parametrize(
    "config", bfcnn.configs)
def test_prune_function_builder(config):
    train_config = config["train"]
    if "prune" not in train_config:
        return
    prune_config = train_config["prune"]
    pruning_fn = \
        bfcnn.pruning.prune_function_builder(
            config=prune_config)

    assert pruning_fn is not None
    assert callable(pruning_fn)

# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "config", bfcnn.configs)
def test_build_model_and_prune(config):
    models = bfcnn.model_builder(config=config["model_denoise"])
    train_config = config["train"]
    if "prune" not in train_config:
        return
    prune_config = train_config["prune"]
    pruning_fn = \
        bfcnn.pruning.prune_function_builder(
            config=prune_config)
    _ = pruning_fn(models.denoiser)

# ---------------------------------------------------------------------
