import os
import sys
import pytest
import numpy as np

# ---------------------------------------------------------------------

from .constants import *

sys.path.append(os.getcwd() + "/../")

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

import bfcnn
from bfcnn.pruning import \
    prune_strategy_helper, \
    PruneStrategy, \
    reshape_to_4d_to_2d, \
    reshape_to_2d_to_4d


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
            config=prune_config["strategies"])

    assert pruning_fn is not None
    assert callable(pruning_fn)


# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "config", bfcnn.configs)
def test_build_model_and_prune_configs(config):
    models = bfcnn.model_builder(config=config["model_denoise"])
    train_config = config["train"]
    if "prune" not in train_config:
        return
    prune_config = train_config["prune"]
    pruning_fn = \
        bfcnn.pruning.prune_function_builder(
            config=prune_config["strategies"])
    _ = pruning_fn(models.denoiser)


# ---------------------------------------------------------------------

# X,Y,C,F
KERNEL_SHAPES = [
    (1, 1, 16, 16),
    (5, 5, 3, 16),
    (3, 3, 16, 32),
    (7, 7, 16, 32),
]

# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape", KERNEL_SHAPES)
def test_reshape_4d_to_2d_to_4d(shape):
    x = np.random.random(size=shape)
    x_2d, x_shape = reshape_to_4d_to_2d(x)
    x_4d = reshape_to_2d_to_4d(x_2d, x_shape)
    assert x_4d.shape == x.shape
    assert (x_4d == x).all()

# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape", KERNEL_SHAPES)
def test_none_strategy(shape):
    fn = prune_strategy_helper(strategy=PruneStrategy.NONE)
    x = np.random.random(size=shape)
    x_p = fn(x)
    assert x_p.shape == x.shape
    assert (x == x_p).all()

# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape", KERNEL_SHAPES)
def test_minimum_threshold_strategy(shape):
    minimum_threshold = 0.01
    fn = \
        prune_strategy_helper(
            strategy=PruneStrategy.MINIMUM_THRESHOLD,
            minimum_threshold=minimum_threshold)
    x = np.random.random(size=shape)
    x_p = fn(x)
    assert x_p.shape == x.shape
    assert (x[np.abs(x) > minimum_threshold] == x_p[np.abs(x) > minimum_threshold]).all()
    assert (x_p[np.abs(x) < minimum_threshold] == 0).all()

# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape", KERNEL_SHAPES)
def test_minimum_threshold_bifurcate_strategy(shape):
    minimum_threshold = 0.01
    fn = \
        prune_strategy_helper(
            strategy=PruneStrategy.MINIMUM_THRESHOLD_BIFURCATE,
            minimum_threshold=minimum_threshold)
    x = np.random.random(size=shape)
    x_p = fn(x)
    assert x_p.shape == x.shape
    assert (x[np.abs(x) > minimum_threshold] == x_p[np.abs(x) > minimum_threshold]).all()
    assert (x_p[np.abs(x_p) < minimum_threshold] == 0).all()

# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape", KERNEL_SHAPES)
def test_minimum_threshold_shrinkage_strategy(shape):
    shrinkage = 0.9
    minimum_threshold = 0.01
    shrinkage_threshold = 0.1

    fn = \
        prune_strategy_helper(
            strategy=PruneStrategy.MINIMUM_THRESHOLD_SHRINKAGE,
            shrinkage=shrinkage,
            minimum_threshold=minimum_threshold,
            shrinkage_threshold=shrinkage_threshold)
    x = np.random.random(size=shape)
    x_p = fn(x)
    assert x_p.shape == x.shape
    assert (x[np.abs(x) > shrinkage_threshold] == x_p[np.abs(x) > shrinkage_threshold]).all()
    assert (x_p[np.abs(x_p) < minimum_threshold] == 0).all()

# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape", KERNEL_SHAPES)
def test_pca_projection_strategy(shape):
    scale = True
    variance = 0.9
    minimum_threshold = 0.01

    fn = \
        prune_strategy_helper(
            strategy=PruneStrategy.PCA_PROJECTION,
            scale=scale,
            variance=variance,
            minimum_threshold=minimum_threshold)
    x = np.random.random(size=shape)
    x_p = fn(x)
    assert x.shape == x_p.shape
    x_2d, _ = reshape_to_4d_to_2d(x)
    x_p_2d, _ = reshape_to_4d_to_2d(x_p)
    assert x_2d.shape == x_p_2d.shape
    assert np.linalg.matrix_rank(x_2d) >= np.linalg.matrix_rank(x_p_2d)

# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape", KERNEL_SHAPES)
def test_drop_bottom_strategy(shape):
    percentage = 0.25

    fn = \
        prune_strategy_helper(
            strategy=PruneStrategy.DROP_BOTTOM,
            percentage=percentage)
    x = np.random.random(size=shape)
    x_p = fn(x)
    assert x.shape == x_p.shape
    x_p = x_p.flatten()
    n_zeros = np.count_nonzero(x_p == 0)
    assert n_zeros >= int(np.round(len(x_p) * percentage))


# ---------------------------------------------------------------------
