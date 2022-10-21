import os
import sys
import pytest
import numpy as np
import tensorflow as tf
from tensorflow import keras

from .constants import *

sys.path.append(os.getcwd() + "/../")

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

import bfcnn


# ---------------------------------------------------------------------

@pytest.mark.parametrize(
    "target_size", [(32, 32), (64, 64), (128, 128), (256, 256), (512, 512)])
def test_grayscale_load_image(target_size):
    x = \
        bfcnn.load_image(
            path=LENA_IMAGE_PATH,
            color_mode="grayscale",
            target_size=target_size,
            normalize=True)
    assert x.shape == (1,) + target_size + (1,)


# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "target_size", [(32, 32), (64, 64), (128, 128), (256, 256), (512, 512)])
def test_color_load_image(target_size):
    x = \
        bfcnn.load_image(
            path=LENA_IMAGE_PATH,
            color_mode="rgb",
            target_size=target_size,
            normalize=True)
    assert x.shape == (1,) + target_size + (3,)

# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "target_size", [(256, 256), (512, 512), (1024, 1024)])
def test_kitti_load_images(target_size):
    for filename in KITTI_IMAGES:
        x = \
            bfcnn.load_image(
                path=filename,
                color_mode="rgb",
                target_size=target_size,
                normalize=True)
        assert x.shape == (1,) + target_size + (3,)

# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "target_size", [(256, 256), (512, 512), (1024, 1024)])
def test_megadepth_load_images(target_size):
    for filename in MEGADEPTH_IMAGES:
        x = \
            bfcnn.load_image(
                path=filename,
                color_mode="rgb",
                target_size=target_size,
                normalize=True)
        assert x.shape == (1,) + target_size + (3,)

# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "input_dims", [(64, 64, 1), (256, 256, 1), (1024, 1024, 1),
                   (64, 64, 3), (256, 256, 3), (1024, 1024, 3)])
def test_build_normalize_model(input_dims):
    model = \
        bfcnn.utilities.build_normalize_model(
            input_dims=input_dims)
    assert model is not None
    assert isinstance(model, keras.Model)

# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "input_dims", [(64, 64, 1), (256, 256, 1), (1024, 1024, 1),
                   (64, 64, 3), (256, 256, 3), (1024, 1024, 3)])
def test_build_denormalize_model(input_dims):
    model = \
        bfcnn.utilities.build_denormalize_model(
            input_dims=input_dims)
    assert model is not None
    assert isinstance(model, keras.Model)

# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "input_dims", [(1, 64, 64, 1), (2, 256, 256, 1), (3, 1024, 1024, 1),
                   (1, 64, 64, 3), (2, 256, 256, 3), (3, 1024, 1024, 3),
                   (1, 64, 64, 5), (2, 256, 256, 5), (3, 1024, 1024, 5)])
def test_stats_2d_block(input_dims):
    tensor = \
        tf.random.truncated_normal(
            seed=0,
            mean=0,
            stddev=1,
            shape=input_dims)
    results = bfcnn.utilities.stats_2d_block(tensor)
    results = results.numpy()
    assert len(results.shape) == 2
    assert results.shape[0] == tensor.shape[0]
    assert results.shape[1] == (tensor.shape[3] * 4)
    channels = input_dims[3]

    # max
    results_max = results[0, 0]
    assert results_max < 2.0
    # min
    results_min = results[0, channels]
    assert results_min > -2.0
    # mean
    results_mean = results[0, 2*channels]
    assert np.abs(results_mean) < 0.5


# ---------------------------------------------------------------------

