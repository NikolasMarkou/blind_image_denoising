import keras
import pytest

import os
import sys
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
