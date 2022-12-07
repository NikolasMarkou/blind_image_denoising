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
from bfcnn.file_operations import load_image_crop

# ---------------------------------------------------------------------

@pytest.mark.parametrize(
    "target_size", [(32, 32), (64, 64), (128, 128), (256, 256), (512, 512)])
def test_grayscale_load_image(target_size):
    x = \
        bfcnn.load_image(
            path=LENA_IMAGE_PATH,
            num_channels=1,
            image_size=target_size,
            expand_dims=True,
            normalize=True)
    assert x.shape == (1,) + target_size + (1,)


# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "target_size", [(32, 32), (64, 64), (128, 128), (256, 256), (512, 512)])
def test_color_load_image(target_size):
    x = \
        bfcnn.load_image(
            path=LENA_IMAGE_PATH,
            num_channels=3,
            image_size=target_size,
            expand_dims=True,
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
                num_channels=3,
                image_size=target_size,
                expand_dims=True,
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
                num_channels=3,
                image_size=target_size,
                expand_dims=True,
                normalize=True)
        assert x.shape == (1,) + target_size + (3,)

# ---------------------------------------------------------------------


@pytest.mark.parametrize("num_channels", [1, 3])
@pytest.mark.parametrize("no_crops_per_image", [4, 8, 16, 32, 64])
@pytest.mark.parametrize("crop_size", [(32, 32), (64, 64), (128, 128)])
def test_load_image_crop_default(
        num_channels, no_crops_per_image, crop_size):
    x = \
        load_image_crop(
            path=LENA_IMAGE_PATH,
            num_channels=num_channels,
            image_size=None,
            no_crops_per_image=no_crops_per_image,
            crop_size=crop_size)
    assert x.shape[0] == no_crops_per_image
    assert x.shape[1] == crop_size[0]
    assert x.shape[2] == crop_size[1]
    assert x.shape[3] == num_channels

# ---------------------------------------------------------------------
