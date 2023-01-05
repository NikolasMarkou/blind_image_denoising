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


@pytest.mark.parametrize("num_channels", [1, 3])
@pytest.mark.parametrize(
    "target_size", [(32, 32), (64, 64), (128, 128), (256, 256), (512, 512)])
def test_grayscale_load_image(num_channels, target_size):
    x = \
        bfcnn.load_image(
            path=LENA_IMAGE_PATH,
            num_channels=num_channels,
            image_size=target_size,
            expand_dims=True,
            normalize=True)
    assert x.shape[0] == 1
    assert x.shape[1:3] == target_size
    assert x.shape[3] == num_channels

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
