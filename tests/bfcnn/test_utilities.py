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
