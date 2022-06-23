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
# none
# ---------------------------------------------------------------------

@pytest.mark.parametrize(
    "target_size", [(64, 64), (128, 128), (256, 256), (512, 512), (1024, 1024)])
def test_none_grayscale_1_level(target_size):
    input_shape = (None, None, 1)

    laplacian_config = None

    levels = 1

    laplacian_pyramid_model = \
        bfcnn.build_pyramid_model(
            input_dims=input_shape,
            config=laplacian_config)

    inverse_laplacian_pyramid_model = \
        bfcnn.build_inverse_pyramid_model(
            input_dims=input_shape,
            config=laplacian_config)

    x = \
        bfcnn.load_image(
            path=LENA_IMAGE_PATH,
            color_mode="grayscale",
            target_size=target_size,
            normalize=True)

    x_pyramid = laplacian_pyramid_model.predict(x)
    assert len(x_pyramid) == levels

    x_recovered = inverse_laplacian_pyramid_model.predict(x_pyramid)
    assert len(x_recovered) == 1

    x_error = np.abs(x_recovered - x)
    assert np.mean(x_error) < 1e-7

# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "target_size", [(64, 64), (128, 128), (256, 256), (512, 512), (1024, 1024)])
def test_none_color_1_level(target_size):
    input_shape = (None, None, 3)

    laplacian_config = None

    levels = 1

    laplacian_pyramid_model = \
        bfcnn.build_pyramid_model(
            input_dims=input_shape,
            config=laplacian_config)

    inverse_laplacian_pyramid_model = \
        bfcnn.build_inverse_pyramid_model(
            input_dims=input_shape,
            config=laplacian_config)

    x = \
        bfcnn.load_image(
            path=LENA_IMAGE_PATH,
            color_mode="rgb",
            target_size=target_size,
            normalize=True)

    x_pyramid = laplacian_pyramid_model.predict(x)
    assert len(x_pyramid) == levels

    x_recovered = inverse_laplacian_pyramid_model.predict(x_pyramid)
    assert len(x_recovered) == 1

    x_error = np.abs(x_recovered - x)
    assert np.mean(x_error) < 1e-7

# ---------------------------------------------------------------------
# laplacian
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "target_size", [(64, 64), (128, 128), (256, 256), (512, 512), (1024, 1024)])
def test_laplacian_grayscale_1_level(target_size):
    input_shape = (None, None, 1)

    laplacian_config = {
        "levels": 1,
        "type": "laplacian",
        "xy_max": (1.0, 1.0),
        "kernel_size": (3, 3)
    }

    levels = laplacian_config["levels"]

    laplacian_pyramid_model = \
        bfcnn.build_pyramid_model(
            input_dims=input_shape,
            config=laplacian_config)

    inverse_laplacian_pyramid_model = \
        bfcnn.build_inverse_pyramid_model(
            input_dims=input_shape,
            config=laplacian_config)

    x = \
        bfcnn.load_image(
            path=LENA_IMAGE_PATH,
            color_mode="grayscale",
            target_size=target_size,
            normalize=True)

    x_pyramid = laplacian_pyramid_model.predict(x)
    assert len(x_pyramid) == levels

    x_recovered = inverse_laplacian_pyramid_model.predict(x_pyramid)
    assert len(x_recovered) == 1

    x_error = np.abs(x_recovered - x)
    assert np.mean(x_error) < 1e-7


# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "target_size", [(64, 64), (128, 128), (256, 256), (512, 512), (1024, 1024)])
def test_laplacian_grayscale_3_level(target_size):
    input_shape = (None, None, 1)

    laplacian_config = {
        "levels": 3,
        "type": "laplacian",
        "xy_max": (1.0, 1.0),
        "kernel_size": (3, 3)
    }

    levels = laplacian_config["levels"]

    laplacian_pyramid_model = \
        bfcnn.build_pyramid_model(
            input_dims=input_shape,
            config=laplacian_config)

    inverse_laplacian_pyramid_model = \
        bfcnn.build_inverse_pyramid_model(
            input_dims=input_shape,
            config=laplacian_config)

    x = \
        bfcnn.load_image(
            path=LENA_IMAGE_PATH,
            color_mode="grayscale",
            target_size=target_size,
            normalize=True)

    x_pyramid = laplacian_pyramid_model.predict(x)
    assert len(x_pyramid) == levels

    x_recovered = inverse_laplacian_pyramid_model.predict(x_pyramid)
    assert len(x_recovered) == 1

    x_error = np.abs(x_recovered - x)
    assert np.mean(x_error) < 1e-7


# ---------------------------------------------------------------------

@pytest.mark.parametrize(
    "target_size", [(64, 64), (128, 128), (256, 256), (512, 512), (1024, 1024)])
def test_laplacian_color_1_level(target_size):
    input_shape = (None, None, 3)

    laplacian_config = {
        "levels": 1,
        "type": "laplacian",
        "xy_max": (1.0, 1.0),
        "kernel_size": (3, 3)
    }

    levels = laplacian_config["levels"]

    laplacian_pyramid_model = \
        bfcnn.build_pyramid_model(
            input_dims=input_shape,
            config=laplacian_config)

    inverse_laplacian_pyramid_model = \
        bfcnn.build_inverse_pyramid_model(
            input_dims=input_shape,
            config=laplacian_config)

    x = \
        bfcnn.load_image(
            path=LENA_IMAGE_PATH,
            color_mode="rgb",
            target_size=target_size,
            normalize=True)

    x_pyramid = laplacian_pyramid_model.predict(x)
    assert len(x_pyramid) == levels

    x_recovered = inverse_laplacian_pyramid_model.predict(x_pyramid)
    assert len(x_recovered) == 1

    x_error = np.abs(x_recovered - x)
    assert np.mean(x_error) < 1e-7


# ---------------------------------------------------------------------

@pytest.mark.parametrize(
    "target_size", [(64, 64), (128, 128), (256, 256), (512, 512), (1024, 1024)])
def test_laplacian_color_3_level(target_size):
    input_shape = (None, None, 3)

    laplacian_config = {
        "levels": 3,
        "type": "laplacian",
        "xy_max": (1.0, 1.0),
        "kernel_size": (3, 3)
    }

    levels = laplacian_config["levels"]

    laplacian_pyramid_model = \
        bfcnn.build_pyramid_model(
            input_dims=input_shape,
            config=laplacian_config)

    inverse_laplacian_pyramid_model = \
        bfcnn.build_inverse_pyramid_model(
            input_dims=input_shape,
            config=laplacian_config)

    x = \
        bfcnn.load_image(
            path=LENA_IMAGE_PATH,
            color_mode="rgb",
            target_size=target_size,
            normalize=True)

    x_pyramid = laplacian_pyramid_model.predict(x)
    assert len(x_pyramid) == levels

    x_recovered = inverse_laplacian_pyramid_model.predict(x_pyramid)
    assert len(x_recovered) == 1

    x_error = np.abs(x_recovered - x)
    assert np.mean(x_error) < 1e-7

# ---------------------------------------------------------------------
# gaussian
# ---------------------------------------------------------------------

@pytest.mark.parametrize(
    "target_size", [(64, 64), (128, 128), (256, 256), (512, 512), (1024, 1024)])
def test_gaussian_grayscale_3_level(target_size):
    input_shape = (None, None, 1)

    laplacian_config = {
        "levels": 3,
        "type": "gaussian",
        "xy_max": (1.0, 1.0),
        "kernel_size": (3, 3)
    }

    levels = laplacian_config["levels"]

    gaussian_pyramid_model = \
        bfcnn.build_pyramid_model(
            input_dims=input_shape,
            config=laplacian_config)

    inverse_gaussian_pyramid_model = \
        bfcnn.build_inverse_pyramid_model(
            input_dims=input_shape,
            config=laplacian_config)

    x = \
        bfcnn.load_image(
            path=LENA_IMAGE_PATH,
            color_mode="grayscale",
            target_size=target_size,
            normalize=True)

    x_pyramid = gaussian_pyramid_model.predict(x)
    assert len(x_pyramid) == levels

    x_recovered = inverse_gaussian_pyramid_model.predict(x_pyramid)
    assert len(x_recovered) == 1

    x_error = np.abs(x_recovered - x)
    assert np.mean(x_error) < 1e-7


# ---------------------------------------------------------------------

@pytest.mark.parametrize(
    "target_size", [(64, 64), (128, 128), (256, 256), (512, 512), (1024, 1024)])
def test_gaussian_color_1_level(target_size):
    input_shape = (None, None, 3)

    laplacian_config = {
        "levels": 1,
        "type": "gaussian",
        "xy_max": (1.0, 1.0),
        "kernel_size": (3, 3)
    }

    levels = laplacian_config["levels"]

    gaussian_pyramid_model = \
        bfcnn.build_pyramid_model(
            input_dims=input_shape,
            config=laplacian_config)

    inverse_gaussian_pyramid_model = \
        bfcnn.build_inverse_pyramid_model(
            input_dims=input_shape,
            config=laplacian_config)

    x = \
        bfcnn.load_image(
            path=LENA_IMAGE_PATH,
            color_mode="rgb",
            target_size=target_size,
            normalize=True)

    x_pyramid = gaussian_pyramid_model.predict(x)
    assert len(x_pyramid) == levels

    x_recovered = inverse_gaussian_pyramid_model.predict(x_pyramid)
    assert len(x_recovered) == 1

    x_error = np.abs(x_recovered - x)
    assert np.mean(x_error) < 1e-7


# ---------------------------------------------------------------------

@pytest.mark.parametrize(
    "target_size", [(64, 64), (128, 128), (256, 256), (512, 512), (1024, 1024)])
def test_gaussian_color_3_level(target_size):
    input_shape = (None, None, 3)

    laplacian_config = {
        "levels": 3,
        "type": "gaussian",
        "xy_max": (1.0, 1.0),
        "kernel_size": (3, 3)
    }

    levels = laplacian_config["levels"]

    gaussian_pyramid_model = \
        bfcnn.build_pyramid_model(
            input_dims=input_shape,
            config=laplacian_config)

    inverse_gaussian_pyramid_model = \
        bfcnn.build_inverse_pyramid_model(
            input_dims=input_shape,
            config=laplacian_config)

    x = \
        bfcnn.load_image(
            path=LENA_IMAGE_PATH,
            color_mode="rgb",
            target_size=target_size,
            normalize=True)

    x_pyramid = gaussian_pyramid_model.predict(x)
    assert len(x_pyramid) == levels

    x_recovered = inverse_gaussian_pyramid_model.predict(x_pyramid)
    assert len(x_recovered) == 1

    x_error = np.abs(x_recovered - x)
    assert np.mean(x_error) < 1e-7

# ---------------------------------------------------------------------
