import pytest

import os
import sys
import keras
import numpy as np
import tensorflow as tf

from .constants import *
sys.path.append(os.getcwd() + "/../")

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

import bfcnn

# ---------------------------------------------------------------------


def test_laplacian_grayscale_1_level_resnet():
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
            target_size=(256, 256),
            normalize=True)

    x_pyramid = laplacian_pyramid_model.predict(x)
    assert len(x_pyramid) == levels

    x_recovered = inverse_laplacian_pyramid_model.predict(x_pyramid)
    assert len(x_recovered) == 1

    x_error = np.abs(x_recovered - x)
    assert np.mean(x_error) < 1e-7

# ---------------------------------------------------------------------


def test_laplacian_grayscale_3_level_resnet():
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
            target_size=(256, 256),
            normalize=True)

    x_pyramid = laplacian_pyramid_model.predict(x)
    assert len(x_pyramid) == levels

    x_recovered = inverse_laplacian_pyramid_model.predict(x_pyramid)
    assert len(x_recovered) == 1

    x_error = np.abs(x_recovered - x)
    assert np.mean(x_error) < 1e-7

# ---------------------------------------------------------------------


def test_laplacian_color_1_level_resnet():
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
            target_size=(256, 256),
            normalize=True)

    x_pyramid = laplacian_pyramid_model.predict(x)
    assert len(x_pyramid) == levels

    x_recovered = inverse_laplacian_pyramid_model.predict(x_pyramid)
    assert len(x_recovered) == 1

    x_error = np.abs(x_recovered - x)
    assert np.mean(x_error) < 1e-7

# ---------------------------------------------------------------------


def test_laplacian_color_3_level_resnet():
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
            target_size=(256, 256),
            normalize=True)

    x_pyramid = laplacian_pyramid_model.predict(x)
    assert len(x_pyramid) == levels

    x_recovered = inverse_laplacian_pyramid_model.predict(x_pyramid)
    assert len(x_recovered) == 1

    x_error = np.abs(x_recovered - x)
    assert np.mean(x_error) < 1e-7

# ---------------------------------------------------------------------




