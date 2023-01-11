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
    "config", [{
        "batch_size": 2,
        "value_range": [0, 255],
        "clip_value": True,
        "random_blur": True,
        "round_values": True,
        "random_invert": True,
        "random_rotate": 0.314,
        "random_up_down": True,
        "color_mode": "grayscale",
        "random_left_right": True,
        "input_shape": [256, 256, 1],
        "multiplicative_noise": [0.1, 0.2],
        "additional_noise": [1, 5, 10, 20, 40],
        "inputs": [{
            "directory": str(KITTI_DIR)
        }]
    }, {
        "batch_size": 2,
        "value_range": [0, 255],
        "clip_value": True,
        "random_blur": True,
        "round_values": True,
        "random_invert": True,
        "random_rotate": 0.314,
        "random_up_down": True,
        "color_mode": "rgb",
        "random_left_right": True,
        "input_shape": [256, 256, 3],
        "multiplicative_noise": [0.1, 0.2],
        "additional_noise": [1, 5, 10, 20, 40],
        "inputs": [{
            "directory": str(KITTI_DIR)
        }]
    }, {
        "batch_size": 2,
        "value_range": [0, 255],
        "clip_value": True,
        "random_blur": True,
        "round_values": True,
        "random_invert": True,
        "random_rotate": 0.314,
        "random_up_down": True,
        "color_mode": "grayscale",
        "random_left_right": True,
        "input_shape": [128, 128, 1],
        "multiplicative_noise": [0.1, 0.2],
        "additional_noise": [1, 5, 10, 20, 40],
        "inputs": [{
            "directory": str(MEGADEPTH_DIR)
        }]
    }, {
        "batch_size": 2,
        "value_range": [0, 255],
        "clip_value": True,
        "random_blur": True,
        "round_values": True,
        "random_invert": True,
        "random_rotate": 0.314,
        "random_up_down": True,
        "color_mode": "rgb",
        "random_left_right": True,
        "input_shape": [64, 64, 3],
        "multiplicative_noise": [0.1, 0.2],
        "additional_noise": [1, 5, 10, 20, 40],
        "inputs": [{
            "directory": str(MEGADEPTH_DIR)
        }]
    }, {
        "batch_size": 2,
        "value_range": [0, 255],
        "clip_value": True,
        "random_blur": True,
        "round_values": True,
        "random_invert": True,
        "random_rotate": 0.314,
        "random_up_down": True,
        "color_mode": "rgb",
        "random_left_right": True,
        "input_shape": [64, 64, 3],
        "multiplicative_noise": [0.1, 0.2],
        "additional_noise": [1, 5, 10, 20, 40],
        "inputs": [{
            "directory": str(MEGADEPTH_DIR)
        }, {
            "directory": str(KITTI_DIR)
        }]
    }])
def test_dataset_builder_build(config):
    dataset_training = bfcnn.dataset.dataset_builder(config=config)

    for (input_batch, noisy_batch, downsampled_batch) in dataset_training:
        assert input_batch.shape[0] <= config["batch_size"]
        assert input_batch.shape[1] == config["input_shape"][0]
        assert input_batch.shape[2] == config["input_shape"][1]

        assert noisy_batch.shape[0] <= config["batch_size"]
        assert noisy_batch.shape[1] == config["input_shape"][0] / 2
        assert noisy_batch.shape[2] == config["input_shape"][1] / 2

        assert downsampled_batch.shape[0] <= config["batch_size"]
        assert downsampled_batch.shape[1] == config["input_shape"][0] / 2
        assert downsampled_batch.shape[2] == config["input_shape"][1] / 2

        if config["color_mode"] == "grayscale":
            assert input_batch.shape[3] == 1
        if config["color_mode"] == "rgb":
            assert input_batch.shape[3] == 3
        if config["color_mode"] == "rgba":
            assert input_batch.shape[3] == 4

        assert np.min(input_batch) >= config["value_range"][0]
        assert np.max(input_batch) <= config["value_range"][1]
        assert input_batch.dtype == np.float32

# ---------------------------------------------------------------------

