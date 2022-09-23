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
        "min_value": 0,
        "max_value": 255,
        "clip_value": True,
        "random_blur": True,
        "subsample_size": 3,
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
            "dataset_shape": [256, 768],
            "directory": str(KITTI_DIR)
        }]
    }, {
        "batch_size": 2,
        "min_value": 0,
        "max_value": 255,
        "clip_value": True,
        "random_blur": True,
        "subsample_size": 3,
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
            "dataset_shape": [256, 768],
            "directory": str(KITTI_DIR)
        }]
    }, {
        "batch_size": 2,
        "min_value": 0,
        "max_value": 255,
        "clip_value": True,
        "random_blur": True,
        "subsample_size": 3,
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
            "dataset_shape": [256, 768],
            "directory": str(MEGADEPTH_DIR)
        }]
    }, {
        "batch_size": 2,
        "min_value": 0,
        "max_value": 255,
        "clip_value": True,
        "random_blur": True,
        "subsample_size": 3,
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
            "dataset_shape": [256, 768],
            "directory": str(MEGADEPTH_DIR)
        }]
    }, {
        "batch_size": 2,
        "min_value": 0,
        "max_value": 255,
        "clip_value": True,
        "random_blur": True,
        "subsample_size": 3,
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
            "dataset_shape": [256, 768],
            "directory": str(MEGADEPTH_DIR)
        },{
            "dataset_shape": [256, 768],
            "directory": str(KITTI_DIR)
        }]
    }])
def test_dataset_builder_build(config):
    dataset_results = bfcnn.dataset.dataset_builder(config=config)
    assert bfcnn.dataset.DATASET_FN_STR in dataset_results
    assert bfcnn.dataset.AUGMENTATION_FN_STR in dataset_results

    for input_batch in dataset_results[bfcnn.dataset.DATASET_FN_STR]:
        assert input_batch.shape[0] <= config["batch_size"]
        assert input_batch.shape[1] == config["input_shape"][0]
        assert input_batch.shape[2] == config["input_shape"][1]

        if config["color_mode"] == "grayscale":
            assert input_batch.shape[3] == 1
        if config["color_mode"] == "rgb":
            assert input_batch.shape[3] == 3

        assert np.max(input_batch) <= config["max_value"]
        assert np.min(input_batch) >= config["min_value"]
        assert input_batch.dtype == np.float32


# ---------------------------------------------------------------------
