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

def test_dataset_builder_kitti_build():
    config = {
        "batch_size": 2,
        "min_value": 0,
        "max_value": 255,
        "clip_value": True,
        "random_blur": True,
        "subsample_size": -1,
        "round_values": True,
        "random_invert": False,
        "random_rotate": 0.314,
        "random_up_down": True,
        "color_mode": "grayscale",
        "random_left_right": True,
        "dataset_shape": [256, 768],
        "input_shape": [128, 128, 1],
        "multiplicative_noise": [],
        "additional_noise": [1, 5, 10, 20, 40],
        "directory": str(KITTI_DIR)
    }
    dataset_results = bfcnn.dataset.dataset_builder(config=config)
    assert "dataset" in dataset_results
    assert "augmentation" in dataset_results

# ---------------------------------------------------------------------

def test_dataset_builder_megadepth_build():
    config = {
        "batch_size": 2,
        "min_value": 0,
        "max_value": 255,
        "clip_value": True,
        "random_blur": True,
        "subsample_size": -1,
        "round_values": True,
        "random_invert": False,
        "random_rotate": 0.314,
        "random_up_down": True,
        "color_mode": "grayscale",
        "random_left_right": True,
        "dataset_shape": [256, 768],
        "input_shape": [128, 128, 1],
        "multiplicative_noise": [],
        "additional_noise": [1, 5, 10, 20, 40],
        "directory": str(MEGADEPTH_DIR)
    }
    dataset_results = bfcnn.dataset.dataset_builder(config=config)
    assert "dataset" in dataset_results
    assert "augmentation" in dataset_results

# ---------------------------------------------------------------------
