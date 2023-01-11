import os
import sys
import pytest
import numpy as np
import tensorflow as tf

# ---------------------------------------------------------------------

from .constants import *

sys.path.append(os.getcwd() + "/../")

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

import bfcnn
from bfcnn.constants import *

# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "model_name", bfcnn.models.keys())
def test_pretrained_models(model_name):
    model_structure = bfcnn.models[model_name]
    model = model_structure[DENOISER_STR]()
    for img_path in KITTI_IMAGES:
        # load image
        img_original = \
            bfcnn.load_image(
                path=img_path,
                num_channels=3,
                expand_dims=True,
                normalize=False,
                dtype=tf.uint8)
        img_original = tf.cast(img_original, dtype=tf.float32)
        # corrupt it
        img_noisy = \
            img_original + \
            tf.random.truncated_normal(
                seed=0,
                mean=0,
                stddev=10,
                shape=img_original.shape)
        img_noisy = \
            tf.cast(
                tf.round(
                    tf.clip_by_value(
                        img_noisy,
                        clip_value_min=0,
                        clip_value_max=255)),
                dtype=tf.uint8)
        # denoise it
        img_denoised = model(img_noisy)
        # compare
        img_noisy = tf.cast(img_noisy, dtype=tf.float32)
        img_original = tf.cast(img_original, dtype=tf.float32)
        img_denoised = tf.cast(img_denoised, dtype=tf.float32)
        # mae
        mae_noisy_original = np.mean(np.abs(img_noisy - img_original), axis=None)
        mae_denoised_original = np.mean(np.abs(img_denoised - img_original), axis=None)
        assert img_denoised.shape == img_noisy.shape
        assert img_denoised.shape == img_original.shape
        assert mae_noisy_original < 10
        assert mae_denoised_original < mae_noisy_original

# ---------------------------------------------------------------------

