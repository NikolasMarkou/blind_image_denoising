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
    "noise_std", [10.0, 15.0, 20.0, 25.0, 30.0])
@pytest.mark.parametrize(
    "model_name", bfcnn.models.keys())
def test_pretrained_models(noise_std, model_name):
    model_structure = bfcnn.models[model_name]
    module_denoiser = model_structure[DENOISER_STR]()
    for img_path in KITTI_IMAGES:
        # load image
        img_original = \
            bfcnn.load_image(
                path=img_path,
                num_channels=3,
                expand_dims=True,
                normalize=False,
                dtype=tf.uint8)
        # corrupt it
        img_noisy = \
            tf.cast(img_original, dtype=tf.float32) + \
            tf.random.truncated_normal(
                seed=0,
                mean=0.0,
                stddev=noise_std,
                dtype=tf.float32,
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
        img_denoised = module_denoiser(img_noisy)
        # convert
        img_noisy = tf.cast(img_noisy, dtype=tf.float32)
        img_denoised = tf.cast(img_denoised, dtype=tf.float32)
        img_original = tf.cast(img_original, dtype=tf.float32)
        # assertions
        assert img_denoised.shape == img_noisy.shape
        assert img_denoised.shape == img_original.shape

        # psnr test
        psnr_original_noisy = tf.reduce_mean(tf.image.psnr(img_original, img_noisy, max_val=255.0))
        psnr_original_denoised = tf.reduce_mean(tf.image.psnr(img_original, img_denoised, max_val=255.0))
        assert psnr_original_noisy < psnr_original_denoised

        # ssim test
        ssim_original_noisy = tf.reduce_mean(tf.image.ssim(img_original, img_noisy, max_val=255.0))
        ssim_original_denoised = tf.reduce_mean(tf.image.ssim(img_original, img_denoised, max_val=255.0))
        assert ssim_original_noisy < ssim_original_denoised

        # mae test
        mae_original_noisy = tf.reduce_mean(tf.abs(img_original - img_noisy))
        mae_original_denoised = tf.reduce_mean(tf.abs(img_original - img_denoised))
        assert mae_original_denoised < mae_original_noisy

# ---------------------------------------------------------------------
