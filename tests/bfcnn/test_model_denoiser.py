import os
import sys
import pytest
import tensorflow as tf
from tensorflow import keras

sys.path.append(os.getcwd() + "/../")

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

import bfcnn

# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "config", bfcnn.CONFIGS)
def test_model_builder(config):
    config = config[1]
    models = bfcnn.model_builder(config=config["model_denoise"])

    # denoise
    assert isinstance(models.denoiser, keras.Model)
    # normalize
    assert isinstance(models.normalizer, keras.Model)
    # denormalize
    assert isinstance(models.denormalizer, keras.Model)

    # testing denoiser
    no_channels = models.denoiser.input_shape[3]
    for i in [32, 64, 128, 256]:
        x = tf.random.uniform(
            shape=[1, i, i, no_channels],
            minval=-0.5,
            maxval=+0.5,
            dtype=tf.float32)
        y = models.denoiser(x)

        assert y.shape == x.shape

    # export
    module = \
        bfcnn.model_denoiser.module_builder(
            model_denoise=models.denoiser,
            model_normalize=models.normalizer,
            model_denormalize=models.denormalizer)

    assert isinstance(module, tf.Module)

    # testing denoiser module
    no_channels = models.denoiser.input_shape[3]
    for i in [32, 64, 128, 256]:
        x = tf.random.uniform(
            shape=[1, i, i, no_channels],
            minval=0,
            maxval=255,
            dtype=tf.int32)
        x = tf.cast(
            x, dtype=tf.uint8)
        y = models.denoiser(x)

        assert y.shape == x.shape

# ---------------------------------------------------------------------
