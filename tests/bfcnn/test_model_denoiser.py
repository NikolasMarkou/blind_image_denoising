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
from bfcnn.module_denoiser import DenoiserModule
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "config", bfcnn.configs)
def test_model_builder(config):
    config = config[1]
    models = bfcnn.model_builder(config=config["model"])

    # hydra
    assert isinstance(models.hydra, keras.Model)
    # backbone
    assert isinstance(models.backbone, keras.Model)
    # denoiser
    assert isinstance(models.denoiser, keras.Model)
    # superres
    assert isinstance(models.superres, keras.Model)
    # normalize
    assert isinstance(models.normalizer, keras.Model)
    # denormalize
    assert isinstance(models.denormalizer, keras.Model)

    # testing denoiser
    no_channels = models.backbone.input_shape[-1]
    for i in [32, 64, 128, 256]:
        x = tf.random.uniform(
            shape=[1, i, i, no_channels],
            minval=-0.5,
            maxval=+0.5,
            dtype=tf.float32)
        # denoiser_output,
        # superres_output
        d, ds, de, s, ss, se = models.hydra(x)

        assert d.shape == x.shape
        assert ds.shape == x.shape
        assert de.shape == x.shape
        assert s.shape[0] == x.shape[0]
        assert s.shape[1] == x.shape[1] * 2
        assert s.shape[2] == x.shape[2] * 2
        assert s.shape[3] == x.shape[3]

    # export
    denoiser_module = \
        DenoiserModule(
            model_backbone=models.backbone,
            model_denoiser=models.denoiser,
            model_normalizer=models.normalizer,
            model_denormalizer=models.denormalizer)

    assert isinstance(denoiser_module, tf.Module)

    # testing denoiser module
    no_channels = models.backbone.input_shape[-1]
    for i in [32, 64, 128, 256]:
        x = tf.random.uniform(
            shape=[1, i, i, no_channels],
            minval=0,
            maxval=255,
            dtype=tf.int32)
        x = tf.cast(x, dtype=tf.uint8)
        y = denoiser_module(x)

        assert y.shape == x.shape

# ---------------------------------------------------------------------
