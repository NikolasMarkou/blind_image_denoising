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

    # export
    module = \
        bfcnn.model_denoiser.module_builder(
            model_denoise=models.denoiser,
            model_normalize=models.normalizer,
            model_denormalize=models.denormalizer)

    assert isinstance(module, tf.Module)

# ---------------------------------------------------------------------
