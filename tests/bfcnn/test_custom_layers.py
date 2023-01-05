import pytest

import os
import sys
import numpy as np
from tensorflow import keras

from .constants import *

sys.path.append(os.getcwd() + "/../")

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

import bfcnn


# ---------------------------------------------------------------------

@pytest.mark.parametrize(
    "units", [8, 16, 32, 64])
def test_multiplier_layer(units):
    dense_params = dict(
        use_bias=False,
        activation="relu",
        units=units,
        kernel_regularizer="l1",
        kernel_initializer="glorot_normal")
    params = dict(
        multiplier=1.0,
        regularizer=None,
        trainable=True,
        activation="linear"
    )
    input_layer = \
        keras.Input(
            name="input_tensor",
            shape=(256,))
    x = input_layer
    x = keras.layers.Dense(**dense_params)(x)
    layer = bfcnn.Multiplier(**params)
    x = layer(x)
    output_layer = x
    model = keras.Model(
        name="model",
        trainable=True,
        inputs=input_layer,
        outputs=output_layer)
    result = model(np.random.normal(size=(10, 256)), training=True)
    assert result.shape == (10, units)
    result = model(np.random.normal(size=(10, 256)), training=False)
    assert result.shape == (10, units)

    assert len(layer.weights) == 2
    assert layer.weights[0].numpy().shape == (1, )

# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "units", [8, 16, 32, 64])
def test_channelwise_multiplier_layer_on_dense(units):
    dense_params = dict(
        use_bias=False,
        activation="relu",
        units=units,
        kernel_regularizer="l1",
        kernel_initializer="glorot_normal")
    params = dict(
        multiplier=1.0,
        regularizer=None,
        trainable=True,
        activation="linear"
    )
    input_layer = \
        keras.Input(
            name="input_tensor",
            shape=(256,))
    x = input_layer
    x = keras.layers.Dense(**dense_params)(x)
    layer = bfcnn.ChannelwiseMultiplier(**params)
    x = layer(x)
    output_layer = x
    model = keras.Model(
        name="model",
        trainable=True,
        inputs=input_layer,
        outputs=output_layer)
    result = model(np.random.normal(size=(10, 256)), training=True)
    assert result.shape == (10, units)
    result = model(np.random.normal(size=(10, 256)), training=False)
    assert result.shape == (10, units)

    assert len(layer.weights) == 2
    assert layer.weights[0].numpy().shape == (units,)

# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "filters", [8, 16, 32, 64])
def test_channelwise_multiplier_layer_on_conv2d(filters):
    conv_params = dict(
        kernel_size=(3, 3),
        padding="same",
        strides=(1, 1),
        use_bias=False,
        filters=filters,
        trainable=True,
        activation="relu")
    params = dict(
        multiplier=1.0,
        regularizer=None,
        trainable=True,
        activation="linear"
    )
    input_layer = \
        keras.Input(
            name="input_tensor",
            shape=(32, 32, 1))
    x = input_layer
    x = keras.layers.Conv2D(**conv_params)(x)
    layer = bfcnn.ChannelwiseMultiplier(**params)
    x = layer(x)
    output_layer = x
    model = keras.Model(
        name="model",
        trainable=True,
        inputs=input_layer,
        outputs=output_layer)
    result = model(np.random.normal(size=(10, 32, 32, 1)), training=True)
    assert result.shape == (10, 32, 32, filters)
    result = model(np.random.normal(size=(10, 32, 32, 1)), training=False)
    assert result.shape == (10, 32, 32, filters)

    assert len(layer.weights) == 2
    assert layer.weights[0].numpy().shape == (filters,)

# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "units", [8, 16, 32, 64])
def test_random_on_off_layer_dense(units):
    dense_params = dict(
        use_bias=False,
        activation="relu",
        units=units,
        kernel_regularizer="l1",
        kernel_initializer="glorot_normal")
    params = dict(
        rate=0.5,
    )
    input_layer = \
        keras.Input(
            name="input_tensor",
            shape=(256,))
    x = input_layer
    x = keras.layers.Dense(**dense_params)(x)
    layer = bfcnn.RandomOnOff(**params)
    x = layer(x)
    output_layer = x
    model = keras.Model(
        name="model",
        trainable=True,
        inputs=input_layer,
        outputs=output_layer)
    result = model(np.random.normal(size=(10, 256)), training=True)
    assert result.shape == (10, units)
    result = model(np.random.normal(size=(10, 256)), training=False)
    assert result.shape == (10, units)

    assert len(layer.weights) == 1
    assert layer.weights[0].numpy().shape == (1,)

# ---------------------------------------------------------------------


def test_gelu_layer():
    # TODO
    pass


# ---------------------------------------------------------------------


def test_differentiable_relu_layer():
    # TODO
    pass

# ---------------------------------------------------------------------
