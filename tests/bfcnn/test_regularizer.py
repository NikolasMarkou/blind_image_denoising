import keras
import pytest

import os
import sys
import numpy as np
import tensorflow as tf
from .constants import *
sys.path.append(os.getcwd() + "/../")

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

import bfcnn
from bfcnn.regularizers import SoftOrthonormalConstraintRegularizer

# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape", [
        (5, 5, 3, 2),
        (1, 1, 1, 4),
        (2, 2, 2, 8),
        (4, 4, 4, 16),
        (8, 8, 8, 32)
    ])
def test_4d_reshape_to_2d(shape):
    x_random = \
        tf.random.uniform(
            dtype=tf.float32,
            minval=-1,
            maxval=+1,
            shape=shape)
    x_reshaped = bfcnn.regularizers.reshape_to_2d(x_random)
    assert x_reshaped.shape[0] == shape[3]
    assert x_reshaped.shape[1] == (shape[0] * shape[1] * shape[2])

# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape", [
        (3, 2),
        (1, 4),
        (2, 8),
        (4, 16),
        (8, 32)
    ])
def test_4d_reshape_to_2d(shape):
    x_random = \
        tf.random.uniform(
            dtype=tf.float32,
            minval=-1,
            maxval=+1,
            shape=shape)
    x_reshaped = bfcnn.regularizers.reshape_to_2d(x_random)
    assert x_reshaped.shape[0] == shape[1]
    assert x_reshaped.shape[1] == shape[0]

# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape", [
        (5, 5, 3, 2),
        (1, 1, 1, 4),
        (2, 2, 2, 8),
        (4, 4, 4, 16),
        (8, 8, 8, 32)
    ])
def test_4d_wt_x_w(shape):
    x_random = \
        tf.random.uniform(
            dtype=tf.float32,
            minval=-1,
            maxval=+1,
            shape=shape)
    wt_w = bfcnn.regularizers.wt_x_w(x_random)
    assert wt_w.shape[0] == shape[3]
    assert wt_w.shape[1] == shape[3]

# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape", [
        (3, 2),
        (1, 4),
        (2, 8),
        (4, 16),
        (8, 32)
    ])
def test_2d_wt_x_w(shape):
    x_random = \
        tf.random.uniform(
            dtype=tf.float32,
            minval=-1,
            maxval=+1,
            shape=shape)
    wt_w = bfcnn.regularizers.wt_x_w(x_random)
    assert wt_w.shape[0] == shape[1]
    assert wt_w.shape[1] == shape[1]

# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape", [
        (5, 5, 3, 2),
        (1, 1, 1, 4),
        (2, 2, 2, 8),
        (4, 4, 4, 16),
        (8, 8, 8, 32)
    ])
def test_create_4d_soft_orthogonal_constraint(shape):
    x_random = \
        tf.random.uniform(
            dtype=tf.float32,
            minval=-1,
            maxval=+1,
            shape=shape)
    regularizer = \
        bfcnn.regularizers.SoftOrthonormalConstraintRegularizer(1.0)
    result = regularizer(x_random)
    assert result >= 0


# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape", [
        (3, 2),
        (1, 4),
        (2, 8),
        (4, 16),
        (8, 32)
    ])
def test_create_2d_soft_orthogonal_constraint(shape):
    x_random = \
        tf.random.uniform(
            dtype=tf.float32,
            minval=-1,
            maxval=+1,
            shape=shape)
    regularizer = \
        bfcnn.regularizers.SoftOrthonormalConstraintRegularizer(1.0)
    result = regularizer(x_random)
    assert result >= 0

# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "config", [
        ["l1"],
        ["l1l2"],
        ["l1", "l2"],
        ["soft_orthogonal"],
        ["soft_orthogonal", "l1"]
    ])
def test_builder(config):
    prune_fns = bfcnn.regularizers.builder(config=config)
    assert prune_fns is not None

# ---------------------------------------------------------------------

