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
from bfcnn.regularizer import SoftOrthogonalConstraintRegularizer

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
        bfcnn.regularizer.SoftOrthogonalConstraintRegularizer(1.0)
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
        bfcnn.regularizer.SoftOrthogonalConstraintRegularizer(1.0)
    result = regularizer(x_random)
    assert result >= 0

# ---------------------------------------------------------------------

