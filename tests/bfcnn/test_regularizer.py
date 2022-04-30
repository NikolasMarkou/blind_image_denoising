import keras
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
from bfcnn.regularizer import SoftOrthogonalConstraintRegularizer

# ---------------------------------------------------------------------


def test_create_so():
    x_random = np.random.normal(loc=0, scale=1, size=(5, 5, 3, 16))
    regularizer = \
        bfcnn.regularizer.SoftOrthogonalConstraintRegularizer(1.0)
    result = regularizer(x_random)
    assert result >= 0

# ---------------------------------------------------------------------

