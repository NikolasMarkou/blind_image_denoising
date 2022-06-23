import pytest

import os
import sys
import numpy as np

# ---------------------------------------------------------------------

from .constants import *

sys.path.append(os.getcwd() + "/../")

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

import bfcnn

# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "config", bfcnn.configs)
def test_optimizer_builder(config):
    optimizer, learning_rate = \
        bfcnn.optimizer_builder(config=config["train"]["optimizer"])
    assert optimizer is not None
    assert learning_rate is not None

# ---------------------------------------------------------------------
