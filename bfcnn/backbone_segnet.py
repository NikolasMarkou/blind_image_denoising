import copy
import tensorflow as tf
from typing import List

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .constants import *


# ---------------------------------------------------------------------

def builder(
        name="segnet",
        **kwargs) -> keras.Model:
    raise NotImplementedError("segnet builder not implemented yet")

# ---------------------------------------------------------------------
