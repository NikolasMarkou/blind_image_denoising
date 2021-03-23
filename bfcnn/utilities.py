import numpy as np
from keras import backend as K

# ==============================================================================


def normalize(v, min_value=0.0, max_value=255.0):
    """
    Normalize input from [min_value, max_value] to [0, 1]

    :param v:
    :param min_value:
    :param max_value:
    :return:
    """
    v = np.clip(v, a_min=min_value, a_max=max_value)
    return (v - min_value) / (max_value - min_value)

# ==============================================================================


def denormalize(v, min_value=0.0, max_value=255.0):
    """
    Denormalize input from [0, 1] to [min_value, max_value]

    :param v:
    :param min_value:
    :param max_value:
    :return:
    """
    v = np.clip(v, a_min=0.0, a_max=1.0)
    return v * (max_value - min_value) + min_value

# ==============================================================================


def layer_normalize(args):
    """
    Convert input from [v0, v1] to [-1, +1] range
    """
    y, v0, v1 = args
    y_clip = K.clip(y, min_value=v0, max_value=v1)
    return 2.0 * (y_clip - v0) / (v1 - v0) - 1.0

# ==============================================================================


def layer_denormalize(args):
    """
    Convert input [-1, +1] to [v0, v1] range
    """
    y, v0, v1 = args
    y0 = (y + 1.0) * (v1 - v0) / 2.0 + v0
    return K.clip(y0, min_value=v0, max_value=v1)

# ==============================================================================
