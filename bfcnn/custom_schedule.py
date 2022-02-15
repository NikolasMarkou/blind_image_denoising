# ---------------------------------------------------------------------

__author__ = "Nikolas Markou"
__version__ = "1.0.0"
__license__ = "MIT"

# ---------------------------------------------------------------------

import numpy as np
from tensorflow.keras.callbacks import LearningRateScheduler

# ---------------------------------------------------------------------


def step_decay_schedule(initial_lr, 
                        decay_factor=0.5, 
                        step_size=1):
    """
    Wrapper function to create a LearningRateScheduler with step decay schedule

    :param initial_lr:
    :param decay_factor:
    :param step_size:
    :return:
    """
    def schedule(epoch):
        new_lr = initial_lr * (decay_factor ** np.floor(epoch/step_size))
        return new_lr

    return LearningRateScheduler(schedule)

# ---------------------------------------------------------------------
