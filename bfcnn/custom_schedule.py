import numpy as np
from tensorflow.keras.callbacks import LearningRateScheduler

# ---------------------------------------------------------------------


def step_decay_schedule(
        initial_lr: float,
        decay_factor: float = 0.5,
        step_size: float = 1.0) -> LearningRateScheduler:
    """
    Wrapper function to create a LearningRateScheduler with step decay schedule

    :param initial_lr: initial learning rate
    :param decay_factor: decay factor
    :param step_size:
    :return:
    """
    def schedule(epoch):
        new_lr = initial_lr * (decay_factor ** np.floor(epoch/step_size))
        return new_lr

    return LearningRateScheduler(schedule)

# ---------------------------------------------------------------------
