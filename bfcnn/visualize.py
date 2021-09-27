r"""Tensorboard samples visualization"""

# ---------------------------------------------------------------------

__author__ = "Nikolas Markou"
__version__ = "0.1.0"
__license__ = "None"

# ---------------------------------------------------------------------

import tensorflow as tf

# ---------------------------------------------------------------------


def visualize(
        global_step,
        input_batch,
        noisy_batch,
        prediction_batch,
        visualization_number: int):
    """
    Prepare images and add them to tensorboard
    """
    # --- save train images
    tf.summary.image(
        name="input",
        step=global_step,
        data=input_batch / 255,
        max_outputs=visualization_number)

    # --- noisy
    tf.summary.image(
        name="noisy",
        step=global_step,
        data=noisy_batch / 255,
        max_outputs=visualization_number)

    # --- prediction
    tf.summary.image(
        name="prediction",
        step=global_step,
        data=prediction_batch / 255,
        max_outputs=visualization_number)

# ---------------------------------------------------------------------
