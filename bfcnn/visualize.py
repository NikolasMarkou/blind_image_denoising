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
    images_normalized = input_batch / 255

    tf.summary.image(
        name="input",
        step=global_step,
        data=images_normalized,
        max_outputs=visualization_number)

    # --- noisy
    noisy_normalized = noisy_batch / 255

    tf.summary.image(
        name="input",
        step=global_step,
        data=noisy_normalized,
        max_outputs=visualization_number)

    # --- prediction
    prediction_normalized = prediction_batch / 255

    tf.summary.image(
        name="prediction",
        step=global_step,
        data=prediction_normalized,
        max_outputs=visualization_number)

# ---------------------------------------------------------------------
