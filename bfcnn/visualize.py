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
        input_image,
        prediction,
        visualization_number: int):
    """
    Prepare images and add them to tensorboard
    """
    # --- save train images
    images_normalized = input_image / 255

    tf.summary.image(
        name="input",
        step=global_step,
        data=images_normalized,
        max_outputs=visualization_number)

    prediction_normalized = prediction / 255

    tf.summary.image(
        name="prediction",
        step=global_step,
        data=prediction_normalized,
        max_outputs=visualization_number)
