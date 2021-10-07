r"""Tensorboard samples visualization"""

# ---------------------------------------------------------------------

__author__ = "Nikolas Markou"
__version__ = "0.1.0"
__license__ = "None"

# ---------------------------------------------------------------------

import numpy as np
import tensorflow as tf

# ---------------------------------------------------------------------


def collage(
        images_batch):
    """
    Create a collage of image from a batch

    :param images_batch:
    :return:
    """
    shape = images_batch.shape
    no_images = shape[0]
    images = []
    result = None
    width = np.ceil(np.sqrt(no_images))

    for i in range(no_images):
        images.append(images_batch[i, :, :, :])

        if len(images) % width == 0:
            if result is None:
                result = np.hstack(images)
            else:
                tmp = np.hstack(images)
                result = np.vstack([result, tmp])
            images.clear()
    return result

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
