r"""Tensorboard samples visualization"""

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
        denoiser_batch,
        denoiser_uq_batch,
        superres_batch,
        inpaint_batch,
        test_denoiser_batch=None,
        test_superres_batch=None,
        visualization_number: int = 3):
    """
    Prepare images and add them to tensorboard
    """
    # --- save train images
    if input_batch is not None:
        tf.summary.image(
            name="input",
            step=global_step,
            data=input_batch / 255,
            max_outputs=visualization_number)

    # --- noisy
    if noisy_batch is not None:
        tf.summary.image(
            name="noisy",
            step=global_step,
            data=noisy_batch / 255,
            max_outputs=visualization_number)

    # --- output denoiser
    if denoiser_batch is not None:
        tf.summary.image(
            name="output/denoiser",
            step=global_step,
            data=denoiser_batch / 255,
            max_outputs=visualization_number)

    # --- output denoiser
    if denoiser_uq_batch is not None:
        denoiser_uq_expected, denoiser_uq_variance = \
            tf.unstack(denoiser_uq_batch, axis=3)
        tf.summary.image(
            name="uncertainty/expected",
            step=global_step,
            data=denoiser_uq_expected,
            max_outputs=visualization_number)
        tf.summary.image(
            name="uncertainty/variance",
            step=global_step,
            data=denoiser_uq_variance,
            max_outputs=visualization_number)

    # --- output superres
    if superres_batch is not None:
        tf.summary.image(
            name="output/superres",
            step=global_step,
            data=superres_batch / 255,
            max_outputs=visualization_number)

    # --- output inpaint
    if inpaint_batch is not None:
        tf.summary.image(
            name="output/inpaint",
            step=global_step,
            data=inpaint_batch / 255,
            max_outputs=visualization_number)

    # --- test denoiser
    if test_denoiser_batch is not None:
        tf.summary.image(
            name="test/denoiser",
            step=global_step,
            data=test_denoiser_batch / 255,
            max_outputs=visualization_number)

    # --- test superres
    if test_superres_batch is not None:
        tf.summary.image(
            name="test/superres",
            step=global_step,
            data=test_superres_batch / 255,
            max_outputs=visualization_number)

# ---------------------------------------------------------------------
