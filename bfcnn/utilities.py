import numpy as np
from enum import Enum
from typing import List
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator

# ==============================================================================


def collage(images_batch):
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
    height = no_images / width

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

# ==============================================================================


class Loss(Enum):
    MeanAbsoluteError = 1
    MeanSquareError = 2


def mae_loss(y_true, y_pred):
    tmp_pixels = K.abs(y_true[0] - y_pred[0])
    return K.mean(tmp_pixels)


def mse_loss(y_true, y_pred):
    tmp_pixels = K.pow(y_true[0] - y_pred[0], 2.0)
    return K.mean(tmp_pixels)


def build_loss_fn(loss: List[Loss]):
    # --- argument checking
    if len(loss) <= 0:
        raise ValueError("loss list cannot be empty")

    # --- define loss functions
    loss_fn = []

    if Loss.MeanSquareError in loss:
        loss_fn.append(mse_loss)

    if Loss.MeanAbsoluteError in loss:
        loss_fn.append(mae_loss)

    # --- total loss
    def total_loss(y_true, y_pred):
        result = 0.0
        for fn in loss_fn:
            result = result + fn(y_true, y_pred)
        return result

    return total_loss, loss_fn


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


def noisy_image_data_generator(
        dataset,
        batch_size: int = 32,
        min_value: float = 0.0,
        max_value: float = 255.0,
        min_noise_std: float = 0.01,
        max_noise_std: float = 10.0,
        random_invert: bool = False,
        random_noise: bool = True,
        random_brightness: bool = False,
        zoom_range: float = 0.25,
        rotation_range: int = 90,
        width_shift_range: float = 0.25,
        height_shift_range: float = 0.25,
        vertical_flip: bool = True,
        horizontal_flip: bool = True):
    """
    Create a dataset generator flow that adds noise to a dateset

    :param dataset:
    :param min_value: Minimum allowed value
    :param max_value: Maximum allowed value
    :param batch_size: Batch size
    :param random_invert: Randomly (50%) invert the image
    :param random_noise: Randomly (50%) add noise
    :param random_brightness: Randomly add offset or multiplier
    :param zoom_range: Randomly zoom in (percentage)
    :param rotation_range: Add random rotation range (in degrees)
    :param min_noise_std: Min standard deviation of noise
    :param max_noise_std: Max standard deviation of noise
    :param horizontal_flip: Randomly horizontally flip image
    :param vertical_flip: Randomly vertically flip image
    :param height_shift_range: Add random vertical shift (percentage of image)
    :param width_shift_range: Add random horizontal shift (percentage of image)

    :return:
    """
    # --- argument checking
    if dataset is None:
        raise ValueError("dataset cannot be empty")
    if random_noise:
        if min_noise_std > max_noise_std:
            raise ValueError("min_noise_std must be < max_noise_std")

    # --- variables setup
    max_min_diff = (max_value - min_value)

    # --- create data generator
    if isinstance(dataset, np.ndarray):
        data_generator = \
            ImageDataGenerator(
                zoom_range=zoom_range,
                rotation_range=rotation_range,
                width_shift_range=width_shift_range,
                height_shift_range=height_shift_range,
                vertical_flip=vertical_flip,
                horizontal_flip=horizontal_flip,
                zca_whitening=False,
                featurewise_center=False,
                featurewise_std_normalization=False)
    else:
        raise NotImplementedError()

    # --- iterate over random batches
    for x_batch in \
            data_generator.flow(
                x=dataset,
                shuffle=True,
                batch_size=batch_size):
        # randomly invert batch
        if random_invert:
            if np.random.choice([False, True]):
                x_batch = (max_value - x_batch) + min_value

        # adjust the std of the noise
        if random_noise:
            if np.random.choice([False, True]):
                # pick std between min and max std
                std = \
                    np.random.uniform(
                        low=min_noise_std,
                        high=max_noise_std)

                # add noise to create the noisy input
                x_batch_noisy = \
                    x_batch + \
                    np.random.normal(0.0, std, x_batch.shape)
            else:
                x_batch_noisy = np.copy(x_batch)
        else:
            x_batch_noisy = np.copy(x_batch)

        # adjust brightness
        if random_brightness:
            if np.random.choice([False, True]):
                brightness_offset = \
                    np.random.uniform(
                        low=0.1 * max_min_diff,
                        high=0.5 * max_min_diff,
                        size=1)
                x_batch = x_batch + brightness_offset
                x_batch_noisy = x_batch_noisy + brightness_offset
            elif np.random.choice([False, True]):
                brightness_multiplier = \
                    np.random.uniform(
                        low=0.25,
                        high=0.75,
                        size=1)
                x_batch = x_batch * brightness_multiplier
                x_batch_noisy = x_batch_noisy * brightness_multiplier

        # clip all to be between min and max value
        x_batch = np.clip(x_batch, a_min=min_value, a_max=max_value)
        x_batch_noisy = np.clip(x_batch_noisy, a_min=min_value, a_max=max_value)

        yield x_batch_noisy, x_batch

# ==============================================================================
