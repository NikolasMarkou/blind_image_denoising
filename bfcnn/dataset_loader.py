import os
import glob
import numpy as np
import tensorflow as tf
from typing import Tuple
from keras.utils import image_utils

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .custom_logger import logger

ALLOWED_FORMATS = [".bmp", ".jpeg", ".jpg", ".png"]

# ---------------------------------------------------------------------


def image_dataset_from_directory(
        directory,
        color_mode="rgb",
        image_size=(256, 256),
        interpolation="bilinear",
        crop_to_aspect_ratio=False,
        random_crop: Tuple[int, int] = (32, 32),
        **kwargs,
):
    """Generates a `tf.data.Dataset` from image files in a directory"""
    if "smart_resize" in kwargs:
        crop_to_aspect_ratio = kwargs.pop("smart_resize")
    if kwargs:
        raise TypeError(f"Unknown keywords argument(s): {tuple(kwargs.keys())}")
    if color_mode == "rgb":
        num_channels = 3
    elif color_mode == "rgba":
        num_channels = 4
    elif color_mode == "grayscale":
        num_channels = 1
    else:
        raise ValueError(
            '`color_mode` must be one of {"rgb", "rgba", "grayscale"}. '
            f"Received: color_mode={color_mode}"
        )
    if random_crop is  None:
        raise ValueError("random_crop cannot be None")

    # --- set utilities
    random_crop = (random_crop[0], random_crop[1], num_channels)
    interpolation = image_utils.get_interpolation(interpolation)

    def generator_fn():
        allowed_formats = set(ALLOWED_FORMATS)
        for file_path in glob.iglob(pathname=os.path.join(directory, "**"), recursive=True):
            # check if directory
            if os.path.isdir(file_path):
                continue
            suffix = os.path.splitext(file_path)[-1]
            suffix = suffix.strip().lower()
            if suffix not in allowed_formats:
                continue
            img = \
                load_image(
                    path=file_path,
                    image_size=image_size,
                    num_channels=num_channels,
                    interpolation=interpolation,
                    crop_to_aspect_ratio=crop_to_aspect_ratio,
                    random_crop=random_crop)
            if img is None:
                continue
            yield img


    dataset = \
        tf.data.Dataset.from_generator(
            generator=generator_fn,
            output_signature=(
                tf.TensorSpec(
                    shape=(random_crop[0], random_crop[1], num_channels),
                    dtype=tf.uint8)
            ))

    return dataset


# ---------------------------------------------------------------------


def load_image(
        path,
        image_size,
        num_channels,
        interpolation,
        crop_to_aspect_ratio: bool = False,
        random_crop: Tuple[int, int, int] = None):
    """Load an image from a path and resize it."""

    if (path is None) or (not os.path.isfile(path)):
        return None

    try:
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=num_channels, expand_animations=False, dtype=tf.uint8)
        # img is [H, W, 3]
        img = \
            tf.image.resize(
                images=img,
                size=image_size,
                method=interpolation,
                preserve_aspect_ratio=True)
        img = tf.image.resize_with_crop_or_pad(
            img, target_height=image_size[0], target_width=image_size[1]
        )

        #img = tf.image.random_crop(img, size=random_crop)
        #img = tf.cast(img, dtype=tf.uint8)
        return img
    except Exception as e:
        logger.info(f"failed: {img.shape}:{path}:{e}")
        return None

# ---------------------------------------------------------------------
