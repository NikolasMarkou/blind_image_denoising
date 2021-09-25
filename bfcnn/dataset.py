# ---------------------------------------------------------------------

__author__ = ""
__version__ = "0.1.0"
__license__ = "None"

# ---------------------------------------------------------------------

import os
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras.backend as K
from typing import Dict, Callable, Iterator

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .custom_logger import logger

# ---------------------------------------------------------------------


def dataset_builder(
        config: Dict):
    # --- argument parsing
    batch_size = config["batch_size"]
    input_shape = config["input_shape"]
    directory = config.get("directory", None)
    # --- clip values
    min_value = config.get("min_value", 0)
    max_value = config.get("max_value", 255)
    clip_value = config.get("clip_value", False)
    # --- dataset augmentation
    random_blur = config.get("random_blur", False)
    # in radians
    random_rotate = config.get("random_rotate", 0.0)
    random_invert = config.get("random_invert", False)
    random_up_down = config.get("random_up_down", False)
    additive_noise = config.get("additive_noise", [0.1])
    random_left_right = config.get("random_left_right", False)
    multiplicative_noise = config.get("multiplicative_noise", [0.01])

    # --- define generator function from directory
    if directory is not None:
        dataset = \
            tf.keras.preprocessing.image_dataset_from_directory(
                seed=0,
                shuffle=True,
                label_mode=None,
                directory=directory,
                batch_size=batch_size,
                image_size=(input_shape[0], input_shape[1]))
    else:
        raise ValueError("don't know how to handle non directory datasets")

    # --- define augmentation function
    def augmentation(input_batch):
        # --- multiplicative noise
        noise_std = np.random.choice(multiplicative_noise)
        noisy_batch = \
            input_batch * \
            tf.random.normal(
                mean=1,
                stddev=noise_std,
                shape=tf.shape(input_batch))

        # --- blur to embed noise
        if random_blur:
            if np.random.choice([True, False]):
                kernel = np.random.choice([(3, 3), (5, 5), (7, 7)])
                noisy_batch = \
                    tfa.image.gaussian_filter2d(
                        image=noisy_batch,
                        filter_shape=kernel)

        # --- additive noise (independent)
        noise_std = np.random.choice(additive_noise)
        noisy_batch = \
            noisy_batch + \
            tf.random.normal(
                mean=0,
                stddev=noise_std,
                shape=tf.shape(noisy_batch))

        # --- flip left right
        if random_left_right:
            if np.random.choice([True, False]):
                input_batch = \
                    tf.image.flip_left_right(input_batch)
                noisy_batch = \
                    tf.image.flip_left_right(noisy_batch)

        # --- flip up down
        if random_up_down:
            if np.random.choice([True, False]):
                input_batch = \
                    tf.image.flip_up_down(input_batch)
                noisy_batch = \
                    tf.image.flip_up_down(noisy_batch)

        # --- randomly rotate input
        if random_rotate > 0.0:
            if np.random.choice([True, False]):
                tmp_batch_size = \
                    K.int_shape(input_batch)[0]
                angles = \
                    tf.random.uniform(
                        dtype=tf.float32,
                        minval=-random_rotate,
                        maxval=random_rotate,
                        shape=(tmp_batch_size,))
                input_batch = \
                    tfa.image.rotate(
                        images=input_batch,
                        angles=angles,
                        fill_value=0,
                        fill_mode="constant",
                        interpolation="bilinear")
                noisy_batch = \
                    tfa.image.rotate(
                        images=noisy_batch,
                        angles=angles,
                        fill_value=0,
                        fill_mode="constant",
                        interpolation="bilinear")

        # --- random invert colors
        if random_invert:
            if np.random.choice([True, False]):
                input_batch = max_value - (input_batch - min_value)
                noisy_batch = max_value - (noisy_batch - min_value)

        # --- clip values within boundaries
        if clip_value:
            input_batch = \
                tf.clip_by_value(
                    input_batch,
                    clip_value_min=min_value,
                    clip_value_max=max_value)

            noisy_batch = \
                tf.clip_by_value(
                    noisy_batch,
                    clip_value_min=min_value,
                    clip_value_max=max_value)

        return input_batch, noisy_batch

    # --- create the dataset
    return {
        "dataset": dataset.prefetch(2),
        "augmentation": augmentation
    }

# ---------------------------------------------------------------------
