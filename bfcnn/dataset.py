# ---------------------------------------------------------------------

__author__ = "Nikolas Markou"
__version__ = "0.1.0"
__license__ = "None"

# ---------------------------------------------------------------------

import os

import keras.layers
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
    # ---
    logger.info("creating dataset_builder with configuration [{0}]".format(config))
    # --- argument parsing
    batch_size = config["batch_size"]
    # crop image from dataset
    input_shape = config["input_shape"]
    directory = config.get("directory", None)
    color_mode = config.get("color_mode", "rgb")
    # resolution of the files loaded (reshape)
    dataset_shape = config.get("dataset_shape", [256, 256])
    # --- clip values
    min_value = config.get("min_value", 0)
    max_value = config.get("max_value", 255)
    clip_value = config.get("clip_value", True)
    # --- dataset augmentation
    random_blur = config.get("random_blur", False)
    subsample_size = config.get("subsample_size", -1)
    # in radians
    random_rotate = config.get("random_rotate", 0.0)
    # if true randomly invert
    random_invert = config.get("random_invert", False)
    # if true randomly invert upside down image
    random_up_down = config.get("random_up_down", False)
    # if true randomly invert left right image
    random_left_right = config.get("random_left_right", False)
    additional_noise = config.get("additional_noise", [])
    multiplicative_noise = config.get("multiplicative_noise", [])

    # build noise options
    noise_choices = []
    if len(additional_noise) > 0:
        noise_choices.append("additional")
    if len(multiplicative_noise) > 0:
        noise_choices.append("multiplicative")
    if subsample_size > 0:
        noise_choices.append("subsample_size")

    # --- define generator function from directory
    if directory is not None:
        dataset = \
            tf.keras.preprocessing.image_dataset_from_directory(
                seed=0,
                shuffle=True,
                label_mode=None,
                directory=directory,
                batch_size=batch_size,
                color_mode=color_mode,
                image_size=dataset_shape)
    else:
        raise ValueError("don't know how to handle non directory datasets")

    # --- define augmentation function
    def augmentation(input_batch):
        input_batch = \
            tf.image.random_crop(
                value=input_batch,
                size=(
                    tf.shape(input_batch)[0],
                    input_shape[0],
                    input_shape[1],
                    tf.shape(input_batch)[3])
            )

        # --- flip left right
        if random_left_right:
            if np.random.choice([True, False]):
                input_batch = \
                    tf.image.flip_left_right(input_batch)

        # --- flip up down
        if random_up_down:
            if np.random.choice([True, False]):
                input_batch = \
                    tf.image.flip_up_down(input_batch)

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
                        fill_mode="reflect",
                        interpolation="bilinear")

        # --- random invert colors
        if random_invert:
            if np.random.choice([True, False]):
                input_batch = max_value - (input_batch - min_value)

        # --- random select noise type
        noisy_batch = tf.identity(input_batch)
        noise_type = np.random.choice(noise_choices)
        # TODO assign a difficulty
        difficulty = 0

        if noise_type == "additional":
            # additional noise
            noise_std = np.random.choice(additional_noise)
            noisy_batch = \
                noisy_batch + \
                tf.random.truncated_normal(
                    mean=0,
                    stddev=noise_std,
                    shape=tf.shape(input_batch))
        elif noise_type == "multiplicative":
            # multiplicative noise
            noise_std = np.random.choice(multiplicative_noise)
            noisy_batch = \
                noisy_batch * \
                tf.random.truncated_normal(
                    mean=1,
                    stddev=noise_std,
                    shape=tf.shape(input_batch))

            # blur to embed noise
            if random_blur:
                if np.random.choice([True, False]):
                    noisy_batch = \
                        tfa.image.gaussian_filter2d(
                            image=noisy_batch,
                            sigma=1,
                            filter_shape=(3, 3))
        elif noise_type == "subsample":
            # subsample
            stride = (subsample_size, subsample_size)
            noisy_batch = \
                keras.layers.MaxPool2D(
                    pool_size=(1, 1),
                    strides=stride)(noisy_batch)
            noisy_batch = \
                keras.layers.UpSampling2D(
                    size=stride)(noisy_batch)
        else:
            logger.info(
                "don't know how to handle noise_type [{0}]".format(
                    noise_type))

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

        return input_batch, noisy_batch, difficulty

    # --- create the dataset
    return {
        "dataset": dataset.prefetch(2),
        "augmentation": augmentation
    }

# ---------------------------------------------------------------------
