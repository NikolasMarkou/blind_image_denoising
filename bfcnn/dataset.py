# ---------------------------------------------------------------------

__author__ = "Nikolas Markou"
__version__ = "1.0.0"
__license__ = "MIT"

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
    # --- clip values to min max
    min_value = config.get("min_value", 0)
    max_value = config.get("max_value", 255)
    clip_value = tf.constant(config.get("clip_value", True))
    # --- if true round values
    round_values = tf.constant(config.get("round_values", True))
    # --- dataset augmentation
    random_blur = tf.constant(config.get("random_blur", False))
    subsample_size = config.get("subsample_size", -1)
    # in radians
    random_rotate = tf.constant(config.get("random_rotate", 0.0))
    # if true randomly invert
    random_invert = tf.constant(config.get("random_invert", False))
    # if true randomly invert upside down image
    random_up_down = tf.constant(config.get("random_up_down", False))
    # if true randomly invert left right image
    random_left_right = tf.constant(config.get("random_left_right", False))
    additional_noise = config.get("additional_noise", [])
    multiplicative_noise = config.get("multiplicative_noise", [])
    interpolation = config.get("interpolation", "area")
    # whether to crop or not
    random_crop = dataset_shape[0:2] != input_shape[0:2]

    # build noise options
    noise_choices = []
    if len(additional_noise) > 0:
        noise_choices.append(0)
    if len(multiplicative_noise) > 0:
        noise_choices.append(1)
    if subsample_size > 0:
        noise_choices.append(2)
    noise_choices = tf.constant(noise_choices)

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
                image_size=dataset_shape,
                interpolation=interpolation)
    else:
        raise ValueError("don't know how to handle non directory datasets")

    tf.random.set_seed(0)
    np.random.seed(0)

    def random_choice(x, size, axis=0):
        dim_x = tf.cast(tf.shape(x)[axis], tf.int64)
        indices = tf.range(0, dim_x, dtype=tf.int64)
        sample_index = tf.random.shuffle(indices)[:size]
        sample = tf.gather(x, sample_index, axis=axis)
        return sample, sample_index

    # --- define augmentation function
    def augmentation(input_batch):
        input_shape_inference = tf.shape(input_batch)

        # --- crop randomly
        if random_crop:
            input_batch = \
                tf.image.random_crop(
                    value=input_batch,
                    seed=0,
                    size=(
                        input_shape_inference[0],
                        input_shape[0],
                        input_shape[1],
                        input_shape_inference[3])
                )
            input_shape_inference = tf.shape(input_batch)

        # --- flip left right
        if random_left_right:
            if tf.random.uniform(()) > 0.5:
                input_batch = \
                    tf.image.flip_left_right(input_batch)

        # --- flip up down
        if random_up_down:
            if tf.random.uniform(()) > 0.5:
                input_batch = \
                    tf.image.flip_up_down(input_batch)

        # --- randomly rotate input
        if random_rotate > 0.0:
            if tf.random.uniform(()) > 0.5:
                angles = \
                    tf.random.uniform(
                        seed=0,
                        dtype=tf.float32,
                        minval=-random_rotate,
                        maxval=random_rotate,
                        shape=(input_shape_inference[0],))
                input_batch = \
                    tfa.image.rotate(
                        angles=angles,
                        images=input_batch,
                        fill_mode="reflect",
                        interpolation="bilinear")

        # --- random invert colors
        if random_invert:
            if tf.random.uniform(()) > 0.5:
                input_batch = max_value - (input_batch - min_value)

        # --- random select noise type
        noisy_batch = tf.identity(input_batch)
        noise_type, _ = random_choice(noise_choices, size=1)

        if noise_type[0] == 0:
            # additional noise
            noise_std = np.random.choice(additional_noise)
            if tf.random.uniform(()) > 0.5:
                # channel independent noise
                noisy_batch = \
                    noisy_batch + \
                    tf.random.truncated_normal(
                        seed=0,
                        mean=0,
                        stddev=noise_std,
                        shape=input_shape_inference)
            else:
                # channel dependent noise
                tmp_noisy_batch = \
                    tf.random.truncated_normal(
                        seed=0,
                        mean=0,
                        stddev=noise_std,
                        shape=(input_shape_inference[0], input_shape[0], input_shape[1], 1))
                tmp_noisy_batch = \
                    tf.repeat(
                        tmp_noisy_batch,
                        axis=3,
                        repeats=[input_shape_inference[3]])
                noisy_batch = tmp_noisy_batch + noisy_batch
        elif noise_type[0] == 1:
            # multiplicative noise
            noise_std = np.random.choice(multiplicative_noise)
            if tf.random.uniform(()) > 0.5:
                # channel independent noise
                noisy_batch = \
                    noisy_batch * \
                    tf.random.truncated_normal(
                        seed=0,
                        mean=1,
                        stddev=noise_std,
                        shape=input_shape_inference)
            else:
                # channel dependent noise
                tmp_noisy_batch = \
                    tf.random.truncated_normal(
                        seed=0,
                        mean=1,
                        stddev=noise_std,
                        shape=(input_shape_inference[0], input_shape[0], input_shape[1], 1))
                tmp_noisy_batch = \
                    tf.repeat(
                        tmp_noisy_batch,
                        axis=3,
                        repeats=[input_shape_inference[3]])
                noisy_batch = tmp_noisy_batch * noisy_batch

            # blur to embed noise
            if random_blur:
                if tf.random.uniform(()) > 0.5:
                    noisy_batch = \
                        tfa.image.gaussian_filter2d(
                            image=noisy_batch,
                            sigma=1,
                            filter_shape=(3, 3))
        elif noise_type[0] == 2:
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

        # --- round values to nearest integer
        if round_values:
            input_batch = tf.round(input_batch)
            noisy_batch = tf.round(noisy_batch)

        return input_batch, noisy_batch

    # --- create the dataset
    return {
        "dataset": dataset.prefetch(2),
        "augmentation": augmentation
    }

# ---------------------------------------------------------------------
