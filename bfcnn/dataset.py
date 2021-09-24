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
    # dataset augmentation
    random_rotate = config.get("random_rotate", 0.0)
    random_up_down = config.get("random_up_down", False)
    random_left_right = config.get("random_left_right", False)

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
    def augment_df(input_batch):
        # --- additive noise
        noisy_batch = \
            input_batch + \
            tf.random.normal(
                shape=input_batch.shape,
                mean=0,
                stddev=0.1)

        # --- multiplicative noise
        noisy_batch = \
            noisy_batch * \
            tf.random.normal(
                shape=noisy_batch.shape,
                mean=1,
                stddev=0.05)

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

        return input_batch, noisy_batch

    # --- create the dataset
    return \
        dataset\
            .apply(transformation_func=augment_df)\
            .prefetch(2)

# ---------------------------------------------------------------------
