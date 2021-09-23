# ---------------------------------------------------------------------

__author__ = ""
__version__ = "0.1.0"
__license__ = "None"

# ---------------------------------------------------------------------

import os
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras as keras
from typing import List, Tuple, Union, Dict, Callable, Iterator

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .utilities import *
from .custom_logger import logger

# ---------------------------------------------------------------------


def dataset_builder(
        config: Dict):
    # --- argument parsing
    batch_size = config["batch_size"]
    input_shape = config["input_shape"]
    # dataset augmentation
    noise_std = config.get("noise_std", 0.0)
    random_noise = noise_std > 0.0
    random_rotate = config.get("random_rotate", 0.0)
    random_up_down = config.get("random_up_down", False)
    random_left_right = config.get("random_left_right", False)

    # --- define generator function
    def generator_fn():
        input_batch = None
        noisy_batch = input_batch

        # --- data augmentation (jitter noise)
        if random_noise:
            if np.random.choice([True, False]):
                noisy_batch = \
                    noisy_batch + \
                    tf.random.normal(
                        shape=noisy_batch.shape,
                        mean=0,
                        stddev=noise_std)
            if np.random.choice([True, False]):
                noisy_batch = \
                    noisy_batch * \
                    tf.random.normal(
                        shape=noisy_batch.shape,
                        mean=1,
                        stddev=0.05)

        # --- flip left right (dataset augmentation)
        if random_left_right:
            if np.random.choice([True, False]):
                noisy_batch = \
                    tf.image.flip_left_right(noisy_batch)
                input_batch = \
                    tf.image.flip_left_right(input_batch)

        # --- flip up down (dataset augmentation)
        if random_up_down:
            if np.random.choice([True, False]):
                noisy_batch = \
                    tf.image.flip_up_down(noisy_batch)
                input_batch = \
                    tf.image.flip_up_down(input_batch)

        # --- randomly rotate input (dataset augmentation)
        if random_rotate > 0.0:
            if np.random.choice([True, False]):
                tmp_batch_size = \
                    K.int_shape(noisy_batch)[0]
                angles = \
                    tf.random.uniform(
                        dtype=tf.float32,
                        minval=-random_rotate,
                        maxval=random_rotate,
                        shape=(tmp_batch_size,))
                noisy_batch = \
                    tfa.image.rotate(
                        images=noisy_batch,
                        interpolation="bilinear",
                        angles=angles,
                        fill_value=0,
                        fill_mode="constant")
                input_batch = \
                    tfa.image.rotate(
                        images=input_batch,
                        interpolation="bilinear",
                        angles=angles,
                        fill_value=0,
                        fill_mode="constant")
        pass

    # --- create the dataset
    return \
        tf.data.Dataset.from_generator(
            generator=generator_fn,
            output_signature=(
                tf.TensorSpec(shape=(input_shape[0], input_shape[1], 3),
                              dtype=tf.float32),
                tf.TensorSpec(shape=(no_landmarks, 2), dtype=tf.float32),
                tf.TensorSpec(shape=(3,), dtype=tf.float32))) \
            .batch(batch_size) \
            .prefetch(2)

# ---------------------------------------------------------------------
