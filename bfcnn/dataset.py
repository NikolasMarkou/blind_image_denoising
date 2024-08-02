import copy
import tensorflow as tf
from collections import namedtuple
from typing import Dict, Callable, Iterator, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .file_operations import *
from .custom_logger import logger
from .utilities import (
    random_crops,
    depthwise_gaussian_kernel)

# ---------------------------------------------------------------------

PREPARE_DATA_FN_STR = "prepare_data"
DATASET_TESTING_FN_STR = "dataset_testing"
DATASET_TRAINING_FN_STR = "dataset_training"
DATASET_VALIDATION_FN_STR = "dataset_validation"
NOISE_AUGMENTATION_FN_STR = "noise_augmentation"
GEOMETRIC_AUGMENTATION_FN_STR = "geometric_augmentation"


# ---------------------------------------------------------------------


DatasetResults = namedtuple(
    "DatasetResults",
    {
        "config",
        "batch_size",
        "input_shape",
        "training",
        "testing"
    })

# ---------------------------------------------------------------------


def dataset_builder(
        config: Dict) -> tf.data.Dataset:
    # ---
    logger.info("creating dataset_builder with configuration [{0}]".format(config))

    # --- argument parsing
    batch_size = config["batch_size"]
    # crop image from dataset
    input_shape = config["input_shape"]
    color_mode = config.get("color_mode", "rgb").strip().lower()
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
    # ---
    inputs = config["inputs"]
    # directory to load data from
    directory = []
    if isinstance(inputs, list):
        for i in inputs:
            directory.append(i.get("directory", None))
    elif isinstance(inputs, dict):
        directory.append(config.get("directory", None))
    else:
        raise ValueError("dont know how to handle anything else than list and dict")

    # --- clip values to min max
    clip_value = config.get("clip_value", True)
    value_range = config.get("value_range", [0, 255])
    no_crops_per_image = config.get("no_crops_per_image", 1)
    min_value = tf.constant(value_range[0], dtype=tf.float32)
    max_value = tf.constant(value_range[1], dtype=tf.float32)

    # --- if true round values
    round_values = config.get("round_values", True)

    # --- dataset augmentation
    use_random_blur = config.get("random_blur", False)
    inpaint_drop_rate = config.get("inpaint_drop_rate", 0.0)

    # in radians
    random_rotate = config.get("random_rotate", 0.0)
    use_rotate = random_rotate > 0.0
    # if true randomly invert upside down image
    use_up_down = config.get("random_up_down", False)
    # if true randomly invert left right image
    use_left_right = config.get("random_left_right", False)
    additional_noise = config.get("additional_noise", [])
    use_additive_noise = len(additional_noise) > 0
    additive_noise = (min(additional_noise), max(additional_noise))
    multiplicative_noise = config.get("multiplicative_noise", [])
    use_multiplicative_noise = len(multiplicative_noise) > 0
    multiplicative_noise = (min(multiplicative_noise), max(multiplicative_noise))
    # quantization value, -1 disabled, otherwise 2, 4, 8
    quantization = config.get("quantization", -1)
    use_quantization = quantization > 1
    # jpeg noise
    use_jpeg_noise = config.get("use_jpeg_noise", False)
    jpeg_quality = [25, 75]

    # --- build noise options
    if len(additional_noise) <= 0:
        additional_noise = [1.0]

    if len(multiplicative_noise) <= 0:
        multiplicative_noise = [1.0]

    if quantization <= 1:
        quantization = 1

    additive_noise = tf.constant(additive_noise, dtype=tf.float32)
    multiplicative_noise = tf.constant(multiplicative_noise, dtype=tf.float32)

    gaussian_kernel = (
        tf.constant(
            depthwise_gaussian_kernel(
                channels=num_channels,
                kernel_size=(5, 5),
                nsig=(1, 1)).astype("float32")))

    @tf.function(reduce_retracing=True)
    def prepare_data_fn(input_batch: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Prepare the data for training by applying geometric and noise augmentations,
        and generating a binary mask for inpainting.

        Args:
            input_batch (tf.Tensor): The input tensor batch to be processed.

        Returns:
            Tuple[tf.Tensor, tf.Tensor]: A tuple containing the processed input batch,
            the noisy batch, and the binary mask.
        """

        def geometric_augmentation_fn(
                _input_batch: tf.Tensor) -> tf.Tensor:
            """
            perform all the geometric augmentations
            """

            # --- get shape and options
            random_option_flip_left_right = (
                tf.greater(tf.random.uniform(()), tf.constant(0.5)))
            random_option_flip_up_down = (
                tf.greater(tf.random.uniform(()), tf.constant(0.5)))

            # --- flip left right
            if use_left_right:
                _input_batch = \
                    tf.cond(
                        pred=random_option_flip_left_right,
                        true_fn=lambda: tf.image.flip_left_right(_input_batch),
                        false_fn=lambda: _input_batch)

            # --- flip up down
            if use_up_down:
                _input_batch = \
                    tf.cond(
                        pred=random_option_flip_up_down,
                        true_fn=lambda: tf.image.flip_up_down(_input_batch),
                        false_fn=lambda: _input_batch)
            return _input_batch

        def noise_augmentation_fn(
                _input_batch: tf.Tensor) -> tf.Tensor:
            """
            perform all the noise augmentations
            """
            # --- copy input batch
            _noisy_batch = _input_batch
            input_shape_inference = tf.shape(_noisy_batch)

            random_option_additive_noise = \
                tf.greater(
                    x=tf.random.uniform(()),
                    y=tf.constant(0.5))
            random_option_multiplicative_noise = \
                tf.greater(
                    x=tf.random.uniform(()),
                    y=tf.constant(0.5))
            random_option_embed_noise = \
                tf.greater(
                    x=tf.random.uniform(()),
                    y=tf.constant(0.5))
            additive_noise_std = \
                tf.random.uniform(
                    shape=(),
                    minval=additive_noise[0],
                    maxval=additive_noise[1])
            multiplicative_noise_std = \
                tf.random.uniform(
                    shape=(),
                    minval=multiplicative_noise[0],
                    maxval=multiplicative_noise[1])

            # --- set flags
            flag_multiplicative_noise = (
                tf.logical_and(
                    random_option_multiplicative_noise,
                    use_multiplicative_noise)
            )
            flag_additive_noise = (
                tf.logical_and(
                    random_option_additive_noise,
                    use_additive_noise)
            )
            flag_embed_noise = (
                random_option_embed_noise
            )

            # --- multiplicative noise
            _noisy_batch = \
                tf.cond(
                    pred=flag_multiplicative_noise,
                    true_fn=lambda:
                    tf.multiply(
                        x=_noisy_batch,
                        y=tf.random.truncated_normal(
                            mean=1.0,
                            seed=1,
                            stddev=multiplicative_noise_std,
                            shape=input_shape_inference,
                            dtype=tf.float32),
                    ),
                    false_fn=lambda: _noisy_batch
                )

            # --- additive noise
            _noisy_batch = \
                tf.cond(
                    pred=flag_additive_noise,
                    true_fn=lambda:
                    tf.add(
                        x=_noisy_batch,
                        y=tf.random.truncated_normal(
                            mean=0.0,
                            seed=1,
                            dtype=tf.float32,
                            stddev=additive_noise_std,
                            shape=input_shape_inference)
                    ),
                    false_fn=lambda: _noisy_batch
                )

            # --- embedd noise with a small filter
            _noisy_batch = \
                tf.cond(
                    pred=flag_embed_noise,
                    true_fn=lambda:
                        tf.nn.depthwise_conv2d(
                            input=_noisy_batch,
                            filter=gaussian_kernel,
                            strides=(1, 1, 1, 1),
                            data_format=None,
                            dilations=None,
                            padding="SAME"),
                    false_fn=lambda: _noisy_batch
                )

            # --- round values to nearest integer
            _noisy_batch = tf.round(x=_noisy_batch)

            return _noisy_batch

        # Apply geometric augmentations to the input batch
        input_batch = geometric_augmentation_fn(input_batch)
        input_batch = tf.round(input_batch)
        input_batch = tf.cast(input_batch, dtype=tf.float32)
        # Create a new batch with noise augmentations
        noisy_batch = noise_augmentation_fn(input_batch)
        return input_batch, noisy_batch

    # --- define generator function from directory
    if directory:
        dataset_generator = \
            image_filenames_generator(
                directory=directory)
        dataset_training = \
            tf.data.Dataset.from_generator(
                generator=dataset_generator,
                output_signature=(
                    tf.TensorSpec(shape=(), dtype=tf.string)
                ))
    else:
        raise ValueError("don't know how to handle non directory datasets")

    # --- save the augmentation functions
    @tf.function(reduce_retracing=True)
    def load_image_fn(path: tf.Tensor) -> tf.Tensor:
        img = \
            load_image(
                path=path,
                image_size=None,
                num_channels=num_channels,
                expand_dims=True,
                normalize=False)
        crops = \
            random_crops(
                input_batch=img,
                crop_size=(input_shape[0], input_shape[1]),
                x_range=None,
                y_range=None,
                no_crops_per_image=no_crops_per_image)
        del img

        return crops

    # --- create the dataset
    dataset_training = \
        dataset_training \
            .shuffle(
                seed=0,
                buffer_size=1024,
                reshuffle_each_iteration=True) \
            .map(
                map_func=load_image_fn,
                num_parallel_calls=tf.data.AUTOTUNE,
                deterministic=False) \
            .map(map_func=prepare_data_fn,
                 num_parallel_calls=tf.data.AUTOTUNE,
                 deterministic=False) \
            .unbatch() \
            .shuffle(
                seed=0,
                buffer_size=batch_size * 128,
                reshuffle_each_iteration=True) \
            .batch(
                batch_size=batch_size,
                drop_remainder=True) \
            .prefetch(buffer_size=2)

    return (
        DatasetResults(
            batch_size=batch_size,
            input_shape=input_shape,
            training=dataset_training,
            testing=None,
            config=config))

# ---------------------------------------------------------------------
