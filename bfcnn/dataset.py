import copy
import tensorflow as tf
from collections import namedtuple
from typing import Dict, Callable, Iterator, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .file_operations import *
from .custom_logger import logger
from .utilities import random_crops

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
    # resolution of the files loaded (reshape)
    dataset_shape = []
    if isinstance(inputs, list):
        for i in inputs:
            directory.append(i.get("directory", None))
            dataset_shape.append(i.get("dataset_shape", [256, 256]))
    elif isinstance(inputs, dict):
        directory.append(config.get("directory", None))
        dataset_shape.append(config.get("dataset_shape", [256, 256]))
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
    # in radians
    random_rotate = config.get("random_rotate", 0.0)
    use_rotate = random_rotate > 0.0
    # if true randomly invert upside down image
    use_up_down = config.get("random_up_down", False)
    # if true randomly invert left right image
    use_left_right = config.get("random_left_right", False)
    additional_noise = config.get("additional_noise", [])
    use_additive_noise = len(additional_noise) > 0
    multiplicative_noise = config.get("multiplicative_noise", [])
    use_multiplicative_noise = len(multiplicative_noise) > 0
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

    additional_noise = tf.constant(additional_noise, dtype=tf.float32)
    multiplicative_noise = tf.constant(multiplicative_noise, dtype=tf.float32)

    def adjust_jpeg_quality(x: tf.Tensor) -> tf.Tensor:
        tmp_list = []
        tmp = tf.cast(x, dtype=tf.float32) / 255

        for t in tf.unstack(tmp, axis=0):
            tmp_list.append(
                tf.image.random_jpeg_quality(
                    image=t,
                    min_jpeg_quality=jpeg_quality[0],
                    max_jpeg_quality=jpeg_quality[1],
                    seed=0
                )
            )

        return tf.stack(tmp_list, axis=0) * 255
    def geometric_augmentation_fn(
            input_batch: tf.Tensor) -> tf.Tensor:
        """
        perform all the geometric augmentations
        """

        # --- get shape and options
        input_shape_inference = tf.shape(input_batch)
        random_option_flip_left_right = tf.greater(tf.random.uniform((), seed=0), tf.constant(0.5))
        random_option_flip_up_down = tf.greater(tf.random.uniform((), seed=0), tf.constant(0.5))
        random_option_rotate = tf.greater(tf.random.uniform((), seed=0), tf.constant(0.5))

        # --- flip left right
        if use_left_right:
            input_batch = \
                tf.cond(
                    pred=random_option_flip_left_right,
                    true_fn=lambda: tf.image.flip_left_right(input_batch),
                    false_fn=lambda: input_batch)

        # --- flip up down
        if use_up_down:
            input_batch = \
                tf.cond(
                    pred=random_option_flip_up_down,
                    true_fn=lambda: tf.image.flip_up_down(input_batch),
                    false_fn=lambda: input_batch)

        # # --- randomly rotate input
        # if use_rotate:
        #     input_batch = \
        #         tf.cond(
        #             pred=random_option_rotate,
        #             true_fn=lambda:
        #             tfa.image.rotate(
        #                 angles=tf.random.uniform(
        #                     dtype=tf.float32,
        #                     seed=0,
        #                     minval=-random_rotate,
        #                     maxval=random_rotate,
        #                     shape=(input_shape_inference[0],)),
        #                 images=input_batch,
        #                 fill_mode="reflect",
        #                 interpolation="bilinear"),
        #             false_fn=lambda: input_batch)

        return input_batch

    # --- define noise augmentation function
    def random_choice(
            x: tf.Tensor,
            size=tf.constant(1, dtype=tf.int64),
            axis=tf.constant(0, dtype=tf.int64)) -> tf.Tensor:
        """
        Randomly select size options from x
        """
        dim_x = tf.cast(tf.shape(x)[axis], tf.int64)
        if dim_x == tf.constant(0, tf.int64):
            return tf.constant(-1, dtype=x.dtype)
        indices = tf.range(0, dim_x, dtype=tf.int64)
        sample_index = tf.random.shuffle(indices, seed=0)[:size]
        return tf.gather(x, sample_index, axis=axis)[0]

    def noise_augmentation_fn(
            input_batch: tf.Tensor) -> tf.Tensor:
        """
        perform all the noise augmentations
        """
        # --- copy input batch
        noisy_batch = input_batch
        input_shape_inference = tf.shape(noisy_batch)

        # --- random select noise type and options
        additive_noise_std = random_choice(additional_noise, size=1)
        multiplicative_noise_std = random_choice(multiplicative_noise, size=1)

        random_option_blur = tf.greater(tf.random.uniform((), seed=0), tf.constant(0.5))
        random_option_quantize = tf.greater(tf.random.uniform((), seed=0), tf.constant(0.5))
        random_option_jpeg_noise = tf.greater(tf.random.uniform((), seed=0), tf.constant(0.5))
        random_option_additive_noise = tf.greater(tf.random.uniform((), seed=0), tf.constant(0.5))
        random_option_multiplicative_noise = tf.greater(tf.random.uniform((), seed=0), tf.constant(0.5))
        random_option_channel_independent_noise = tf.greater(tf.random.uniform((), seed=0), tf.constant(0.5))

        # --- jpeg noise
        noisy_batch = \
            tf.cond(
                pred=tf.math.logical_and(
                    use_jpeg_noise,
                    random_option_jpeg_noise),
                true_fn=lambda: adjust_jpeg_quality(noisy_batch),
                false_fn=lambda: noisy_batch
            )

        # --- quantization noise
        noisy_batch = \
            tf.cond(
                pred=tf.math.logical_and(
                    use_quantization,
                    random_option_quantize),
                true_fn=lambda:
                tf.quantization.quantize_and_dequantize_v2(
                    tf.cast(noisy_batch, dtype=tf.float32),
                    input_min=min_value,
                    input_max=max_value,
                    num_bits=quantization),
                false_fn=lambda: noisy_batch
            )

        # --- additive noise
        if use_additive_noise:
            noisy_batch = \
                tf.cond(
                    pred=random_option_additive_noise,
                    true_fn=lambda:
                    tf.add(
                        x=noisy_batch,
                        y=tf.cond(
                            pred=random_option_channel_independent_noise,
                            true_fn=lambda:
                            # channel independent noise
                            tf.random.truncated_normal(
                                mean=0.0,
                                seed=0,
                                dtype=tf.float32,
                                stddev=additive_noise_std,
                                shape=input_shape_inference),
                            false_fn=lambda:
                            # channel dependent noise
                            tf.random.truncated_normal(
                                mean=0.0,
                                seed=0,
                                dtype=tf.float32,
                                stddev=additive_noise_std,
                                shape=(input_shape_inference[0],
                                       input_shape_inference[1],
                                       input_shape_inference[2],
                                       1))
                        )),
                    false_fn=lambda: noisy_batch
                )

        # --- multiplicative noise
        if use_multiplicative_noise:
            noisy_batch = \
                tf.cond(
                    pred=random_option_multiplicative_noise,
                    true_fn=lambda:
                    tf.multiply(
                        x=noisy_batch,
                        y=tf.cond(
                            pred=random_option_channel_independent_noise,
                            true_fn=lambda:
                            # channel independent noise
                            tf.random.truncated_normal(
                                mean=1.0,
                                seed=0,
                                stddev=multiplicative_noise_std,
                                shape=input_shape_inference,
                                dtype=tf.float32),
                            false_fn=lambda:
                            # channel dependent noise
                            tf.random.truncated_normal(
                                mean=1.0,
                                seed=0,
                                stddev=multiplicative_noise_std,
                                shape=(input_shape_inference[0],
                                       input_shape_inference[1],
                                       input_shape_inference[2],
                                       1),
                                dtype=tf.float32)
                        )),
                    false_fn=lambda: noisy_batch
                )

        # # --- blur to embed noise
        # if use_random_blur:
        #     noisy_batch = \
        #         tf.cond(
        #             pred=random_option_blur,
        #             true_fn=lambda:
        #             tfa.image.gaussian_filter2d(
        #                 image=noisy_batch,
        #                 sigma=0.5,
        #                 filter_shape=(5, 5)),
        #             false_fn=lambda: noisy_batch
        #         )

        # --- round values to nearest integer
        if round_values:
            noisy_batch = tf.round(x=noisy_batch)

        # --- clip values within boundaries
        if clip_value:
            noisy_batch = \
                tf.clip_by_value(
                    t=noisy_batch,
                    clip_value_min=min_value,
                    clip_value_max=max_value)

        return noisy_batch

    def prepare_data_fn(input_batch: tf.Tensor) -> \
            Tuple[tf.Tensor, tf.Tensor]:
        input_batch = geometric_augmentation_fn(input_batch)
        input_batch = tf.round(input_batch)
        input_batch = tf.cast(input_batch, dtype=tf.float32)
        noisy_batch = noise_augmentation_fn(input_batch)
        return input_batch, noisy_batch

    # --- define generator function from directory
    if directory:
        dataset_generator = \
            image_filenames_generator(
                directory=directory)
        dataset_size = sum(1 for _ in copy.deepcopy(dataset_generator)())
        dataset_training = \
            tf.data.Dataset.from_generator(
                generator=dataset_generator,
                output_signature=(
                    tf.TensorSpec(shape=(), dtype=tf.string)
                ))
        logger.info(f"dataset_size: [{dataset_size}]")
    else:
        raise ValueError("don't know how to handle non directory datasets")

    # --- save the augmentation functions
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
            .rebatch(
                batch_size=batch_size,
                drop_remainder=True) \
            .prefetch(buffer_size=1)

    return (
        DatasetResults(
            batch_size=batch_size,
            input_shape=input_shape,
            training=dataset_training,
            testing=None,
            config=config))

# ---------------------------------------------------------------------
