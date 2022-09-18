import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa
from typing import Dict, Callable, Iterator

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .custom_logger import logger
from .utilities import merge_iterators, random_choice

# ---------------------------------------------------------------------

DATASET_FN_STR = "dataset"
AUGMENTATION_FN_STR = "augmentation"
DATASET_TESTING_FN_STR = "dataset_testing"


# ---------------------------------------------------------------------


def dataset_builder(
        config: Dict):
    # ---
    logger.info("creating dataset_builder with configuration [{0}]".format(config))
    # --- argument parsing
    batch_size = config["batch_size"]
    # crop image from dataset
    input_shape = config["input_shape"]
    color_mode = config.get("color_mode", "rgb")

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
    # quantization value, -1 disabled, otherwise 2, 4, 8
    quantization = tf.constant(config.get("quantization", -1))
    # min/max scale
    min_scale = tf.constant(config.get("min_scale", 0.1))
    max_scale = tf.constant(config.get("max_scale", 0.9))
    # whether to crop or not
    random_crop = dataset_shape[0][0:2] != input_shape[0:2]
    random_crop = tf.constant(random_crop)

    # build noise options
    noise_choices = []
    if len(additional_noise) > 0:
        noise_choices.append(0)
    else:
        additional_noise = [1.0]

    if len(multiplicative_noise) > 0:
        noise_choices.append(1)
    else:
        multiplicative_noise = [1.0]

    if subsample_size > 0:
        noise_choices.append(2)
    else:
        subsample_size = 2

    if quantization > 1:
        noise_choices.append(3)
    else:
        quantization = 1

    noise_choices = tf.constant(noise_choices)
    additional_noise = tf.constant(additional_noise, dtype=tf.float32)
    multiplicative_noise = tf.constant(multiplicative_noise, dtype=tf.float32)

    # --- define generator function from directory
    if directory:
        dataset = [
            tf.keras.utils.image_dataset_from_directory(
                directory=d,
                labels=None,
                label_mode=None,
                class_names=None,
                color_mode=color_mode,
                batch_size=batch_size,
                shuffle=True,
                image_size=s,
                seed=0,
                validation_split=None,
                subset=None,
                interpolation="bilinear",
                crop_to_aspect_ratio=True)
            for d, s in zip(directory, dataset_shape)
        ]

    else:
        raise ValueError("don't know how to handle non directory datasets")

    def input_batch_augmentations(input_batch):
        input_shape_inference = tf.shape(input_batch)

        # --- crop randomly
        if random_crop:
            # pick random number
            random_numbers = \
                tf.random.uniform(
                    shape=(1,),
                    minval=min_scale,
                    maxval=max_scale,
                    dtype=tf.dtypes.float32,
                )
            crop_width = \
                tf.cast(
                    tf.round(
                        random_numbers[0] *
                        tf.cast(input_shape_inference[1], tf.float32)),
                    dtype=tf.int32) + 1
            crop_height = \
                tf.cast(
                    tf.round(
                        random_numbers[0] *
                        tf.cast(input_shape_inference[2], tf.float32)),
                    dtype=tf.int32) + 1
            crop_size = \
                tf.math.minimum(
                    x=crop_width,
                    y=crop_height)
            # crop
            input_batch = \
                tf.image.random_crop(
                    value=input_batch,
                    size=(
                        input_shape_inference[0],
                        crop_size,
                        crop_size,
                        input_shape_inference[3])
                )

        # --- resize to input_shape
        input_batch = \
            tf.image.resize(
                images=input_batch,
                size=(input_shape[0], input_shape[1]))
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

        # --- clip values within boundaries
        if clip_value:
            input_batch = \
                tf.clip_by_value(
                    input_batch,
                    clip_value_min=min_value,
                    clip_value_max=max_value)

        # --- round values to nearest integer
        if round_values:
            input_batch = tf.round(input_batch)

        # --- convert to float32
        input_batch = \
            tf.cast(input_batch, dtype=tf.dtypes.float32)

        return input_batch

    # --- define augmentation function
    def augmentation(input_batch):
        # --- copy input batch
        noisy_batch = tf.identity(input_batch)
        input_shape_inference = tf.shape(noisy_batch)

        # --- random select noise type
        noise_type = \
            random_choice(noise_choices, size=1)[0]

        logger.info(f"noise_type: {noise_type}")

        if noise_type == 0:
            # additional noise
            noise_std = random_choice(additional_noise, size=1)[0]
            if tf.random.uniform(()) > 0.5:
                # channel independent noise
                noisy_batch = \
                    noisy_batch + \
                    tf.random.truncated_normal(
                        mean=0,
                        stddev=noise_std,
                        shape=input_shape_inference)
            else:
                # channel dependent noise
                tmp_noisy_batch = \
                    tf.random.truncated_normal(
                        mean=0,
                        stddev=noise_std,
                        shape=(input_shape_inference[0],
                               input_shape[0],
                               input_shape[1],
                               1))
                tmp_noisy_batch = \
                    tf.repeat(
                        tmp_noisy_batch,
                        axis=3,
                        repeats=[input_shape_inference[3]])
                noisy_batch = noisy_batch + tmp_noisy_batch
            # blur to embed noise
            if random_blur:
                if tf.random.uniform(()) > 0.5:
                    noisy_batch = \
                        tfa.image.gaussian_filter2d(
                            image=noisy_batch,
                            sigma=1,
                            filter_shape=(3, 3))
        elif noise_type == 1:
            # multiplicative noise
            noise_std = \
                random_choice(multiplicative_noise, size=1)[0]
            if tf.random.uniform(()) > 0.5:
                # channel independent noise
                noisy_batch = \
                    noisy_batch * \
                    tf.random.truncated_normal(
                        mean=1,
                        stddev=noise_std,
                        shape=input_shape_inference)
            else:
                # channel dependent noise
                tmp_noisy_batch = \
                    tf.random.truncated_normal(
                        mean=1,
                        stddev=noise_std,
                        shape=(input_shape_inference[0],
                               input_shape[0],
                               input_shape[1],
                               1))
                tmp_noisy_batch = \
                    tf.repeat(
                        tmp_noisy_batch,
                        axis=3,
                        repeats=[input_shape_inference[3]])
                noisy_batch = noisy_batch * tmp_noisy_batch

            # blur to embed noise
            if random_blur:
                if tf.random.uniform(()) > 0.5:
                    noisy_batch = \
                        tfa.image.gaussian_filter2d(
                            image=noisy_batch,
                            sigma=1,
                            filter_shape=(3, 3))
        elif noise_type == 2:
            # subsample
            stride = (subsample_size, subsample_size)
            noisy_batch = \
                keras.layers.MaxPool2D(
                    pool_size=(1, 1),
                    strides=stride)(noisy_batch)
            noisy_batch = \
                keras.layers.UpSampling2D(
                    size=stride)(noisy_batch)
        elif noise_type == 3:
            # quantize values
            noisy_batch = tf.round(noisy_batch / quantization)
            noisy_batch = noisy_batch * quantization
            noisy_batch = tf.round(noisy_batch)
        else:
            logger.info(
                "don't know how to handle noise_type [{0}]".format(
                    noise_type))

        # --- clip values within boundaries
        if clip_value:
            noisy_batch = \
                tf.clip_by_value(
                    noisy_batch,
                    clip_value_min=min_value,
                    clip_value_max=max_value)

        # --- round values to nearest integer
        if round_values:
            noisy_batch = tf.round(noisy_batch)

        return noisy_batch

    # --- create the dataset
    result = dict()

    result[AUGMENTATION_FN_STR] = augmentation

    # dataset produces the dataset with basic geometric distortions
    if len(dataset) == 0:
        raise ValueError("don't know how to handle zero datasets")
    elif len(dataset) == 1:
        result[DATASET_FN_STR] = \
            dataset[0].map(
                map_func=input_batch_augmentations,
                num_parallel_calls=tf.data.AUTOTUNE).prefetch(2)
    else:
        result[DATASET_FN_STR] = \
            tf.data.Dataset.sample_from_datasets(dataset).map(
                map_func=input_batch_augmentations,
                num_parallel_calls=tf.data.AUTOTUNE).prefetch(2)

    return result

# ---------------------------------------------------------------------
