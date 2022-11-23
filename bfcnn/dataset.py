import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa
from typing import Dict, Callable, Iterator, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .custom_logger import logger
from .utilities import random_choice

# ---------------------------------------------------------------------

AUGMENTATION_FN_STR = "augmentation"
DATASET_TESTING_FN_STR = "dataset_testing"
DATASET_TRAINING_FN_STR = "dataset_training"
DATASET_VALIDATION_FN_STR = "dataset_validation"


# ---------------------------------------------------------------------


def dataset_builder(
        config: Dict):
    # ---
    logger.info("creating dataset_builder with configuration [{0}]".format(config))

    # --- argument parsing
    batch_size = config["batch_size"]
    # crop image from dataset
    input_shape = config["input_shape"]
    color_mode = config.get("color_mode", "rgb").strip().lower()
    if color_mode == "grayscale":
        channels = 1
    elif color_mode == "rgb":
        channels = 3
    elif color_mode == "rgba":
        channels = 4
    else:
        raise ValueError(f"don't know how to handle color_mode [{color_mode}]")
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
    value_range = config.get("value_range", [0, 255])
    min_value = tf.constant(value_range[0], dtype=tf.float32)
    max_value = tf.constant(value_range[1], dtype=tf.float32)
    clip_value = tf.constant(config.get("clip_value", True))

    # --- if true round values
    round_values = tf.constant(config.get("round_values", True))

    # --- dataset augmentation
    random_blur = tf.constant(config.get("random_blur", False))
    subsample_size = config.get("subsample_size", -1)
    # in radians
    random_rotate = tf.constant(config.get("random_rotate", 0.0))
    use_random_rotate = tf.constant(random_rotate > 0.0)
    # if true randomly invert upside down image
    random_up_down = tf.constant(config.get("random_up_down", False))
    # if true randomly invert left right image
    random_left_right = tf.constant(config.get("random_left_right", False))
    additional_noise = config.get("additional_noise", [])
    multiplicative_noise = config.get("multiplicative_noise", [])
    # quantization value, -1 disabled, otherwise 2, 4, 8
    quantization = tf.constant(config.get("quantization", -1))
    # min/max scale
    scale_range = config.get("scale_range", [0.9, 1.0])
    min_scale = tf.constant(scale_range[0], dtype=tf.float32)
    max_scale = tf.constant(scale_range[1], dtype=tf.float32)
    # whether to crop or not
    random_crop = tf.constant(dataset_shape[0][0:2] != input_shape[0:2])
    # mix noise types
    mix_noise_types = config.get("mix_noise_types", False)

    # --- build noise options
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

    noise_choices = tf.constant(noise_choices, dtype=tf.int64)
    additional_noise = tf.constant(additional_noise, dtype=tf.float32)
    multiplicative_noise = tf.constant(multiplicative_noise, dtype=tf.float32)

    # --- set random seed to get the same result
    tf.random.set_seed(0)

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=[None,
                                 None,
                                 None,
                                 channels],
                          dtype=tf.float32)],
        reduce_retracing=True,
        jit_compile=False)
    def cast_to_uint8(input_batch: tf.Tensor) -> tf.Tensor:
        return tf.cast(input_batch, tf.uint8)

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=[None,
                                 None,
                                 None,
                                 channels],
                          dtype=tf.uint8)],
        reduce_retracing=True,
        jit_compile=False)
    def cast_to_float32(input_batch: tf.Tensor) -> tf.Tensor:
        return tf.cast(input_batch, tf.float32)

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=[None,
                                 None,
                                 None,
                                 channels],
                          dtype=tf.float32)],
        reduce_retracing=True,
        jit_compile=False)
    def geometric_augmentations_fn(
            input_batch: tf.Tensor) -> tf.Tensor:
        """
        perform all the geometric augmentations
        """
        input_shape_inference = tf.shape(input_batch)

        # --- crop randomly
        if random_crop:
            # pick random number
            random_number = \
                tf.random.uniform(
                    shape=(1,),
                    minval=min_scale,
                    maxval=max_scale,
                    dtype=tf.dtypes.float32)[0]
            crop_height = \
                tf.cast(
                    tf.round(
                        random_number *
                        tf.cast(input_shape_inference[1], tf.float32)),
                    dtype=tf.int32)
            crop_width = \
                tf.cast(
                    tf.round(
                        random_number *
                        tf.cast(input_shape_inference[2], tf.float32)),
                    dtype=tf.int32)
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

        # --- cast down to get memory boost
        input_batch = tf.cast(input_batch, dtype=tf.uint8)

        # --- resize to input_shape
        input_batch = \
            tf.image.resize(
                images=input_batch,
                method=tf.image.ResizeMethod.BILINEAR,
                size=(input_shape[0], input_shape[1]))

        # --- flip left right
        if random_left_right and tf.random.uniform(()) > tf.constant(0.5):
            input_batch = \
                tf.image.flip_left_right(input_batch)

        # --- flip up down
        if random_up_down and tf.random.uniform(()) > tf.constant(0.5):
            input_batch = \
                tf.image.flip_up_down(input_batch)

        # --- randomly rotate input
        if use_random_rotate and tf.random.uniform(()) > tf.constant(0.5):
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

        return input_batch

    # --- define augmentation function
    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=[None,
                                 input_shape[0],
                                 input_shape[1],
                                 channels],
                          dtype=tf.float32)],
        reduce_retracing=True,
        jit_compile=False)
    def noise_augmentations_fn(
            input_batch: tf.Tensor) -> tf.Tensor:
        """
        perform all the noise augmentations
        """
        # --- copy input batch
        noisy_batch = tf.identity(input_batch)
        input_shape_inference = tf.shape(noisy_batch)

        # --- random select noise type
        noise_type = \
            random_choice(noise_choices, size=1)[0]
        additional_noise_std = \
            random_choice(additional_noise, size=1)[0]
        multiplicative_noise_std = \
            random_choice(multiplicative_noise, size=1)[0]

        logger.info(f"noise_type: {noise_type}")

        if noise_type == tf.constant(0, dtype=tf.int64):
            # additional noise
            if tf.random.uniform(()) > tf.constant(0.5):
                # channel independent noise
                noisy_batch = \
                    tf.math.add(
                        noisy_batch,
                        tf.random.truncated_normal(
                            mean=0,
                            stddev=additional_noise_std,
                            shape=input_shape_inference))
            else:
                # channel dependent noise
                tmp_noisy_batch = \
                    tf.random.truncated_normal(
                        mean=0,
                        stddev=additional_noise_std,
                        shape=(input_shape_inference[0],
                               input_shape[0],
                               input_shape[1],
                               1))
                tmp_noisy_batch = \
                    tf.repeat(
                        tmp_noisy_batch,
                        axis=3,
                        repeats=[input_shape_inference[3]])
                noisy_batch = tf.math.add(noisy_batch, tmp_noisy_batch)
            # blur to embed noise
            if random_blur and tf.random.uniform(()) > tf.constant(0.5):
                noisy_batch = \
                    tfa.image.gaussian_filter2d(
                        image=noisy_batch,
                        sigma=1,
                        filter_shape=(3, 3))
        elif noise_type == tf.constant(1, dtype=tf.int64):
            # multiplicative noise
            if tf.random.uniform(()) > tf.constant(0.5):
                # channel independent noise
                noisy_batch = \
                    noisy_batch * \
                    tf.random.truncated_normal(
                        mean=1,
                        stddev=multiplicative_noise_std,
                        shape=input_shape_inference)
            else:
                # channel dependent noise
                tmp_noisy_batch = \
                    tf.random.truncated_normal(
                        mean=1,
                        stddev=multiplicative_noise_std,
                        shape=(input_shape_inference[0],
                               input_shape[0],
                               input_shape[1],
                               1))
                tmp_noisy_batch = \
                    tf.repeat(
                        tmp_noisy_batch,
                        axis=3,
                        repeats=[input_shape_inference[3]])
                noisy_batch = tf.math.add(noisy_batch, tmp_noisy_batch)
            # blur to embed noise
            if random_blur and tf.random.uniform(()) > tf.constant(0.5):
                noisy_batch = \
                    tfa.image.gaussian_filter2d(
                        image=noisy_batch,
                        sigma=1,
                        filter_shape=(3, 3))
        elif noise_type == tf.constant(2, dtype=tf.int64):
            # downsample and upsample
            noisy_batch = \
                tf.image.resize(
                    images=noisy_batch,
                    method=tf.image.ResizeMethod.AREA,
                    size=(int(input_shape[0] / subsample_size),
                          int(input_shape[1] / subsample_size)))
            noisy_batch = \
                tf.image.resize(
                    images=noisy_batch,
                    method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
                    size=(input_shape[0], input_shape[1]))
        elif noise_type == tf.constant(3, dtype=tf.int64):
            # quantize values
            noisy_batch = tf.round(noisy_batch / quantization)
            noisy_batch = noisy_batch * quantization
            noisy_batch = tf.round(noisy_batch)
        else:
            logger.info(
                "don't know how to handle noise_type [{0}]".format(
                    noise_type))

        # --- round values to nearest integer
        if round_values:
            noisy_batch = tf.round(noisy_batch)

        # --- clip values within boundaries
        if clip_value:
            noisy_batch = \
                tf.clip_by_value(
                    noisy_batch,
                    clip_value_min=min_value,
                    clip_value_max=max_value)

        return noisy_batch

    # this will be use to augment data by mapping,
    # so each image in the tensor
    # is treated independently and gets a different noise type
    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=[input_shape[0],
                                 input_shape[1],
                                 channels],
                          dtype=tf.float32)])
    def noise_augmentations_map_fn(
            x_input: tf.Tensor) -> tf.Tensor:
        x_input = tf.expand_dims(x_input, axis=0)
        x_input = noise_augmentations_fn(x_input)
        x_input = tf.squeeze(x_input, axis=0)
        return x_input

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=[None,
                                 input_shape[0],
                                 input_shape[1],
                                 channels],
                          dtype=tf.float32)])
    def noise_augmentations_mix_fn(
            x_input: tf.Tensor) -> tf.Tensor:
        return tf.map_fn(
            fn=noise_augmentations_map_fn,
            elems=x_input,
            dtype=tf.float32,
            parallel_iterations=tf.shape(x_input)[0],
            back_prop=False,
            swap_memory=False,
            infer_shape=False,
        )

    # --- define generator function from directory
    if directory:
        dataset_training = [
            tf.keras.utils
                .image_dataset_from_directory(
                    directory=d,
                    labels=None,
                    label_mode=None,
                    class_names=None,
                    color_mode=color_mode,
                    batch_size=max(1, int(round(batch_size / len(directory)))),
                    shuffle=True,
                    image_size=s,
                    seed=0,
                    validation_split=None,
                    subset=None,
                    interpolation="area",
                    crop_to_aspect_ratio=True)
                .map(
                    map_func=geometric_augmentations_fn)
            for d, s in zip(directory, dataset_shape)
        ]
    else:
        raise ValueError("don't know how to handle non directory datasets")

    # --- create the dataset
    result = dict()

    if mix_noise_types:
        result[AUGMENTATION_FN_STR] = noise_augmentations_mix_fn

        @tf.function(
            input_signature=[
                tf.TensorSpec(shape=[None,
                                     input_shape[0],
                                     input_shape[1],
                                     channels],
                              dtype=tf.float32)])
        def augmentation_map_fn(
                x_input: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
            return x_input, noise_augmentations_mix_fn(x_input)
    else:
        result[AUGMENTATION_FN_STR] = noise_augmentations_fn

        @tf.function(
            input_signature=[
                tf.TensorSpec(shape=[None,
                                     input_shape[0],
                                     input_shape[1],
                                     channels],
                              dtype=tf.float32)])
        def augmentation_map_fn(
                x_input: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
            return x_input, noise_augmentations_fn(x_input)

    # dataset produces the dataset with basic geometric distortions
    if len(dataset_training) == 0:
        raise ValueError("don't know how to handle zero datasets")
    elif len(dataset_training) == 1:
        result[DATASET_TRAINING_FN_STR] = dataset_training[0]
    else:
        result[DATASET_TRAINING_FN_STR] = \
            tf.data.Dataset \
                .sample_from_datasets(datasets=dataset_training)

    # --- create proper batches by sampling from each dataset independently
    result[DATASET_TRAINING_FN_STR] = \
        result[DATASET_TRAINING_FN_STR] \
            .prefetch(buffer_size=(batch_size * 2)) \
            .rebatch(batch_size=batch_size) \
            .map(map_func=cast_to_float32,
                 num_parallel_calls=2) \
            .prefetch(2)

    return result

# ---------------------------------------------------------------------
