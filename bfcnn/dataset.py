import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa
from typing import Dict, Callable, Iterator, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .custom_logger import logger
from .utilities import merge_iterators
from .dataset_file_operation import image_dataset_from_directory

# ---------------------------------------------------------------------

AUGMENTATION_FN_STR = "augmentation"
DATASET_TESTING_FN_STR = "dataset_testing"
DATASET_TRAINING_FN_STR = "dataset_training"
DATASET_VALIDATION_FN_STR = "dataset_validation"
GEOMETRIC_AUGMENTATION_FN_STR = "geometric_augmentation"


def random_crops(
        input_batch: tf.Tensor,
        no_crops_per_image: int = 16,
        crop_size: Tuple[int, int] = (64, 64),
        x_range: Tuple[float, float] = None,
        y_range: Tuple[float, float] = None,
        extrapolation_value: float = 0.0,
        interpolation_method: str = "bilinear") -> tf.Tensor:
    """
    random crop from each image in the batch

    :param input_batch: 4D tensor
    :param no_crops_per_image: number of crops per image in batch
    :param crop_size: final crop size output
    :param x_range: manually set x_range
    :param y_range: manually set y_range
    :param extrapolation_value: value set to beyond the image crop
    :param interpolation_method: interpolation method
    :return: tensor with shape
        [input_batch[0] * no_crops_per_image,
         crop_size[0],
         crop_size[1],
         input_batch[3]]
    """
    shape = tf.shape(input_batch)
    batch_size = shape[0]
    # computer the total number of crops
    total_crops = no_crops_per_image * batch_size

    # fill y_range, x_range based on crop size and input batch size
    if y_range is None:
        y_range = (float(crop_size[0] / shape[1]),
                   float(crop_size[0] / shape[1] + 0.01))

    if x_range is None:
        x_range = (float(crop_size[1] / shape[2]),
                   float(crop_size[1] / shape[2] + 0.01))

    #
    y1 = tf.random.uniform(
        shape=(total_crops, 1), minval=0.0, maxval=1.0 - y_range[0])
    y2 = y1 + \
         tf.random.uniform(
             shape=(total_crops, 1), minval=y_range[0], maxval=y_range[1])
    #
    x1 = tf.random.uniform(
        shape=(total_crops, 1), minval=0.0, maxval=1.0 - x_range[0])
    x2 = x1 + \
         tf.random.uniform(
             shape=(total_crops, 1), minval=x_range[0], maxval=x_range[1])
    # limit the crops to the end of image
    y1 = tf.maximum(y1, 0.0)
    y2 = tf.minimum(y2, 1.0)
    x1 = tf.maximum(x1, 0.0)
    x2 = tf.minimum(x2, 1.0)
    # concat the dimensions to create [total_crops, 4] boxes
    boxes = tf.concat([y1, x1, y2, x2], axis=1)

    # --- randomly choose the image to crop inside the batch
    box_indices = \
        tf.random.uniform(
            shape=(total_crops,),
            minval=0,
            maxval=batch_size,
            dtype=tf.int32)

    result = \
        tf.image.crop_and_resize(
            input_batch,
            boxes,
            box_indices,
            crop_size,
            method=interpolation_method,
            extrapolation_value=extrapolation_value)

    return result


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
    # whether to crop or not
    random_crop = tf.constant(dataset_shape[0][0:2] != input_shape[0:2])
    # mix noise types
    mix_noise_types = config.get("mix_noise_types", False)
    # no crops per image
    no_crops_per_image = config.get("no_crops_per_image", 1)

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

    @tf.function
    def cast_to_float32(input_batch: tf.Tensor) -> tf.Tensor:
        return tf.cast(input_batch, tf.float32)

    @tf.function
    def random_choice(
            x: tf.Tensor,
            size=tf.constant(1, dtype=tf.int64),
            axis=tf.constant(0, dtype=tf.int64)) -> tf.Tensor:
        """
        Randomly select size options from x
        """
        dim_x = tf.cast(tf.shape(x)[axis], tf.int64)
        indices = tf.range(0, dim_x, dtype=tf.int64)
        sample_index = tf.random.shuffle(indices)[:size]
        return tf.gather(x, sample_index, axis=axis)

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=[None,
                                 None,
                                 None,
                                 channels],
                          dtype=tf.float32)],
        reduce_retracing=False,
        jit_compile=False)
    def crop_fn(
            input_batch: tf.Tensor) -> tf.Tensor:
        """
        crop patches from input
        """
        input_batch = \
            tf.cond(
                pred=random_crop,
                true_fn=lambda:
                random_crops(
                    input_batch=input_batch,
                    crop_size=(input_shape[0], input_shape[1]),
                    no_crops_per_image=no_crops_per_image),
                false_fn=lambda: input_batch)
        return tf.cast(input_batch, dtype=tf.uint8)

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=[None,
                                 input_shape[0],
                                 input_shape[1],
                                 channels],
                          dtype=tf.uint8)],
        reduce_retracing=False,
        jit_compile=False)
    def geometric_augmentations_fn(
            input_batch: tf.Tensor) -> tf.Tensor:
        """
        perform all the geometric augmentations
        """

        # --- get shape and options
        input_shape_inference = tf.shape(input_batch)
        random_uniform_option_1 = tf.greater(tf.random.uniform(()), tf.constant(0.5))
        random_uniform_option_2 = tf.greater(tf.random.uniform(()), tf.constant(0.5))
        random_uniform_option_3 = tf.greater(tf.random.uniform(()), tf.constant(0.5))

        # --- flip left right
        input_batch = \
            tf.cond(
                pred=tf.math.logical_and(random_left_right,
                                         random_uniform_option_1),
                true_fn=lambda: tf.image.flip_left_right(input_batch),
                false_fn=lambda: input_batch)

        # --- flip up down
        input_batch = \
            tf.cond(
                pred=tf.math.logical_and(random_up_down,
                                         random_uniform_option_2),
                true_fn=lambda: tf.image.flip_up_down(input_batch),
                false_fn=lambda: input_batch)

        # --- randomly rotate input
        input_batch = \
            tf.cond(
                pred=tf.math.logical_and(use_random_rotate,
                                         random_uniform_option_3),
                true_fn=lambda:
                tfa.image.rotate(
                    angles=tf.random.uniform(
                        dtype=tf.float32,
                        minval=-random_rotate,
                        maxval=random_rotate,
                        shape=(input_shape_inference[0],)),
                    images=input_batch,
                    fill_mode="reflect",
                    interpolation="bilinear"),
                false_fn=lambda: input_batch)

        return input_batch

    # --- define augmentation function
    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=[None,
                                 input_shape[0],
                                 input_shape[1],
                                 channels],
                          dtype=tf.float32)],
        reduce_retracing=False,
        jit_compile=False)
    def noise_augmentations_fn(
            input_batch: tf.Tensor) -> tf.Tensor:
        """
        perform all the noise augmentations
        """
        # --- copy input batch
        noisy_batch = input_batch
        input_shape_inference = tf.shape(noisy_batch)

        # --- random select noise type and options
        noise_type = random_choice(noise_choices, size=1)[0]
        additive_noise_std = random_choice(additional_noise, size=1)[0]
        multiplicative_noise_std = random_choice(multiplicative_noise, size=1)[0]
        random_uniform_option_1 = tf.greater(tf.random.uniform(()), tf.constant(0.5))
        random_uniform_option_2 = tf.greater(tf.random.uniform(()), tf.constant(0.5))

        use_additive_noise = tf.equal(noise_type, tf.constant(0, dtype=tf.int64))
        use_multiplicative_noise = tf.equal(noise_type, tf.constant(1, dtype=tf.int64))
        use_subsample_noise = tf.equal(noise_type, tf.constant(2, dtype=tf.int64))
        use_quantize_noise = tf.equal(noise_type, tf.constant(3, dtype=tf.int64))

        # --- additive noise
        if use_additive_noise:
            additive_noise_batch = \
                tf.cond(
                    pred=random_uniform_option_1,
                    true_fn=lambda:
                    # channel independent noise
                    tf.random.truncated_normal(
                        mean=0,
                        stddev=additive_noise_std,
                        shape=input_shape_inference),
                    false_fn=lambda:
                    # channel dependent noise
                    tf.repeat(
                        tf.random.truncated_normal(
                            mean=0,
                            stddev=additive_noise_std,
                            shape=(input_shape_inference[0],
                                   input_shape[0],
                                   input_shape[1],
                                   1)),
                        axis=3,
                        repeats=[input_shape_inference[3]]))
            noisy_batch = tf.math.add(noisy_batch, additive_noise_batch)
            # blur to embed noise
            noisy_batch = \
                tf.cond(
                    pred=tf.math.logical_and(
                        random_blur,
                        random_uniform_option_2),
                    true_fn=lambda:
                    tfa.image.gaussian_filter2d(
                        image=noisy_batch,
                        sigma=1,
                        filter_shape=(3, 3)),
                    false_fn=lambda: noisy_batch)

        # --- multiplicative noise
        if use_multiplicative_noise:
            multiplicative_noise_batch = \
                tf.cond(
                    pred=random_uniform_option_1,
                    true_fn=lambda:
                    tf.random.truncated_normal(
                        mean=1,
                        stddev=multiplicative_noise_std,
                        shape=input_shape_inference),
                    false_fn=lambda:
                    tf.repeat(
                        tf.random.truncated_normal(
                            mean=1,
                            stddev=multiplicative_noise_std,
                            shape=(input_shape_inference[0],
                                   input_shape_inference[1],
                                   input_shape_inference[2],
                                   1)),
                        axis=3,
                        repeats=[input_shape_inference[3]])
                )
            noisy_batch = tf.math.multiply(noisy_batch, multiplicative_noise_batch)
            # blur to embed noise
            noisy_batch = \
                tf.cond(
                    pred=tf.math.logical_and(
                        random_blur,
                        random_uniform_option_2),
                    true_fn=lambda:
                    tfa.image.gaussian_filter2d(
                        image=noisy_batch,
                        sigma=1,
                        filter_shape=(3, 3)),
                    false_fn=lambda: noisy_batch)

        # -- subsample noise
        if use_subsample_noise:
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

        # --- quantize noise
        if use_quantize_noise:
            noisy_batch = tf.round(noisy_batch / quantization)
            noisy_batch = noisy_batch * quantization
            noisy_batch = tf.round(noisy_batch)

        # --- round values to nearest integer
        noisy_batch = \
            tf.cond(
                pred=round_values,
                true_fn=lambda: tf.round(noisy_batch),
                false_fn=lambda: noisy_batch)

        # --- clip values within boundaries
        noisy_batch = \
            tf.cond(
                pred=clip_value,
                true_fn=lambda:
                tf.clip_by_value(
                    noisy_batch,
                    clip_value_min=min_value,
                    clip_value_max=max_value),
                false_fn=lambda: noisy_batch)

        return noisy_batch

    # this will be use to augment data by mapping,
    # so each image in the tensor
    # is treated independently and gets a different noise type
    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=[None,
                                 input_shape[0],
                                 input_shape[1],
                                 channels],
                          dtype=tf.float32)])
    def noise_augmentations_mix_fn(
            x_input: tf.Tensor) -> tf.Tensor:

        def noise_augmentations_map_fn(
                x: tf.Tensor) -> tf.Tensor:
            x = tf.expand_dims(x, axis=0)
            x = noise_augmentations_fn(x)
            x = tf.squeeze(x, axis=0)
            return x

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
                image_dataset_from_directory(
                    seed=0,
                    directory=d,
                    shuffle=True,
                    image_size=s,
                    interpolation=tf.image.ResizeMethod.AREA,
                    color_mode=color_mode,
                    batch_size=batch_size)
                .map(
                    map_func=crop_fn,
                    num_parallel_calls=tf.data.AUTOTUNE,
                    deterministic=False)\
            for d, s in zip(directory, dataset_shape)
        ]
    else:
        raise ValueError("don't know how to handle non directory datasets")

    # --- create the dataset
    result = dict()
    result[GEOMETRIC_AUGMENTATION_FN_STR] = geometric_augmentations_fn

    if mix_noise_types:
        result[AUGMENTATION_FN_STR] = noise_augmentations_mix_fn
    else:
        result[AUGMENTATION_FN_STR] = noise_augmentations_fn

    # dataset produces the dataset with basic geometric distortions
    if len(dataset_training) == 0:
        raise ValueError("don't know how to handle zero datasets")

    def generator_fn():
        for x in merge_iterators(*dataset_training):
            yield x

    result[DATASET_TRAINING_FN_STR] = \
        tf.data.Dataset.from_generator(
            generator=generator_fn,
            output_signature=(
                tf.TensorSpec(shape=(None,
                                     input_shape[0],
                                     input_shape[1],
                                     channels),
                              dtype=tf.uint8)
            )) \
            .unbatch() \
            .shuffle(
                buffer_size=(no_crops_per_image *
                             batch_size *
                             len(dataset_training)),
                reshuffle_each_iteration=True) \
            .batch(
                batch_size=batch_size,
                deterministic=False,
                num_parallel_calls=tf.data.AUTOTUNE) \
            .prefetch(1)

    return result

# ---------------------------------------------------------------------
