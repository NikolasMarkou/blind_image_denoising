import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa
from typing import Dict, Callable, Iterator, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .custom_logger import logger
from .utilities import merge_iterators
from .pyramid import downsample_2x2_block
from .dataset_file_operation import \
    image_filenames_dataset_from_directory_gen, \
    load_image_crop

# ---------------------------------------------------------------------

DATASET_TESTING_FN_STR = "dataset_testing"
DATASET_TRAINING_FN_STR = "dataset_training"
DATASET_VALIDATION_FN_STR = "dataset_validation"
NOISE_AUGMENTATION_FN_STR = "noise_augmentation"
INPAINT_AUGMENTATION_FN_STR = "inpaint_augmentation"
SUPERRES_AUGMENTATION_FN_STR = "superres_augmentation"
GEOMETRIC_AUGMENTATION_FN_STR = "geometric_augmentation"


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
                                 input_shape[0],
                                 input_shape[1],
                                 channels],
                          dtype=tf.uint8)],
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

    # --- define inpainting augmentation function
    def inpaint_augmentation_fn(
            input_batch: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        input_shape_inference = tf.shape(input_batch)

        # channel dependent noise
        mask_batch = \
            tf.random.uniform(
                seed=0,
                minval=-0.5,
                maxval=0.5,
                dtype=tf.float16,
                shape=(input_shape_inference[0],
                       input_shape_inference[1],
                       input_shape_inference[2],
                       1))

        mask_batch = \
            tf.cast(
                x=tf.greater(mask_batch,
                             tf.constant(0.0, dtype=tf.float16)),
                dtype=tf.float32)

        return input_batch, mask_batch

    # --- define superres augmentation function
    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=[None,
                                 None,
                                 None,
                                 channels],
                          dtype=tf.float32)],
        jit_compile=False)
    def superres_augmentation_fn(
            input_batch: tf.Tensor) -> tf.Tensor:
        downsampled_batch = \
            downsample_2x2_block(
                input_layer=input_batch,
                kernel_size=(3, 3),
                xy_max=(1.0, 1.0))
        return downsampled_batch

    # --- define noise augmentation function
    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=[None,
                                 input_shape[0],
                                 input_shape[1],
                                 channels],
                          dtype=tf.float32)],
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
                        mean=0.0,
                        dtype=tf.float32,
                        stddev=additive_noise_std,
                        shape=input_shape_inference),
                    false_fn=lambda:
                    # channel dependent noise
                    tf.random.truncated_normal(
                        mean=0.0,
                        dtype=tf.float32,
                        stddev=additive_noise_std,
                        shape=(input_shape_inference[0],
                               input_shape[0],
                               input_shape[1],
                               1))
                )
            noisy_batch = tf.add(noisy_batch, additive_noise_batch)
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
                    false_fn=lambda: noisy_batch
                )

        # --- multiplicative noise
        if use_multiplicative_noise:
            multiplicative_noise_batch = \
                tf.cond(
                    pred=random_uniform_option_1,
                    true_fn=lambda:
                    tf.random.truncated_normal(
                        mean=1.0,
                        stddev=multiplicative_noise_std,
                        shape=input_shape_inference,
                        dtype=tf.float32),
                    false_fn=lambda:
                    tf.random.truncated_normal(
                        mean=1.0,
                        stddev=multiplicative_noise_std,
                        shape=(input_shape_inference[0],
                               input_shape_inference[1],
                               input_shape_inference[2],
                               1),
                        dtype=tf.float32)
                )
            noisy_batch = tf.multiply(noisy_batch, multiplicative_noise_batch)
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
                    false_fn=lambda: noisy_batch
                )

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
            image_filenames_dataset_from_directory_gen(
                seed=0,
                directory=d,
                shuffle=True,
                follow_links=True)
            for d in directory
        ]
    else:
        raise ValueError("don't know how to handle non directory datasets")

    # --- save the augmentation functions
    result = dict()
    result[INPAINT_AUGMENTATION_FN_STR] = inpaint_augmentation_fn
    result[SUPERRES_AUGMENTATION_FN_STR] = superres_augmentation_fn
    result[GEOMETRIC_AUGMENTATION_FN_STR] = geometric_augmentations_fn

    if mix_noise_types:
        result[NOISE_AUGMENTATION_FN_STR] = noise_augmentations_mix_fn
    else:
        result[NOISE_AUGMENTATION_FN_STR] = noise_augmentations_fn

    # --- create the dataset
    # dataset produces the dataset with basic geometric distortions
    if len(dataset_training) == 0:
        raise ValueError("don't know how to handle zero datasets")

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

    def gen_fn():
        for x in merge_iterators(*dataset_training):
            yield x

    def load_image_fn(x):
        x = load_image_crop(
            path=x,
            image_size=None,
            num_channels=num_channels,
            interpolation=tf.image.ResizeMethod.BILINEAR,
            crop_size=(input_shape[0], input_shape[1]),
            x_range=None,
            y_range=None,
            no_crops_per_image=no_crops_per_image)
        return x

    result[DATASET_TRAINING_FN_STR] = \
        tf.data.Dataset \
            .from_generator(
            generator=gen_fn,
            output_signature=(
                tf.TensorSpec(
                    shape=(),
                    dtype=tf.string)
            )) \
            .map(
            map_func=load_image_fn,
            num_parallel_calls=12) \
            .unbatch() \
            .shuffle(
            seed=0,
            buffer_size=batch_size * len(directory) * no_crops_per_image,
            reshuffle_each_iteration=False) \
            .batch(
            batch_size=batch_size,
            deterministic=False,
            num_parallel_calls=tf.data.AUTOTUNE) \
            .prefetch(1)

    return result

# ---------------------------------------------------------------------
