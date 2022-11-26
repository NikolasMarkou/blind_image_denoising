import os
import numpy as np
import tensorflow as tf
from typing import Tuple

# ---------------------------------------------------------------------

from .utilities import logger

# ---------------------------------------------------------------------

ALLOWLIST_FORMATS = (".bmp", ".gif", ".jpeg", ".jpg", ".png")

# ---------------------------------------------------------------------


def image_dataset_from_directory_gen(
        directory,
        color_mode="rgb",
        image_size=(256, 256),
        shuffle=True,
        seed=None,
        interpolation=tf.image.ResizeMethod.BILINEAR,
        follow_links=False,
        no_crops_per_image: int = 1,
        crop_size: Tuple[int, int] = (64, 64),
        **kwargs):
    """Generates a `tf.data.Dataset` from image files in a directory.

    Animated gifs are truncated to the first frame.
    Args:
      directory: Directory where the data is located.
      color_mode: One of "grayscale", "rgb", "rgba". Default: "rgb".
          Whether the images will be converted to
          have 1, 3, or 4 channels.
      batch_size: Size of the batches of data. Default: 32.
        If `None`, the data will not be batched
        (the dataset will yield individual samples).
      image_size: Size to resize images to after they are read from disk,
          specified as `(height, width)`. Defaults to `(256, 256)`.
          Since the pipeline processes batches of images that must all have
          the same size, this must be provided.
      shuffle: Whether to shuffle the data. Default: True.
          If set to False, sorts the data in alphanumeric order.
      seed: Optional random seed for shuffling and transformations.
      interpolation: String, the interpolation method used when resizing images.
        Defaults to `bilinear`. Supports `bilinear`, `nearest`, `bicubic`,
        `area`, `lanczos3`, `lanczos5`, `gaussian`, `mitchellcubic`.
      follow_links: Whether to visit subdirectories pointed to by symlinks.
          Defaults to False.
      **kwargs: Legacy keyword arguments.
    Returns:
      A `tf.data.Dataset` object.
        - `(batch_size, image_size[0], image_size[1], num_channels)`,
    Rules regarding number of channels in the yielded images:
      - if `color_mode` is `grayscale`,
        there's 1 channel in the image tensors.
      - if `color_mode` is `rgb`,
        there are 3 channels in the image tensors.
      - if `color_mode` is `rgba`,
        there are 4 channels in the image tensors.
    """
    if kwargs:
        raise TypeError(f"Unknown keywords argument(s): {tuple(kwargs.keys())}")

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

    if seed is None:
        seed = np.random.randint(1e6)

    def gen_fn():
        for x in index_directory_gen(
                directory=directory,
                formats=ALLOWLIST_FORMATS,
                follow_links=follow_links):
            yield x

    def load_image_fn(x):
        return load_image_crop(
            x, image_size, num_channels, interpolation,
            crop_size=crop_size, no_crops_per_image=no_crops_per_image)

    dataset = \
        tf.data.Dataset.from_generator(
            generator=gen_fn,
            output_signature=(
                tf.TensorSpec(shape=(), dtype=tf.string)
            ))

    if shuffle:
        dataset = dataset.shuffle(
            seed=seed,
            buffer_size=1024,
            reshuffle_each_iteration=True)

    dataset = \
        dataset\
            .map(
                map_func=load_image_fn,
                num_parallel_calls=tf.data.AUTOTUNE) \
            .prefetch(tf.data.AUTOTUNE)

    return dataset


# ---------------------------------------------------------------------


def index_directory_gen(
        directory,
        formats,
        follow_links=False):
    """
    Make list of all files in the subdirs of `directory`, with their labels.

      Args:
        directory: The target directory (string).
        formats: Allowlist of file extensions to index (e.g. ".jpg", ".txt").
        shuffle: Whether to shuffle the data. Default: True.
            If set to False, sorts the data in alphanumeric order.
        seed: Optional random seed for shuffling.
        follow_links: Whether to visits subdirectories pointed to by symlinks.
    """
    # in the no-label case, index from the parent directory down.
    sub_dirs = ['']

    # Build an index of the files
    # in the different class sub-folders.
    for dir_path in (os.path.join(directory, subdir) for subdir in sub_dirs):
        for partial_filename in index_subdirectory_gen(dir_path, follow_links, formats):
            yield os.path.join(directory, partial_filename)


# ---------------------------------------------------------------------


def index_subdirectory_gen(
        directory,
        follow_links,
        formats):
    """Recursively walks directory and list image paths and their class index.

    Args:
        directory: string, target directory.
        follow_links: boolean, whether to recursively follow subdirectories
          (if False, we only list top-level images in `directory`).
        formats: Allowlist of file extensions to index (e.g. ".jpg", ".txt").

    Returns:
        tuple `(filenames, labels)`. `filenames` is a list of relative file
        paths, and `labels` is a list of integer labels corresponding to these
        files.
    """
    dirname = os.path.basename(directory)
    for root, filename in iter_valid_files(directory, follow_links, formats):
        absolute_path = os.path.join(root, filename)
        relative_path = os.path.join(
            dirname, os.path.relpath(absolute_path, directory))
        yield relative_path


# ---------------------------------------------------------------------


def iter_valid_files(directory, follow_links, formats):
    walk = os.walk(directory, followlinks=follow_links)
    for root, _, files in sorted(walk, key=lambda x: x[0]):
        for filename in sorted(files):
            if filename.lower().endswith(formats):
                yield root, filename

# ---------------------------------------------------------------------


def load_image_crop(
        path,
        image_size,
        num_channels,
        interpolation,
        no_crops_per_image: int = 1,
        crop_size: Tuple[int, int] = (64, 64)):
    """
    Load an image from a path
    and resize it
    and crop it
    """
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img,
                                channels=num_channels,
                                expand_animations=False)

    if image_size is not None:
        img = tf.image.resize_with_pad(
            image=img,
            target_height=image_size[0],
            target_width=image_size[1],
            method=interpolation,
            antialias=False
        )

    img = tf.cast(img, dtype=tf.uint8)
    img = tf.expand_dims(img, axis=0)

    if no_crops_per_image > 0:
        img =  \
            random_crops(
                input_batch=img,
                crop_size=crop_size,
                no_crops_per_image=no_crops_per_image)
    return img

# ---------------------------------------------------------------------


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
