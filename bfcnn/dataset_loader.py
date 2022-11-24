import os
import tensorflow as tf
from typing import Tuple
from keras.utils import dataset_utils
from keras.utils import image_utils

# ---------------------------------------------------------------------


def image_dataset_from_directory(
        directory,
        color_mode="rgb",
        image_size=(256, 256),
        interpolation="bilinear",
        follow_links=False,
        crop_to_aspect_ratio=False,
        random_crop: Tuple[int, int] = None,
        **kwargs,
):
    """Generates a `tf.data.Dataset` from image files in a directory"""
    if "smart_resize" in kwargs:
        crop_to_aspect_ratio = kwargs.pop("smart_resize")
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
    if random_crop is not None:
        random_crop = (random_crop[0], random_crop[1], num_channels)
    interpolation = image_utils.get_interpolation(interpolation)

    def generator_fn():
        for img_path in \
                index_directory(
                    directory=directory,
                    formats=(".bmp", ".gif", ".jpeg", ".jpg", ".png"),
                    follow_links=follow_links):
            yield \
                load_image(
                    path=img_path,
                    image_size=image_size,
                    num_channels=num_channels,
                    interpolation=interpolation,
                    crop_to_aspect_ratio=crop_to_aspect_ratio,
                    random_crop=random_crop)

    dataset = \
        tf.data.Dataset.from_generator(
            generator=generator_fn,
            output_signature=(
                tf.TensorSpec(
                    shape=(random_crop[0], random_crop[1], num_channels),
                    dtype=tf.uint8)
            ))

    return dataset


# ---------------------------------------------------------------------


def load_image(
        path,
        image_size,
        num_channels,
        interpolation,
        crop_to_aspect_ratio: bool = False,
        random_crop: Tuple[int, int, int] = None
):
    """Load an image from a path and resize it."""
    # img = tf.io.read_file(path)
    # img = tf.image.decode_image(
    #     img, channels=num_channels, expand_animations=False
    # )
    # if crop_to_aspect_ratio:
    #     img = image_utils.smart_resize(
    #         img, image_size, interpolation=interpolation
    #     )
    # else:
    # img = tf.image.resize(img, image_size, method=interpolation)
    # img.set_shape((image_size[0], image_size[1], num_channels))
    # img = tf.image.random_crop(img, size=random_crop)
    # return tf.cast(img, dtype=tf.uint8)
    img = tf.zeros(shape=(random_crop[0], random_crop[1], num_channels), dtype=tf.uint8)

# ---------------------------------------------------------------------


def index_directory(
        directory,
        formats,
        follow_links: bool = False):
    """Make list of all files in the subdirs of `directory`

    Args:
      directory: The target directory (string).
      formats: Allowlist of file extensions to index (e.g. ".jpg", ".txt").
      follow_links: Whether to visits subdirectories pointed to by symlinks.

    Returns:
      tuple (file_paths, labels, class_names).
        file_paths: list of file paths (strings).
        labels: list of matching integer labels (same length as file_paths)
        class_names: names of the classes corresponding to these labels, in
          order.
    """
    subdirs = []
    for subdir in sorted(tf.io.gfile.listdir(directory)):
        if tf.io.gfile.isdir(tf.io.gfile.join(directory, subdir)):
            if subdir.endswith("/"):
                subdir = subdir[:-1]
            subdirs.append(subdir)

    for dirpath in (tf.io.gfile.join(directory, subdir) for subdir in subdirs):
        for partial_filename in index_subdirectory(dirpath, follow_links, formats):
            yield tf.io.gfile.join(directory, partial_filename)


# ---------------------------------------------------------------------


def index_subdirectory(directory, follow_links, formats):
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
    valid_files = dataset_utils.iter_valid_files(directory, follow_links, formats)
    for root, fname in valid_files:
        absolute_path = tf.io.gfile.join(root, fname)
        relative_path = tf.io.gfile.join(
            dirname, os.path.relpath(absolute_path, directory)
        )
        yield relative_path

# ---------------------------------------------------------------------
