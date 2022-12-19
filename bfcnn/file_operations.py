import os
import tensorflow as tf
from pathlib import Path
from typing import Tuple, Union, Any, Generator, List

# ---------------------------------------------------------------------

from .utilities import logger, random_crops, layer_normalize, merge_iterators

# ---------------------------------------------------------------------

SUPPORTED_IMAGE_LIST_FORMATS = (".bmp", ".gif", ".jpeg", ".jpg", ".png")


def image_filenames_dataset(
        directory: Union[str, List[str]],
        follow_links=False) -> Generator[str, None, None]:
    """
    Generates a `tf.data.Dataset`
    from image filenames in a directory or multiple directories.
    """

    if isinstance(directory, str):
        def gen_fn():
            return index_directory_gen(
                    directory=directory,
                    formats=SUPPORTED_IMAGE_LIST_FORMATS,
                    follow_links=follow_links)
    elif isinstance(directory, List):
        def gen_fn():
            iterators = [
                index_directory_gen(
                    directory=d,
                    formats=SUPPORTED_IMAGE_LIST_FORMATS,
                    follow_links=follow_links)
                for d in directory
            ]

            return merge_iterators(*iterators)
    else:
        raise ValueError(f"don't know what to do with [{directory}]")

    return \
        tf.data.Dataset.from_generator(
            generator=gen_fn,
            output_signature=(
                tf.TensorSpec(shape=(), dtype=tf.string)
            ))

# ---------------------------------------------------------------------


def index_directory_gen(
        directory,
        formats,
        follow_links=False) -> Generator[str, None, None]:
    """
    Make list of all files in the subdirs of `directory`, with their labels.

      Args:
        directory: The target directory (string).
        formats: Allowlist of file extensions to index (e.g. ".jpg", ".txt").
        follow_links: Whether to visits subdirectories pointed to by symlinks.
    """
    # in the no-label case, index from the parent directory down.
    sub_dirs = [""]

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
    for root, _, files in walk:
        for filename in files:
            if filename.lower().endswith(formats):
                yield root, filename


# ---------------------------------------------------------------------

@tf.function
def load_image_crop(
        path: Any,
        image_size: Tuple[int, int] = None,
        num_channels: int = 3,
        interpolation: tf.image.ResizeMethod = tf.image.ResizeMethod.BILINEAR,
        no_crops_per_image: int = 1,
        crop_size: Tuple[int, int] = (64, 64),
        x_range: Tuple[float, float] = None,
        y_range: Tuple[float, float] = None) -> tf.Tensor:
    """
    load an image from a path, resize it, crop it

    :param path: string
    :param image_size: resize dimensions
    :param num_channels: number of channels
    :param interpolation: interpolation type
    :param no_crops_per_image:
    :param crop_size:
    :param x_range:
    :param y_range:

    :return: tensor
    """
    # --- load image
    img = \
        load_image(
            path=path,
            image_size=image_size,
            num_channels=num_channels,
            interpolation=interpolation,
            expand_dims=True,
            normalize=False)

    img = \
        random_crops(
            input_batch=img,
            crop_size=crop_size,
            x_range=x_range,
            y_range=y_range,
            no_crops_per_image=no_crops_per_image)

    return img


# ---------------------------------------------------------------------

@tf.function
def load_image(
        path: Any,
        image_size: Tuple[int, int] = None,
        num_channels: int = 3,
        dtype: tf.DType = tf.uint8,
        interpolation: tf.image.ResizeMethod = tf.image.ResizeMethod.BILINEAR,
        expand_dims: bool = False,
        normalize: bool = False) -> tf.Tensor:
    """
    read image, decode it, and resize it

    :param path: string
    :param image_size: resize dimensions
    :param num_channels: number of channels
    :param dtype: tensor type
    :param interpolation: interpolation type
    :param expand_dims: if True add a batch dimension
    :param normalize: if True normalize to (-0.5, +0.5) (convert to float32)

    :return: tensor
    """
    # --- set variables
    if isinstance(path, Path):
        path = str(path)

    # --- read file, decode it
    raw = tf.io.read_file(filename=path)

    img = \
        tf.image.decode_image(
            dtype=dtype,
            contents=raw,
            channels=num_channels,
            expand_animations=False)

    # --- resize it
    if image_size is not None:
        img = \
            tf.image.resize_with_pad(
                image=img,
                target_height=image_size[0],
                target_width=image_size[1],
                method=interpolation,
                antialias=False)

    if expand_dims:
        img = tf.expand_dims(img, axis=0)

    if normalize:
        img = tf.cast(img, dtype=tf.float32)
        img = layer_normalize(img, v_min=0.0, v_max=255.0)

    return img

# ---------------------------------------------------------------------
