import os
import glob
import itertools
import tensorflow as tf
from pathlib import Path
from typing import Tuple, Union, Any, Generator, List

# ---------------------------------------------------------------------

from .utilities import logger, layer_normalize

# ---------------------------------------------------------------------

SUPPORTED_IMAGE_LIST_FORMATS = (".bmp", ".gif", ".jpeg", ".jpg", ".png")


# ---------------------------------------------------------------------


def merge_iterators(
        *iterators):
    """
    Merge different iterators together

    :param iterators:
    """
    empty = {}
    for values in itertools.zip_longest(*iterators, fillvalue=empty):
        for value in values:
            if value is not empty:
                yield value

# ---------------------------------------------------------------------


def image_filenames_generator(
        directory: Union[str, List[str]]) -> Generator[str, None, None]:
    """
    creates a generator function
    """

    if isinstance(directory, str):
        def gen_fn() -> Generator[str, None, None]:
            return index_directory_gen(
                    directory=directory,
                    formats=SUPPORTED_IMAGE_LIST_FORMATS)
    elif isinstance(directory, List):
        def gen_fn() -> Generator[str, None, None]:
            iterators = [
                index_directory_gen(
                    directory=d,
                    formats=SUPPORTED_IMAGE_LIST_FORMATS)
                for d in directory
            ]

            return merge_iterators(*iterators)
    else:
        raise ValueError(f"don't know what to do with [{directory}]")

    return gen_fn


# ---------------------------------------------------------------------


def index_directory_gen(
        directory: str,
        formats: Tuple = SUPPORTED_IMAGE_LIST_FORMATS) -> Generator[str, None, None]:
    """
    get all image files
    """
    for filename in glob.iglob(os.path.join(directory, "**/*"), recursive=True):
        if filename.lower().endswith(formats):
            yield filename

# ---------------------------------------------------------------------


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

    # --- fix arguments
    if isinstance(path, Path):
        path = str(path)

    # --- read file, decode it
    raw = tf.io.read_file(filename=path)

    img = \
        tf.image.decode_image(
            dtype=tf.uint8,
            contents=raw,
            channels=num_channels,
            expand_animations=False)
    del raw

    # --- resize it
    if image_size is not None:
        img_pad = \
            tf.image.resize_with_pad(
                image=img,
                target_height=image_size[0],
                target_width=image_size[1],
                method=interpolation,
                antialias=False)
        del img
        img = img_pad

    if expand_dims:
        img = tf.expand_dims(img, axis=0)

    if normalize:
        img = tf.cast(img, dtype=tf.float32)
        img = layer_normalize(img, v_min=0.0, v_max=255.0)
    else:
        img = tf.cast(img, dtype=dtype)

    return img

# ---------------------------------------------------------------------
