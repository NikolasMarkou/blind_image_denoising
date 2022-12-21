import os
import glob
import tensorflow as tf
from pathlib import Path
from typing import Tuple, Union, Any, Generator, List

# ---------------------------------------------------------------------

from .utilities import logger, layer_normalize, merge_iterators

# ---------------------------------------------------------------------

SUPPORTED_IMAGE_LIST_FORMATS = (".bmp", ".gif", ".jpeg", ".jpg", ".png")


def image_filenames_generator(
        directory: Union[str, List[str]],
        follow_links=False) -> Generator[str, None, None]:
    """
    creates a generator function
    """

    if isinstance(directory, str):
        def gen_fn() -> Generator[str, None, None]:
            return index_directory_gen(
                    directory=directory,
                    formats=SUPPORTED_IMAGE_LIST_FORMATS,
                    follow_links=follow_links)
    elif isinstance(directory, List):
        def gen_fn() -> Generator[str, None, None]:
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

    return gen_fn


# ---------------------------------------------------------------------


def index_directory_gen(
        directory: str,
        formats: List[str]) -> Generator[str, None, None]:
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
    # --- read file, decode it
    raw = tf.io.read_file(filename=path)

    img = \
        tf.image.decode_image(
            dtype=dtype,
            contents=raw,
            channels=num_channels,
            expand_animations=False)
    del raw

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
