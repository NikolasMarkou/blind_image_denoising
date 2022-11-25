import numpy as np
import tensorflow as tf
from keras.preprocessing import dataset_utils

ALLOWLIST_FORMATS = (".bmp", ".gif", ".jpeg", ".jpg", ".png")


def image_dataset_from_directory(
        directory,
        color_mode="rgb",
        batch_size=32,
        image_size=(256, 256),
        shuffle=True,
        seed=None,
        interpolation=tf.ResizeMethod.BILINEAR,
        follow_links=False,
        crop_to_aspect_ratio=False,
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
      crop_to_aspect_ratio: If True, resize the images without aspect
        ratio distortion. When the original aspect ratio differs from the target
        aspect ratio, the output image will be cropped so as to return the
        largest possible window in the image (of size `image_size`) that matches
        the target aspect ratio. By default (`crop_to_aspect_ratio=False`),
        aspect ratio may not be preserved.
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

    if seed is None:
        seed = np.random.randint(1e6)

    image_paths, _, _ = \
        dataset_utils.index_directory(
            directory=directory,
            labels=None,
            formats=ALLOWLIST_FORMATS,
            class_names=None,
            shuffle=shuffle,
            seed=seed,
            follow_links=follow_links,
        )

    dataset = tf.data.Dataset.from_tensor_slices(image_paths)
    args = (image_size, num_channels, interpolation, crop_to_aspect_ratio)
    dataset = dataset.map(
        lambda x: load_image(x, *args), num_parallel_calls=tf.data.AUTOTUNE
    )
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


# ---------------------------------------------------------------------


def load_image(
        path,
        image_size,
        num_channels,
        interpolation):
    """Load an image from a path and resize it."""
    img = tf.io.read_file(path)
    img = tf.image.decode_image(
        img, channels=num_channels, expand_animations=False
    )

    img = tf.image.resize_with_pad(
        image=img,
        target_height=image_size[0],
        target_width=image_size[1],
        method=interpolation,
        antialias=False
    )

    return img

# ---------------------------------------------------------------------
