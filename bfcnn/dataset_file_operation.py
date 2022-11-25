import os
import numpy as np
import tensorflow as tf
import multiprocessing

# ---------------------------------------------------------------------

from .utilities import logger

# ---------------------------------------------------------------------

ALLOWLIST_FORMATS = (".bmp", ".gif", ".jpeg", ".jpg", ".png")


def image_dataset_from_directory(
        directory,
        color_mode="rgb",
        batch_size=32,
        image_size=(256, 256),
        shuffle=True,
        seed=None,
        interpolation=tf.image.ResizeMethod.BILINEAR,
        follow_links=False,
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

    image_paths = \
        index_directory(
            directory=directory,
            formats=ALLOWLIST_FORMATS,
            shuffle=shuffle,
            seed=seed,
            follow_links=follow_links,
        )

    dataset = tf.data.Dataset.from_tensor_slices(image_paths)
    args = (image_size, num_channels, interpolation)
    dataset = dataset.map(
        lambda x: load_image(x, *args), num_parallel_calls=tf.data.AUTOTUNE
    )
    dataset = dataset.batch(batch_size, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset

# ---------------------------------------------------------------------


def index_directory(
        directory,
        formats,
        shuffle=True,
        seed=None,
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

      Returns:
        tuple (file_paths, labels, class_names).
          file_paths: list of file paths (strings).
          labels: list of matching integer labels (same length as file_paths)
          class_names: names of the classes corresponding to these labels, in order.
    """
    # in the no-label case, index from the parent directory down.
    sub_dirs = ['']

    # Build an index of the files
    # in the different class sub-folders.
    pool = multiprocessing.pool.ThreadPool()
    results = []
    filenames = []

    for dir_path in (os.path.join(directory, subdir) for subdir in sub_dirs):
        results.append(
            pool.apply_async(index_subdirectory,
                             (dir_path, follow_links, formats)))

    for res in results:
        partial_filenames = res.get()
        filenames += partial_filenames

    logger.info('Found %d files.' % (len(filenames),))
    pool.close()
    pool.join()
    file_paths = [os.path.join(directory, filename) for filename in filenames]

    if shuffle:
        # Shuffle globally to erase macro-structure
        if seed is None:
            seed = np.random.randint(1e6)
        rng = np.random.RandomState(seed)
        rng.shuffle(file_paths)
    return file_paths


# ---------------------------------------------------------------------


def index_subdirectory(
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
    valid_files = iter_valid_files(directory, follow_links, formats)
    filenames = []
    for root, filename in valid_files:
        absolute_path = os.path.join(root, filename)
        relative_path = os.path.join(
            dirname, os.path.relpath(absolute_path, directory))
        filenames.append(relative_path)
    return filenames

# ---------------------------------------------------------------------


def image_dataset_from_directory_gen(
        directory,
        color_mode="rgb",
        batch_size=32,
        image_size=(256, 256),
        shuffle=True,
        seed=None,
        interpolation=tf.image.ResizeMethod.BILINEAR,
        follow_links=False,
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
        return load_image(x, image_size, num_channels, interpolation)

    dataset = \
        tf.data.Dataset.from_generator(
            generator=gen_fn,
            output_signature=(
                tf.TensorSpec(shape=(), dtype=tf.string)
            )) \
        .shuffle(
            buffer_size=1024,
            reshuffle_each_iteration=True)\
        .map(
            map_func=load_image_fn,
            num_parallel_calls=tf.data.AUTOTUNE)\
        .batch(
            batch_size=batch_size,
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
        for partial_filename in index_subdirectory(dir_path, follow_links, formats):
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
