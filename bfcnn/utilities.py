import copy
import keras
import itertools
import numpy as np
from enum import Enum
from typing import List
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator


# ==============================================================================


def collage(images_batch):
    """
    Create a collage of image from a batch

    :param images_batch:
    :return:
    """
    shape = images_batch.shape
    no_images = shape[0]
    images = []
    result = None
    width = np.ceil(np.sqrt(no_images))

    for i in range(no_images):
        images.append(images_batch[i, :, :, :])

        if len(images) % width == 0:
            if result is None:
                result = np.hstack(images)
            else:
                tmp = np.hstack(images)
                result = np.vstack([result, tmp])
            images.clear()
    return result


# ==============================================================================


def merge_iterators(*iterators):
    """
    Merge different iterators together
    """
    empty = {}
    for values in itertools.zip_longest(*iterators, fillvalue=empty):
        for value in values:
            if value is not empty:
                yield value

# ==============================================================================


def layer_denormalize(args):
    """
    Convert input [-1, +1] to [v0, v1] range
    """
    y, v0, v1 = args
    y_clip = K.clip(y, min_value=-1.0, max_value=+1.0)
    return 0.5 * (y_clip + 1.0) * (v1 - v0) + v0

# ==============================================================================


def layer_normalize(args):
    """
    Convert input from [v0, v1] to [-1, +1] range
    """
    y, v0, v1 = args
    y_clip = K.clip(y, min_value=v0, max_value=v1)
    return 2.0 * (y_clip - v0) / (v1 - v0) - 1.0


# ==============================================================================


def build_normalize_model(
        input_dims,
        min_value: float = 0.0,
        max_value: float = 255.0,
        name: str = "normalize") -> keras.Model:
    """
    Wrap a normalize layer in a model

    :param input_dims: Models input dimensions
    :param min_value: Minimum value
    :param max_value: Maximum value
    :param name: name of the model
    :return:
    """
    model_input = keras.Input(shape=input_dims)
    # --- normalize input from [min_value, max_value] to [-1.0, +1.0]
    model_output = keras.layers.Lambda(layer_normalize, trainable=False)([
        model_input, float(min_value), float(max_value)])
    # --- wrap model
    return keras.Model(
        name=name,
        inputs=model_input,
        outputs=model_output,
        trainable=False)

# ==============================================================================


def build_denormalize_model(
        input_dims,
        min_value: float = 0.0,
        max_value: float = 255.0,
        name: str = "denormalize") -> keras.Model:
    """
    Wrap a denormalize layer in a model

    :param input_dims: Models input dimensions
    :param min_value: Minimum value
    :param max_value: Maximum value
    :param name: name of the model
    :return:
    """
    model_input = keras.Input(shape=input_dims)
    # --- normalize input from [min_value, max_value] to [-1.0, +1.0]
    model_output = \
        keras.layers.Lambda(layer_denormalize, trainable=False)([
            model_input, float(min_value), float(max_value)])
    # --- wrap model
    return keras.Model(
        name=name,
        inputs=model_input,
        outputs=model_output,
        trainable=False)


# ==============================================================================


def build_resnet_model(
        input_dims,
        no_layers: int,
        kernel_size: int,
        filters: int,
        channel_index: int = 2,
        use_bn: bool = True,
        kernel_regularizer=None,
        kernel_initializer=None,
        name="resnet") -> keras.Model:
    """
    Build a resnet model

    :param input_dims: Models input dimensions
    :param no_layers: Number of resnet layers
    :param kernel_size: kernel size of the conv layers
    :param filters: number of filters per convolutional layer
    :param channel_index: Index of the channel in dimensions
    :param use_bn: Use Batch Normalization
    :param kernel_regularizer: Kernel weight regularizer
    :param kernel_initializer: Kernel weight initializer
    :param name: Name of the model
    :return:
    """
    # --- variables
    bn_params = dict(
        center=False,
        scale=True,
        momentum=0.999,
        epsilon=1e-4
    )
    conv_params = dict(
        filters=filters,
        kernel_size=kernel_size,
        use_bias=False,
        strides=(1, 1),
        padding="same",
        activation="linear",
        kernel_regularizer=kernel_regularizer,
        kernel_initializer=kernel_initializer
    )
    intermediate_conv_params = copy.deepcopy(conv_params)
    intermediate_conv_params["kernel_size"] = 1
    final_conv_params = copy.deepcopy(conv_params)
    final_conv_params["filters"] = input_dims[channel_index]
    final_conv_params["kernel_size"] = 1

    # --- set input
    model_input = keras.Input(shape=input_dims)

    # --- add base layer
    x = keras.layers.Conv2D(**conv_params)(model_input)
    x = keras.layers.ReLU()(x)
    if use_bn:
        x = keras.layers.BatchNormalization(**bn_params)(x)

    # --- add resnet layers
    for i in range(no_layers):
        previous_layer = x
        x = keras.layers.Conv2D(**conv_params)(x)
        x = keras.layers.ReLU()(x)
        if use_bn:
            x = keras.layers.BatchNormalization(**bn_params)(x)
        x = keras.layers.Conv2D(**intermediate_conv_params)(x)
        x = keras.layers.ReLU()(x)
        if use_bn:
            x = keras.layers.BatchNormalization(**bn_params)(x)
        x = keras.layers.Add()([previous_layer, x])

    # --- output to original channels
    output_layer = keras.layers.Conv2D(**final_conv_params)(x)

    return keras.Model(
        name=name,
        inputs=model_input,
        outputs=output_layer)


# ==============================================================================


def noisy_image_data_generator(
        dataset,
        batch_size: int = 32,
        min_value: float = 0.0,
        max_value: float = 255.0,
        min_noise_std: float = 0.01,
        max_noise_std: float = 10.0,
        random_invert: bool = False,
        random_brightness: bool = False,
        zoom_range: float = 0.25,
        rotation_range: int = 90,
        width_shift_range: float = 0.25,
        height_shift_range: float = 0.25,
        vertical_flip: bool = True,
        horizontal_flip: bool = True):
    """
    Create a dataset generator flow that adds noise to a dateset

    :param dataset:
    :param min_value: Minimum allowed value
    :param max_value: Maximum allowed value
    :param batch_size: Batch size
    :param random_invert: Randomly (50%) invert the image
    :param random_brightness: Randomly add offset or multiplier
    :param zoom_range: Randomly zoom in (percentage)
    :param rotation_range: Add random rotation range (in degrees)
    :param min_noise_std: Min standard deviation of noise
    :param max_noise_std: Max standard deviation of noise
    :param horizontal_flip: Randomly horizontally flip image
    :param vertical_flip: Randomly vertically flip image
    :param height_shift_range: Add random vertical shift (percentage of image)
    :param width_shift_range: Add random horizontal shift (percentage of image)

    :return:
    """
    # --- argument checking
    if dataset is None:
        raise ValueError("dataset cannot be empty")
    if min_noise_std > max_noise_std:
        raise ValueError("min_noise_std must be < max_noise_std")
    if min_value > max_value:
        raise ValueError("min_value must be < max_value")

    # --- variables setup
    max_min_diff = (max_value - min_value)

    # --- create data generator
    if isinstance(dataset, np.ndarray):
        data_generator = \
            ImageDataGenerator(
                zoom_range=zoom_range,
                rotation_range=rotation_range,
                width_shift_range=width_shift_range,
                height_shift_range=height_shift_range,
                vertical_flip=vertical_flip,
                horizontal_flip=horizontal_flip,
                zca_whitening=False,
                featurewise_center=False,
                featurewise_std_normalization=False)
    else:
        raise NotImplementedError()

    # --- iterate over random batches
    for x_batch in \
            data_generator.flow(
                x=dataset,
                shuffle=True,
                batch_size=batch_size):
        # randomly invert batch
        if random_invert:
            if np.random.choice([False, True]):
                x_batch = (max_value - x_batch) + min_value

        # add random offset
        if random_brightness:
            offset = \
                np.random.uniform(
                    low=0.0,
                    high=0.25 * max_min_diff)
            x_batch = x_batch + offset

        # adjust the std of the noise
        # pick std between min and max std
        if np.random.choice([False, True]):
            std = \
                np.random.uniform(
                    low=min_noise_std,
                    high=max_noise_std)

            # add noise to create the noisy input
            x_batch_noisy = \
                x_batch + \
                np.random.normal(0.0, std, x_batch.shape)
        else:
            x_batch_noisy = np.copy(x_batch)

        # clip all to be between min and max value
        # return input, target
        yield np.clip(x_batch_noisy, a_min=min_value, a_max=max_value), \
              np.clip(x_batch, a_min=min_value, a_max=max_value)


# ==============================================================================


def get_conv2d_weights(
        model: keras.Model) -> np.ndarray:
    """
    Get the conv2d weights from the model concatenated
    """
    weights = []
    for layer in model.layers:
        layer_config = layer.get_config()
        if "layers" not in layer_config:
            continue
        layer_weights = layer.get_weights()
        for i, l in enumerate(layer_config["layers"]):
            if l["class_name"] == "Conv2D":
                for w in layer_weights[i]:
                    w_flat = w.flatten()
                    weights.append(w_flat)
    return np.concatenate(weights)

# ==============================================================================
