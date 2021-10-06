import copy
import keras
import itertools
import numpy as np
import tensorflow as tf
from typing import List, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .custom_logger import logger


# ---------------------------------------------------------------------


def normal_empirical_cdf(
        target_cdf: float = 0.5,
        mean: float = 0.0,
        sigma: float = 1.0,
        samples: int = 1000000,
        bins: int = 1000):
    """
    computes the value x for target_cdf
    """
    # --- argument checking
    if target_cdf <= 0.0001 or target_cdf >= 0.9999:
        raise ValueError(
            "target_cdf [{0}] must be between 0 and 1".format(target_cdf))
    if sigma <= 0:
        raise ValueError("sigma [{0}] must be > 0".format(sigma))

    # --- computer empirical cumulative sum
    z = \
        np.random.normal(
            loc=mean,
            scale=sigma,
            size=samples)
    h, x1 = np.histogram(z, bins=bins, density=True)
    dx = x1[1] - x1[0]
    f1 = np.cumsum(h) * dx

    # --- find the proper bin
    for i in range(bins):
        if f1[i] >= target_cdf:
            if i == 0:
                return x1[i]
            return x1[i]

    return -1


# ---------------------------------------------------------------------


def coords_layer(
        input_layer):
    """
    Create a coords layer

    :param input_layer:
    :return:
    """
    # --- argument checking
    if input_layer is None:
        raise ValueError("input_layer cannot be empty")
    shape = keras.backend.int_shape(input_layer)
    if len(shape) != 4:
        raise ValueError("input_layer must be a 4d tensor")
    # ---
    height = shape[1]
    width = shape[2]
    x_grid = np.linspace(0, 1, width)
    y_grid = np.linspace(0, 1, height)
    xx_grid, yy_grid = np.meshgrid(x_grid, y_grid)
    xx_grid = \
        tf.constant(
            value=xx_grid,
            dtype=tf.float32,
            shape=(1, xx_grid.shape[0], xx_grid.shape[1], 1))
    yy_grid = \
        tf.constant(
            value=yy_grid,
            dtype=tf.float32,
            shape=(1, yy_grid.shape[0], yy_grid.shape[1], 1))
    xx_grid = tf.repeat(xx_grid, axis=0, repeats=tf.shape(input_layer)[0])
    yy_grid = tf.repeat(yy_grid, axis=0, repeats=tf.shape(input_layer)[0])
    return keras.layers.Concatenate(axis=3)([input_layer, yy_grid, xx_grid])

# ---------------------------------------------------------------------


def mean_sigma(
        input_layer,
        kernel_size: Tuple[int, int] = (5, 5)):
    """
    Create a mean sigma

    :param input_layer:
    :param kernel_size:
    :return:
    """
    # --- argument checking
    if input_layer is None:
        raise ValueError("input_layer cannot be empty")
    shape = keras.backend.int_shape(input_layer)
    if len(shape) != 4:
        raise ValueError("input_layer must be a 4d tensor")
    # ---
    avg = \
        keras.layers.AveragePooling2D(
            strides=(1, 1),
            padding="SAME",
            pool_size=kernel_size)(input_layer)
    diff_2 = tf.square(input_layer - avg)
    variance = \
        keras.layers.AveragePooling2D(
            strides=(1, 1),
            padding="SAME",
            pool_size=kernel_size)(diff_2)
    sigma = tf.sqrt(variance)
    return avg, sigma

# ---------------------------------------------------------------------


def sparse_block(
        input_layer,
        threshold_sigma: float = 1.0,
        negative_slope: float = 0.0,
        max_value: float = None,
        symmetric: bool = True,
        axis: List[int] = [-1]):
    """
    Create sparsity in an input layer

    :param input_layer:
    :param threshold_sigma: sparsity of the results
    :param negative_slope: slope of values below the threshold (0 cuts them off)
    :param max_value: max allowed value
    :param symmetric: if true allow negative values else zero them off
    :param axis: axis to perform batch norm (default is channels)

    :return: sparse results
    """
    # --- argument checking
    if threshold_sigma < 0:
        raise ValueError("threshold_sigma must be >= 0")
    if max_value is not None and max_value < 0:
        raise ValueError("max_value must be >= 0")
    if max_value is not None and threshold_sigma >= max_value:
        raise ValueError("threshold_sigma must be < max_value")

    # --- setup parameters
    # batchnorm parameters
    bn_params = dict(
        # must be true so it is always centered (mean 0)
        center=True,
        # must be true so variance is always 1 (sigma 1)
        scale=True,
        axis=axis,
        epsilon=1e-4,
        momentum=0.999
    )
    # relu params
    relu_params = dict(
        max_value=max_value,
        threshold=threshold_sigma,
        negative_slope=negative_slope
    )

    # --- computation is batchnorm - relu with custom threshold
    x_bn = keras.layers.BatchNormalization(**bn_params)(input_layer)

    if symmetric:
        x_abs = keras.backend.abs(x_bn)
        x_sign = keras.backend.sign(x_bn)
        x_relu = keras.layers.ReLU(**relu_params)(x_abs)
        x_result = \
            keras.layers.Multiply()([
                x_relu,
                x_sign,
            ])
    else:
        x_result = keras.layers.ReLU(**relu_params)(x_bn)

    return x_result


# ---------------------------------------------------------------------


def conv2d_sparse(
        input_layer,
        groups: int = 1,
        filters: int = 32,
        padding: str = "same",
        strides: Tuple[int, int] = (1, 1),
        kernel_size: Tuple[int, int] = (3, 3),
        kernel_regularizer: str = "l1",
        kernel_initializer: str = "glorot_normal",
        threshold_sigma: float = 1.0,
        negative_slope: float = 0.0,
        max_value: float = None,
        symmetric: bool = True,
        axis: List[int] = [-1]):
    """
    Create a conv2d layer that is always sparse by %x percent
    This works by applying relu with a given threshold calculated
    by the normal distribution (Cumulative distribution function)
    that is forced by the batch normalization

    :param input_layer:
    :param threshold_sigma: sparsity of the results
    :param negative_slope: slope of values below the threshold (0 cuts them off)
    :param max_value: max allowed value
    :param symmetric: if true allow negative values else zero them off
    :param axis: axis to perform batch norm (default is channels)
    :param groups: groups to perform convolution in
    :param filters:
    :param padding:
    :param strides:
    :param kernel_size:
    :param kernel_regularizer:
    :param kernel_initializer:
    :return: sparse convolution results
    """
    # --- setup parameters
    # convolution parameters
    conv2d_params = dict(
        groups=groups,
        use_bias=False,
        filters=filters,
        padding=padding,
        strides=strides,
        activation="linear",
        kernel_size=kernel_size,
        kernel_regularizer=kernel_regularizer,
        kernel_initializer=kernel_initializer
    )

    # --- run convolution and then sparse it
    x_conv = keras.layers.Conv2D(**conv2d_params)(input_layer)

    return \
        sparse_block(
            input_layer=x_conv,
            threshold_sigma=threshold_sigma,
            negative_slope=negative_slope,
            symmetric=symmetric,
            max_value=max_value,
            axis=axis)


# ---------------------------------------------------------------------


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


# ---------------------------------------------------------------------


def merge_iterators(*iterators):
    """
    Merge different iterators together
    """
    empty = {}
    for values in itertools.zip_longest(*iterators, fillvalue=empty):
        for value in values:
            if value is not empty:
                yield value


# ---------------------------------------------------------------------


def layer_denormalize(args):
    """
    Convert input [-0.5, +0.5] to [v0, v1] range
    """
    y, v_min, v_max = args
    y_clip = tf.clip_by_value(y, clip_value_min=-0.5, clip_value_max=0.5)
    return (y_clip + 0.5) * (v_max - v_min) + v_min


# ---------------------------------------------------------------------


def layer_normalize(args):
    """
    Convert input from [v0, v1] to [-0.5, +0.5] range
    """
    y, v_min, v_max = args
    y_clip = tf.clip_by_value(y, clip_value_min=v_min, clip_value_max=v_max)
    return (y_clip - v_min) / (v_max - v_min) - 0.5


# ---------------------------------------------------------------------


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

    # --- normalize input from [min_value, max_value] to [-0.5, +0.5]
    model_output = \
        keras.layers.Lambda(layer_normalize, trainable=False)([
            model_input, float(min_value), float(max_value)])

    # --- wrap model
    return keras.Model(
        name=name,
        inputs=model_input,
        outputs=model_output,
        trainable=False)


# ---------------------------------------------------------------------


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

    # --- normalize input from [min_value, max_value] to [-0.5, +0.5]
    model_output = \
        keras.layers.Lambda(layer_denormalize, trainable=False)([
            model_input, float(min_value), float(max_value)])

    # --- wrap model
    return keras.Model(
        name=name,
        inputs=model_input,
        outputs=model_output,
        trainable=False)


# ---------------------------------------------------------------------


def build_resnet_model(
        input_dims,
        no_layers: int,
        kernel_size: int,
        filters: int,
        activation: str = "relu",
        final_activation: str = "linear",
        use_bn: bool = True,
        use_bias: bool = False,
        kernel_regularizer="l1",
        kernel_initializer="glorot_normal",
        channel_index: int = 2,
        name="resnet") -> keras.Model:
    """
    Build a resnet model

    :param input_dims: Models input dimensions
    :param no_layers: Number of resnet layers
    :param kernel_size: kernel size of the conv layers
    :param filters: number of filters per convolutional layer
    :param activation: intermediate activation
    :param final_activation: activation of the final layer
    :param channel_index: Index of the channel in dimensions
    :param use_bn: Use Batch Normalization
    :param use_bias: use bias
    :param kernel_regularizer: Kernel weight regularizer
    :param kernel_initializer: Kernel weight initializer
    :param name: Name of the model
    :return:
    """
    # --- variables
    bn_params = dict(
        center=use_bias,
        scale=True,
        momentum=0.999,
        epsilon=1e-4
    )
    conv_params = dict(
        filters=filters,
        strides=(1, 1),
        padding="same",
        use_bias=use_bias,
        activation=activation,
        kernel_size=kernel_size,
        kernel_regularizer=kernel_regularizer,
        kernel_initializer=kernel_initializer
    )
    depth_conv_params = dict(
        depth_multiplier=2,
        strides=(1, 1),
        padding="same",
        use_bias=use_bias,
        activation=activation,
        kernel_size=kernel_size,
        depthwise_regularizer=kernel_regularizer,
        depthwise_initializer=kernel_initializer
    )
    # intermediate conv
    intermediate_conv_params = copy.deepcopy(conv_params)
    intermediate_conv_params["kernel_size"] = 1

    # final conv
    final_conv_params = copy.deepcopy(conv_params)
    final_conv_params["kernel_size"] = 1
    final_conv_params["activation"] = final_activation
    final_conv_params["filters"] = input_dims[channel_index]
    del final_conv_params["kernel_regularizer"]

    # --- set input
    model_input = keras.Input(shape=input_dims)

    # --- add base layer
    x = keras.layers.Conv2D(**conv_params)(model_input)
    if use_bn:
        x = keras.layers.BatchNormalization(**bn_params)(x)

    # --- add resnet layers
    for i in range(no_layers):
        previous_layer = x
        x = keras.layers.DepthwiseConv2D(**depth_conv_params)(x)
        if use_bn:
            x = keras.layers.BatchNormalization(**bn_params)(x)
        x = keras.layers.Conv2D(**intermediate_conv_params)(x)
        if use_bn:
            x = keras.layers.BatchNormalization(**bn_params)(x)
        x = keras.layers.Add()([previous_layer, x])

    # --- output to original channels
    output_layer = \
        keras.layers.Conv2D(**final_conv_params)(x)

    return keras.Model(
        name=name,
        inputs=model_input,
        outputs=output_layer)


# ---------------------------------------------------------------------


def build_sparse_resnet_model(
        input_dims,
        no_layers: int,
        kernel_size: int,
        filters: int,
        final_activation: str = "linear",
        kernel_regularizer="l1",
        kernel_initializer="glorot_normal",
        channel_index: int = 2,
        name="sparse_resnet") -> keras.Model:
    """
    Build a sparse resnet model

    :param input_dims: Models input dimensions
    :param no_layers: Number of resnet layers
    :param kernel_size: kernel size of the conv layers
    :param filters: number of filters per convolutional layer
    :param final_activation: activation of the final layer
    :param channel_index: Index of the channel in dimensions
    :param kernel_regularizer: Kernel weight regularizer
    :param kernel_initializer: Kernel weight initializer
    :param name: Name of the model
    :return:
    """
    # --- variables
    base_conv_params = dict(
        filters=filters,
        padding="same",
        use_bias=False,
        activation="relu",
        kernel_size=5,
        kernel_regularizer=kernel_regularizer,
        kernel_initializer=kernel_initializer
    )
    conv_params = dict(
        filters=filters,
        padding="same",
        use_bias=False,
        activation="relu",
        kernel_size=1,
        kernel_regularizer=kernel_regularizer,
        kernel_initializer=kernel_initializer
    )
    sparse_conv_params = dict(
        threshold_sigma=1.0,
        negative_slope=0.0,
        max_value=None,
        filters=filters,
        padding="same",
        symmetric=False,
        kernel_size=kernel_size,
        kernel_regularizer=kernel_regularizer,
        kernel_initializer=kernel_initializer
    )
    final_conv_params = dict(
        kernel_size=1,
        padding="same",
        strides=(1, 1),
        activation=final_activation,
        filters=input_dims[channel_index],
        kernel_regularizer=kernel_regularizer,
        kernel_initializer=kernel_initializer
    )

    # --- set input
    model_input = keras.Input(shape=input_dims)

    # --- add base layer
    x = keras.layers.Conv2D(**base_conv_params)(model_input)

    # --- add resnet layers
    for i in range(no_layers):
        previous_layer = x
        x = conv2d_sparse(x, **sparse_conv_params)
        x = keras.layers.Conv2D(**conv_params)(x)
        x = keras.layers.Add()([previous_layer, x])

    # --- output to original channels
    output_layer = \
        keras.layers.Conv2D(**final_conv_params)(x)

    return keras.Model(
        name=name,
        inputs=model_input,
        outputs=output_layer)

# ---------------------------------------------------------------------


def build_sparse_resnet_mean_sigma_model(
        input_dims,
        no_layers: int,
        kernel_size: int,
        filters: int,
        final_activation: str = "linear",
        kernel_regularizer="l1",
        kernel_initializer="glorot_normal",
        channel_index: int = 2,
        name="sparse_resnet") -> keras.Model:
    """
    Build a mean variance sparse resnet model

    :param input_dims: Models input dimensions
    :param no_layers: Number of resnet layers
    :param kernel_size: kernel size of the conv layers
    :param filters: number of filters per convolutional layer
    :param final_activation: activation of the final layer
    :param channel_index: Index of the channel in dimensions
    :param kernel_regularizer: Kernel weight regularizer
    :param kernel_initializer: Kernel weight initializer
    :param name: Name of the model
    :return:
    """
    # --- variables
    base_conv_params = dict(
        filters=filters,
        padding="same",
        use_bias=False,
        activation="relu",
        kernel_size=5,
        kernel_regularizer=kernel_regularizer,
        kernel_initializer=kernel_initializer
    )
    conv_params = dict(
        filters=filters,
        padding="same",
        use_bias=False,
        activation="relu",
        kernel_size=1,
        kernel_regularizer=kernel_regularizer,
        kernel_initializer=kernel_initializer
    )
    sparse_conv_params = dict(
        threshold_sigma=1.0,
        negative_slope=0.0,
        max_value=None,
        filters=filters,
        padding="same",
        symmetric=False,
        kernel_size=kernel_size,
        kernel_regularizer=kernel_regularizer,
        kernel_initializer=kernel_initializer
    )
    final_conv_params = dict(
        kernel_size=1,
        padding="same",
        strides=(1, 1),
        activation=final_activation,
        filters=input_dims[channel_index],
        kernel_regularizer=kernel_regularizer,
        kernel_initializer=kernel_initializer
    )

    # --- set input
    model_input = keras.Input(shape=input_dims)

    # --- add base layer
    x = model_input
    x = keras.layers.Conv2D(**base_conv_params)(x)
    _, sigma = mean_sigma(x, kernel_size=(5, 5))
    x = keras.layers.Concatenate()([x, sigma])

    # --- add resnet layers
    for i in range(no_layers):
        previous_layer = x
        x = conv2d_sparse(x, **sparse_conv_params)
        x = keras.layers.Conv2D(**conv_params)(x)
        _, sigma = mean_sigma(x, kernel_size=(5, 5))
        x = keras.layers.Concatenate()([x, sigma])
        x = keras.layers.Add()([previous_layer, x])

    # --- output to original channels
    output_layer = \
        keras.layers.Conv2D(**final_conv_params)(x)

    return keras.Model(
        name=name,
        inputs=model_input,
        outputs=output_layer)

# ---------------------------------------------------------------------


def build_gatenet_model(
        input_dims,
        no_layers: int,
        kernel_size: int,
        filters: int,
        activation: str = "relu",
        final_activation: str = "linear",
        use_bn: bool = True,
        use_bias: bool = False,
        kernel_regularizer="l1",
        kernel_initializer="glorot_normal",
        channel_index: int = 2,
        name="gatenet") -> keras.Model:
    """
    Build a gatenet model

    :param input_dims: Models input dimensions
    :param no_layers: Number of resnet layers
    :param kernel_size: kernel size of the conv layers
    :param filters: number of filters per convolutional layer
    :param activation: intermediate activation
    :param final_activation: activation of the final layer
    :param channel_index: Index of the channel in dimensions
    :param use_bn: Use Batch Normalization
    :param use_bias: use bias
    :param kernel_regularizer: Kernel weight regularizer
    :param kernel_initializer: Kernel weight initializer
    :param name: Name of the model
    :return:
    """
    # --- variables
    bn_params = dict(
        center=use_bias,
        scale=True,
        momentum=0.999,
        epsilon=1e-4
    )
    conv_params = dict(
        filters=filters,
        strides=(1, 1),
        padding="same",
        use_bias=use_bias,
        activation=activation,
        kernel_size=kernel_size,
        kernel_regularizer=kernel_regularizer,
        kernel_initializer=kernel_initializer
    )
    gate_conv_params = dict(
        filters=filters,
        strides=(1, 1),
        padding="same",
        use_bias=use_bias,
        activation="linear",
        kernel_size=kernel_size,
        kernel_regularizer=kernel_regularizer,
        kernel_initializer=kernel_initializer
    )
    depth_conv_params = dict(
        depth_multiplier=2,
        strides=(1, 1),
        padding="same",
        use_bias=use_bias,
        activation=activation,
        kernel_size=kernel_size,
        depthwise_regularizer=kernel_regularizer,
        depthwise_initializer=kernel_initializer
    )
    # intermediate conv
    intermediate_conv_params = copy.deepcopy(conv_params)
    intermediate_conv_params["kernel_size"] = 1

    # final conv
    final_conv_params = copy.deepcopy(conv_params)
    final_conv_params["kernel_size"] = 1
    final_conv_params["activation"] = final_activation
    final_conv_params["filters"] = input_dims[channel_index]
    del final_conv_params["kernel_regularizer"]

    # --- set input
    input_layer = \
        keras.Input(shape=input_dims)

    # --- add base layer
    x = keras.layers.Conv2D(**conv_params)(input_layer)

    # --- add signal/gate resnet layers
    s_layer = x
    g_layer = x
    for i in range(no_layers):
        previous_s_layer = s_layer
        previous_g_layer = g_layer

        # --- add extra layers
        if i == 0:
            s_layer = previous_s_layer
            g_layer = previous_g_layer
        else:
            s_layer = \
                keras.layers.Concatenate()([
                    previous_s_layer,
                    input_layer])
            g_layer = \
                keras.layers.Concatenate()([
                    previous_s_layer,
                    previous_g_layer
                ])
        g_layer = keras.backend.stop_gradient(g_layer)

        # --- expand
        s_layer = \
            keras.layers.DepthwiseConv2D(**depth_conv_params)(s_layer)
        g_layer = \
            keras.layers.Conv2D(**conv_params)(g_layer)

        # --- normalize
        if use_bn:
            s_layer = \
                keras.layers.BatchNormalization(**bn_params)(s_layer)
            g_layer = \
                keras.layers.BatchNormalization(**bn_params)(g_layer)

        # --- compress
        s_layer = \
            keras.layers.Conv2D(**intermediate_conv_params)(s_layer)
        g_layer = \
            keras.layers.Conv2D(**gate_conv_params)(g_layer)

        # --- compute activation per channel
        g_layer_activation = \
            keras.layers.GlobalAvgPool2D()(g_layer)
        g_layer_activation = \
            (keras.layers.Activation("tanh")(
                g_layer_activation * 2) + 1.0) / 2.0

        # mask channels
        s_layer = \
            keras.layers.Multiply()([s_layer, g_layer_activation])

        # add skip connection
        if i != 0:
            s_layer = \
                keras.layers.Add()([previous_s_layer, s_layer])
            g_layer = \
                keras.layers.Add()([previous_g_layer, g_layer])

    # --- output to original channels
    output_layer = \
        keras.layers.Conv2D(**final_conv_params)(s_layer)

    output_layer = \
        keras.layers.Add()([output_layer, input_layer])

    return keras.Model(
        name=name,
        inputs=input_layer,
        outputs=output_layer)


# ==============================================================================


def mobilenet_v2_block(
        input_layer,
        filters: List[int] = [32, 32],
        activation: str = "relu",
        initializer: str = "glorot_normal",
        regularizer: str = None,
        use_bias: bool = False,
        strides: List[int] = [1, 1],
        kernel_size: List[int] = [3, 3],
        bn_momentum: float = 0.999,
        bn_epsilon: float = 0.001):
    """

    :param input_layer:
    :param filters:
    :param activation:
    :param initializer:
    :param regularizer:
    :param use_bias:
    :param strides:
    :param kernel_size:
    :param bn_momentum:
    :param bn_epsilon:
    :return:
    """
    logger.info("building mobilenet_v2_block")

    # --- argument checking
    if input_layer is None:
        raise ValueError("input_layer cannot be empty")

    # ---
    tmp = \
        keras.layers.Conv2D(
            strides=(1, 1),
            padding="same",
            use_bias=use_bias,
            kernel_size=(1, 1),
            activation="linear",
            filters=filters[0],
            kernel_initializer=initializer,
            kernel_regularizer=regularizer)(input_layer)
    tmp = \
        keras.layers.BatchNormalization(
            momentum=bn_momentum,
            epsilon=bn_epsilon)(tmp)
    tmp = \
        keras.layers.Activation(activation)(tmp)
    tmp = \
        keras.layers.DepthwiseConv2D(
            padding="same",
            strides=strides,
            use_bias=use_bias,
            depth_multiplier=1,
            activation="linear",
            kernel_size=kernel_size,
            depthwise_initializer=initializer,
            depthwise_regularizer=regularizer,
            kernel_initializer=initializer,
            kernel_regularizer=regularizer)(tmp)
    tmp = \
        keras.layers.BatchNormalization(
            momentum=bn_momentum,
            epsilon=bn_epsilon)(tmp)
    tmp = \
        keras.layers.Activation(activation)(tmp)
    tmp = \
        keras.layers.Conv2D(
            strides=(1, 1),
            padding="same",
            use_bias=use_bias,
            kernel_size=(1, 1),
            activation="linear",
            filters=filters[1],
            kernel_initializer=initializer,
            kernel_regularizer=regularizer)(tmp)
    tmp = \
        keras.layers.BatchNormalization(
            momentum=bn_momentum,
            epsilon=bn_epsilon)(tmp)
    return tmp


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
