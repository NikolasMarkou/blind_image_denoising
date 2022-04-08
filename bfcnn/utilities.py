# ---------------------------------------------------------------------

__author__ = "Nikolas Markou"
__version__ = "1.0.0"
__license__ = "MIT"

# ---------------------------------------------------------------------

import os
import json
import keras
import itertools
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import List, Tuple, Union, Dict, Iterable

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .custom_logger import logger
from .custom_layers import TrainableMultiplier, RandomOnOff

# ---------------------------------------------------------------------

DEFAULT_EPSILON = 0.001

# ---------------------------------------------------------------------


def load_config(
        config: Union[str, Dict, Path]) -> Dict:
    """
    Load configuration from multiple sources

    :param config: dict configuration or path to json configuration
    :return: dictionary configuration
    """
    if config is None:
        raise ValueError("config should not be empty")
    if isinstance(config, Dict):
        return config
    if isinstance(config, str) or isinstance(config, Path):
        if not os.path.isfile(str(config)):
            return ValueError(
                "configuration path [{0}] is not valid".format(
                    str(config)
                ))
        with open(str(config), "r") as f:
            return json.load(f)
    raise ValueError("don't know how to handle config [{0}]".format(config))

# ---------------------------------------------------------------------


def input_shape_fixer(
        input_shape: Iterable):
    for i, shape in enumerate(input_shape):
        if shape == "?" or \
                shape == "" or \
                shape == "-1":
            input_shape[i] = None
    return input_shape

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


def random_choice(x, size, axis=0):
    """
    Randomly select size options from x
    """
    dim_x = tf.cast(tf.shape(x)[axis], tf.int64)
    indices = tf.range(0, dim_x, dtype=tf.int64)
    sample_index = tf.random.shuffle(indices)[:size]
    sample = tf.gather(x, sample_index, axis=axis)
    return sample

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

def step_function(
        input_layer,
        offset: float = 0.0,
        multiplier: float = 20.0):
    """
    Differentiable step function approximation using tanh
    non-saturation y:(0, 1) range x:(-0.1, +0.1)

    :param input_layer:
    :param offset:
    :param multiplier:
    :result:
    """
    x = input_layer
    if offset != 0.0:
        x = x - offset
    return (tf.math.tanh(x * multiplier) + 1.0) * 0.5

# ---------------------------------------------------------------------


def conv2d_wrapper(
        input_layer,
        conv_params: Dict,
        bn_params: Dict = None):
    use_bn = bn_params is not None
    # ---
    x = input_layer
    if use_bn:
        x = keras.layers.BatchNormalization(**bn_params)(x)
    x = keras.layers.Conv2D(**conv_params)(x)
    return x

# ---------------------------------------------------------------------


def learnable_per_channel_multiplier_layer(
        input_layer,
        multiplier: float = 1.0,
        activation: str = "linear",
        kernel_regularizer: str = "l1",
        trainable: bool = True):
    """
    Constant learnable multiplier layer

    :param input_layer: input layer to be multiplied
    :param multiplier: multiplication constant
    :param activation: activation after the filter (linear by default)
    :param kernel_regularizer: regularize kernel weights (None by default)
    :param trainable: whether this layer is trainable or not
    :return: multiplied input_layer
    """
    # --- initialise to set kernel to required value
    def kernel_init(shape, dtype):
        kernel = np.zeros(shape)
        for i in range(shape[2]):
            kernel[:, :, i, 0] = \
                np.random.normal(scale=DEFAULT_EPSILON, loc=0.0)
        return kernel
    x = \
        keras.layers.DepthwiseConv2D(
            kernel_size=1,
            padding="same",
            strides=(1, 1),
            use_bias=False,
            depth_multiplier=1,
            trainable=trainable,
            activation=activation,
            kernel_initializer=kernel_init,
            depthwise_initializer=kernel_init,
            kernel_regularizer=kernel_regularizer)(input_layer)
    # different scenarios
    if multiplier == 0.0:
        return x
    if multiplier == 1.0:
        return input_layer + x
    return (multiplier * input_layer) + x


# ---------------------------------------------------------------------


def mean_variance_local(
        input_layer,
        kernel_size: Tuple[int, int] = (5, 5)):
    """
    calculate window mean per channel and window variance per channel

    :param input_layer: the layer to operate on
    :param kernel_size: size of the kernel (window)
    :return: mean, variance tensors
    """
    # --- argument checking
    if input_layer is None:
        raise ValueError("input_layer cannot be empty")
    shape = keras.backend.int_shape(input_layer)
    if len(shape) != 4:
        raise ValueError("input_layer must be a 4d tensor")
    if not isinstance(kernel_size, tuple):
        raise ValueError("kernel_size must be a tuple")

    # ---
    local_mean = \
        keras.layers.AveragePooling2D(
            strides=(1, 1),
            padding="same",
            pool_size=kernel_size)(input_layer)
    local_diff = \
        keras.layers.Subtract()(
            [input_layer, local_mean])
    local_diff = keras.backend.square(local_diff)
    local_variance = \
        keras.layers.AveragePooling2D(
            strides=(1, 1),
            padding="same",
            pool_size=kernel_size)(local_diff)

    return local_mean, local_variance

# ---------------------------------------------------------------------


def mean_sigma_local(
        input_layer,
        kernel_size: Tuple[int, int] = (5, 5),
        epsilon: float = DEFAULT_EPSILON):
    """
    calculate window mean per channel and window sigma per channel

    :param input_layer: the layer to operate on
    :param kernel_size: size of the kernel (window)
    :param epsilon: small number for robust sigma calculation
    :return: mean, sigma tensors
    """
    mean, variance = \
        mean_variance_local(
            input_layer=input_layer,
            kernel_size=kernel_size)

    sigma = tf.sqrt(tf.abs(variance) + epsilon)

    return mean, sigma


# ---------------------------------------------------------------------


def mean_sigma_global(
        input_layer,
        axis: List[int] = [1, 2, 3],
        sigma_epsilon: float = DEFAULT_EPSILON):
    """
    Create a global mean sigma per channel

    :param input_layer:
    :param axis:
    :param sigma_epsilon: small number to add for robust sigma calculation
    :return:
    """
    # --- argument checking
    if input_layer is None:
        raise ValueError("input_layer cannot be empty")
    shape = keras.backend.int_shape(input_layer)
    if len(shape) != 4:
        raise ValueError("input_layer must be a 4d tensor")

    # --- compute mean and sigma
    def func(x):
        mean = keras.backend.mean(x, axis=axis, keepdims=True)
        diff_2 = keras.backend.square(x - mean)
        variance = keras.backend.mean(diff_2, axis=axis, keepdims=True)
        sigma = keras.backend.sqrt(variance + sigma_epsilon)
        return mean, sigma

    return \
        keras.layers.Lambda(
            function=func,
            trainable=False)(input_layer)

# ---------------------------------------------------------------------


def differentiable_relu(
        input_layer,
        threshold: float = 0.0,
        max_value: float = 6.0,
        multiplier: float = 10.0):
    """
    Creates a differentiable relu operation

    :param input_layer:
    :param threshold: lower bound value before zeroing
    :param max_value: max allowed value
    :param multiplier: controls steepness
    :result:
    """
    # --- arguments check
    if threshold is None:
        raise ValueError("threshold must not be empty")
    if max_value is not None:
        if threshold > max_value:
            raise ValueError(
                f"max_value [{max_value}] must be > threshold [{threshold}")

    # --- function building
    def func_diff_relu_0(args):
        x = args
        step_threshold = tf.math.sigmoid(multiplier * (x - threshold))
        step_max_value = tf.math.sigmoid(multiplier * (x - max_value))
        result = \
            ((step_max_value * max_value) + ((1.0 - step_max_value) * x)) * \
            step_threshold
        return result

    def func_diff_relu_1(args):
        x = args
        step_threshold = tf.math.sigmoid(multiplier * (x - threshold))
        result = step_threshold * x
        return result

    fn = func_diff_relu_0
    if max_value is None:
        fn = func_diff_relu_1

    return \
        keras.layers.Lambda(
            function=fn,
            trainable=False)(input_layer)

# ---------------------------------------------------------------------


def gelu_block(input_layer):
    x = input_layer
    x_sigmoid = keras.activations.sigmoid(x * 1.702)
    return keras.layers.Multiply()([x, x_sigmoid])

# ---------------------------------------------------------------------


def sparse_block(
        input_layer,
        bn_params: Dict = None,
        threshold_sigma: float = 1.0,
        max_value: float = None,
        symmetric: bool = True,
        per_channel_sparsity: bool = False):
    """
    Create sparsity in an input layer

    :param input_layer:
    :param bn_params: batch norm parameters
    :param threshold_sigma: sparsity of the results
    :param max_value: max allowed value
    :param symmetric: if true allow negative values else zero them off
    :param per_channel_sparsity: if true perform sparsity on per channel level

    :return: sparse results
    """
    # --- argument checking
    if threshold_sigma < 0:
        raise ValueError("threshold_sigma must be >= 0")
    if max_value is not None and max_value < 0:
        raise ValueError("max_value must be >= 0")
    if max_value is not None and threshold_sigma > max_value:
        raise ValueError("threshold_sigma must be <= max_value")
    use_bn = bn_params is not None

    # --- function building
    def func_norm(args):
        x, mean_x, sigma_x = args
        return (x - mean_x) / sigma_x

    def func_abs_sign(args):
        y = args
        y_abs = keras.backend.abs(y)
        y_sign = keras.backend.sign(y)
        return y_abs, y_sign

    # --- computation is relu((mean-x)/sigma) with custom threshold
    # compute axis to perform mean/sigma calculation on
    if use_bn:
        # learnable model
        x_bn = \
            keras.layers.BatchNormalization(
                **bn_params)(input_layer)
    else:
        # not learnable model
        shape = keras.backend.int_shape(input_layer)
        int_shape = len(shape)
        k = 1
        if per_channel_sparsity:
            k = 2
        axis = list([i + 1 for i in range(max(int_shape - k, 1))])
        mean, sigma = \
            mean_sigma_global(
                input_layer,
                axis=axis)
        x_bn = \
            keras.layers.Lambda(func_norm, trainable=False)([
                input_layer, mean, sigma])

    # ---
    if symmetric:
        x_abs, x_sign = \
            keras.layers.Lambda(func_abs_sign, trainable=False)(x_bn)
        x_relu = \
            differentiable_relu(
                input_layer=x_abs,
                threshold=threshold_sigma,
                max_value=max_value)
        x_result = \
            keras.layers.Multiply()([
                x_relu,
                x_sign,
            ])
    else:
        x_result = \
            differentiable_relu(
                input_layer=x_bn,
                threshold=threshold_sigma,
                max_value=max_value)

    return x_result

# ---------------------------------------------------------------------


def layer_denormalize(args):
    """
    Convert input [-0.5, +0.5] to [v0, v1] range
    """
    y, v_min, v_max = args
    y_clip = \
        tf.clip_by_value(
            t=y,
            clip_value_min=-0.5,
            clip_value_max=0.5)
    return (y_clip + 0.5) * (v_max - v_min) + v_min


# ---------------------------------------------------------------------


def layer_normalize(args):
    """
    Convert input from [v0, v1] to [-0.5, +0.5] range
    """
    y, v_min, v_max = args
    y_clip = \
        tf.clip_by_value(
            t=y,
            clip_value_min=v_min,
            clip_value_max=v_max)
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
    :return: normalization model
    """
    model_input = keras.Input(shape=input_dims)

    # --- normalize input
    # from [min_value, max_value] to [-0.5, +0.5]
    model_output = \
        keras.layers.Lambda(
            function=layer_normalize,
            trainable=False)([model_input,
                              float(min_value),
                              float(max_value)])

    # --- wrap model
    return keras.Model(
        name=name,
        trainable=False,
        inputs=model_input,
        outputs=model_output)


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
    :return: denormalization model
    """
    model_input = keras.Input(shape=input_dims)

    # --- normalize input
    # from [min_value, max_value] to [-0.5, +0.5]
    model_output = \
        keras.layers.Lambda(
            function=layer_denormalize,
            trainable=False)([model_input,
                              float(min_value),
                              float(max_value)])

    # --- wrap model
    return \
        keras.Model(
            name=name,
            trainable=False,
            inputs=model_input,
            outputs=model_output)


# ---------------------------------------------------------------------


def resnet_blocks(
        input_layer,
        no_layers: int,
        first_conv_params: Dict,
        second_conv_params: Dict,
        third_conv_params: Dict,
        bn_params: Dict = None,
        gate_params: Dict = None,
        dropout_params: Dict = None,
        multiplier_params: Dict = None,
        **kwargs):
    """
    Create a series of residual network blocks

    :param input_layer: the input layer to perform on
    :param no_layers: how many residual network blocks to add
    :param first_conv_params: the parameters of the first conv
    :param second_conv_params: the parameters of the middle conv
    :param third_conv_params: the parameters of the third conv
    :param bn_params: batch normalization parameters
    :param gate_params: gate optional parameters
    :param dropout_params: dropout optional parameters
    :param multiplier_params: learnable optional parameters

    :return: filtered input_layer
    """
    # --- argument check
    if input_layer is None:
        raise ValueError("input_layer must be none")
    if no_layers < 0:
        raise ValueError("no_layers must be >= 0")
    use_gate = gate_params is not None
    use_dropout = dropout_params is not None
    use_multiplier = multiplier_params is not None

    # --- setup resnet
    x = input_layer
    g_layer = None

    # --- create several number of residual blocks
    for i in range(no_layers):
        previous_layer = x
        x = conv2d_wrapper(x, conv_params=first_conv_params, bn_params=bn_params)
        x = conv2d_wrapper(x, conv_params=second_conv_params, bn_params=bn_params)
        x = conv2d_wrapper(x, conv_params=third_conv_params, bn_params=bn_params)
        # compute activation per channel
        if use_gate:
            y = conv2d_wrapper(y, conv_params=gate_params, bn_params=bn_params)
            y = keras.layers.GlobalAveragePooling2D()(y)
            if g_layer is not None:
                y = (1.0 - g_layer) * y
            g_layer = y
            y = keras.layers.Dense(
                use_bias=False,
                activation="relu",
                units=third_conv_params["filters"],
                kernel_regularizer=third_conv_params.get("kernel_regularizer", None),
                kernel_initializer=third_conv_params.get("kernel_initializer", None))(y)
            y = tf.reshape(y, shape=(-1, 1, 1, third_conv_params["filters"]))
            # --- on by default, requires effort to turn off
            # if x < -2.5: return 0
            # if x > 2.5: return 1
            # if -2.5 <= x <= 2.5: return 0.2 * x + 0.5
            y = keras.activations.hard_sigmoid(2.5 - y)
            x = keras.layers.Multiply()([x, y])
        # optional multiplier
        if use_multiplier:
            x = TrainableMultiplier(**multiplier_params)(x)
        if use_dropout:
            x = RandomOnOff(**dropout_params)(x)
        # skip connection
        x = keras.layers.Add()([x, previous_layer])
    return x

# ---------------------------------------------------------------------


def convnext_blocks(
        input_layer,
        no_layers: int,
        first_conv_params: Dict,
        second_conv_params: Dict,
        third_conv_params: Dict,
        bn_params: Dict = None,
        gate_params: Dict = None,
        dropout_params: Dict = None,
        **kwargs):
    """
    Create a series of residual network blocks

    :param input_layer: the input layer to perform on
    :param no_layers: how many residual network blocks to add
    :param first_conv_params: the parameters of the first conv
    :param second_conv_params: the parameters of the middle conv
    :param third_conv_params: the parameters of the third conv
    :param bn_params: batch normalization parameters
    :param gate_params: gate optional parameters
    :param dropout_params: dropout optional parameters

    :return: filtered input_layer
    """
    # --- argument check
    if input_layer is None:
        raise ValueError("input_layer must be none")
    if no_layers < 0:
        raise ValueError("no_layers must be >= 0")
    use_bn = bn_params is not None
    use_gate = gate_params is not None
    use_dropout = dropout_params is not None

    # --- setup resnet
    x = input_layer
    g_layer = x

    # --- create several number of residual blocks
    for i in range(no_layers):
        previous_layer = x
        if use_dropout:
            x = keras.layers.SpatialDropout2D(**dropout_params)(x)
        # 1st conv
        x = keras.layers.DepthwiseConv2D(**first_conv_params)(x)
        if use_bn:
            x = keras.layers.LayerNormalization(**bn_params)(x)
        # 2nd conv
        x = keras.layers.Conv2D(**second_conv_params)(x)
        x = gelu_block(x)
        # output results
        x = keras.layers.Conv2D(**third_conv_params)(x)
        # compute activation per channel (squeeze and excite)
        if use_gate:
            g_layer = keras.layers.Add()([x, g_layer])
            y = g_layer
            if use_bn:
                y = keras.layers.LayerNormalization(**bn_params)(y)
            # activation per pixel
            y = keras.layers.Conv2D(**gate_params)(y)
            y = keras.layers.GlobalAveragePooling2D()(y)
            # y = \
            #     TrainableMultiplier(
            #         multiplier=1.0,
            #         regularizer="l1",
            #         trainable=True)(y)
            y = keras.layers.Dense(
                units=third_conv_params["filters"],
                activation="linear")(y)
            y = tf.reshape(y, shape=(-1, 1, 1, third_conv_params["filters"]))
            # on by default, requires effort to turn off
            # if x < -2.5: return 0
            # if x > 2.5: return 1
            # if -2.5 <= x <= 2.5: return 0.2 * x + 0.5
            y = keras.activations.hard_sigmoid(2.5 - y)
            x = keras.layers.Multiply()([x, y])
        # skip connection
        x = keras.layers.Add()([x, previous_layer])
    return x

# ---------------------------------------------------------------------


def build_model_resnet(
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
        dropout_rate: float = -1,
        add_skip_with_input: bool = True,
        add_sparsity: bool = False,
        add_gates: bool = False,
        add_var: bool = False,
        add_intermediate_results: bool = False,
        add_learnable_multiplier: bool = False,
        add_projection_to_input: bool = True,
        name="resnet") -> keras.Model:
    """
    builds a resnet model

    :param input_dims: Models input dimensions
    :param no_layers: Number of resnet layers
    :param kernel_size: kernel size of the conv layers
    :param filters: number of filters per convolutional layer
    :param activation: intermediate activation
    :param final_activation: activation of the final layer
    :param channel_index: Index of the channel in dimensions
    :param dropout_rate: probability of resnet block shutting off
    :param use_bn: Use Batch Normalization
    :param use_bias: use bias
    :param kernel_regularizer: Kernel weight regularizer
    :param kernel_initializer: Kernel weight initializer
    :param add_skip_with_input: if true skip with input
    :param add_sparsity: if true add sparsity layer
    :param add_gates: if true add gate layer
    :param add_var: if true add variance for each block
    :param add_intermediate_results: if true output results before projection
    :param add_learnable_multiplier:
    :param add_projection_to_input: if true project to input tensor channel number
    :param name: name of the model
    :return: resnet model
    """
    # --- setup parameters
    bn_params = dict(
        center=use_bias,
        scale=True,
        momentum=0.999,
        epsilon=1e-4
    )

    # this make it 68% sparse
    sparse_params = dict(
        symmetric=True,
        max_value=3.0,
        threshold_sigma=1.0,
        per_channel_sparsity=False
    )

    base_conv_params = dict(
        filters=filters,
        strides=(1, 1),
        padding="same",
        use_bias=use_bias,
        activation="linear",
        kernel_size=kernel_size,
        kernel_regularizer=kernel_regularizer,
        kernel_initializer=kernel_initializer
    )

    gate_params = dict(
        kernel_size=1,
        filters=filters,
        strides=(1, 1),
        padding="same",
        use_bias=use_bias,
        activation=activation,
        kernel_regularizer=kernel_regularizer,
        kernel_initializer=kernel_initializer
    )

    first_conv_params = dict(
        kernel_size=1,
        filters=filters,
        strides=(1, 1),
        padding="same",
        use_bias=use_bias,
        activation=activation,
        kernel_regularizer=kernel_regularizer,
        kernel_initializer=kernel_initializer,
    )

    second_conv_params = dict(
        kernel_size=3,
        filters=filters * 2,
        strides=(1, 1),
        padding="same",
        use_bias=use_bias,
        activation=activation,
        kernel_regularizer=kernel_regularizer,
        kernel_initializer=kernel_initializer
    )

    third_conv_params = dict(
        groups=2,
        kernel_size=1,
        filters=filters,
        strides=(1, 1),
        padding="same",
        use_bias=use_bias,
        # this must be the same as the base
        activation="linear",
        kernel_regularizer=kernel_regularizer,
        kernel_initializer=kernel_initializer
    )

    final_conv_params = dict(
        kernel_size=1,
        strides=(1, 1),
        padding="same",
        use_bias=use_bias,
        # this must be linear because it is capped later
        activation="linear",
        filters=input_dims[channel_index],
        kernel_regularizer=kernel_regularizer,
        kernel_initializer=kernel_initializer
    )

    multiplier_params = dict(
        multiplier=1.0,
        trainable=True,
        regularizer="l1",
        activation="linear"
    )

    dropout_params = dict(
        rate=dropout_rate
    )

    resnet_params = dict(
        no_layers=no_layers,
        bn_params=bn_params,
        first_conv_params=first_conv_params,
        second_conv_params=second_conv_params,
        third_conv_params=third_conv_params,
    )

    # make it linear so it gets sparse afterwards
    if add_sparsity:
        base_conv_params["activation"] = "linear"

    if add_gates:
        resnet_params["gate_params"] = gate_params

    if add_learnable_multiplier:
        resnet_params["multiplier_params"] = multiplier_params

    if dropout_rate != -1:
        resnet_params["dropout_params"] = dropout_params

    # --- build model
    # set input
    input_layer = \
        keras.Input(
            name="input_tensor",
            shape=input_dims)
    x = input_layer
    y = input_layer

    if add_var:
        _, x_var = \
            mean_sigma_local(
                input_layer=x,
                kernel_size=(5, 5))
        x = keras.layers.Concatenate()([x, x_var])

    # add base layer
    x = keras.layers.Conv2D(**base_conv_params)(x)

    if add_sparsity:
        x = \
            sparse_block(
                input_layer=x,
                bn_params=None,
                **sparse_params)

    # add resnet blocks
    x = \
        resnet_blocks(
            input_layer=x,
            **resnet_params)

    # optional batch norm
    if use_bn:
        x = keras.layers.BatchNormalization(**bn_params)(x)

    # --- output layer branches here,
    # to allow space for intermediate results
    output_layer = x

    # --- output to original channels / projection
    if add_projection_to_input:
        output_layer = \
            keras.layers.Conv2D(
                **final_conv_params)(output_layer)

        # learnable multiplier
        if add_learnable_multiplier:
            output_layer = \
                TrainableMultiplier(**multiplier_params)(output_layer)

        # cap it off to limit values
        output_layer = \
            keras.layers.Activation(
                activation=final_activation)(output_layer)

    # --- skip with input layer
    if add_skip_with_input:
        output_layer = \
            keras.layers.Add()([output_layer, y])

    output_layer = \
        keras.layers.Layer(name="output_tensor")(output_layer)

    # return intermediate results if flag is turned on
    output_layers = [output_layer]
    if add_intermediate_results:
        output_layers.append(
            keras.layers.Layer(name="intermediate_tensor")(x))

    return \
        keras.Model(
            name=name,
            trainable=True,
            inputs=input_layer,
            outputs=output_layers)

# ---------------------------------------------------------------------
