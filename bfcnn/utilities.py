import os
import json
import copy
import numpy as np
from enum import Enum
import tensorflow as tf
from pathlib import Path
from typing import List, Tuple, Iterable

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .constants import *
from .custom_logger import logger
from .regularizers import builder as regularizer_builder
from .custom_layers import Mish, ChannelwiseMultiplier, GlobalLearnableMultiplier


# ---------------------------------------------------------------------


def clip_normalized_tensor(
        input_tensor: tf.Tensor) -> tf.Tensor:
    """
    clip an input to [-0.5, +0.5]

    :param input_tensor:
    :return:
    """
    return \
        tf.clip_by_value(
            input_tensor,
            clip_value_min=-0.5,
            clip_value_max=+0.5)


# ---------------------------------------------------------------------


def clip_unnormalized_tensor(
        input_tensor: tf.Tensor) -> tf.Tensor:
    """
    clip an input to [0.0, 255.0]

    :param input_tensor:
    :return:
    """
    return \
        tf.clip_by_value(
            input_tensor,
            clip_value_min=0.0,
            clip_value_max=255.0)


# ---------------------------------------------------------------------


def load_config(
        config: Union[str, Dict, Path]) -> Dict:
    """
    Load configuration from multiple sources

    :param config: dict configuration or path to json configuration
    :return: dictionary configuration
    """
    try:
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
    except Exception as e:
        logger.error(e)
        raise ValueError(f"failed to load [{config}]")


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


class ConvType(Enum):
    CONV2D = 0

    CONV2D_DEPTHWISE = 1

    CONV2D_TRANSPOSE = 2

    CONV2D_SEPARABLE = 3

    @staticmethod
    def from_string(type_str: str) -> "ConvType":
        # --- argument checking
        if type_str is None:
            raise ValueError("type_str must not be null")
        if not isinstance(type_str, str):
            raise ValueError("type_str must be string")
        type_str = type_str.strip().upper()
        if len(type_str) <= 0:
            raise ValueError("stripped type_str must not be empty")

        # --- clean string and get
        return ConvType[type_str]

    def to_string(self) -> str:
        return self.name


# ---------------------------------------------------------------------


def conv2d_wrapper(
        input_layer,
        conv_params: Dict,
        bn_params: Dict = None,
        ln_params: Dict = None,
        dropout_params: Dict = None,
        dropout_2d_params: Dict = None,
        conv_type: Union[ConvType, str] = ConvType.CONV2D):
    """
    wraps a conv2d with a preceding normalizer

    if bn_post_params force a conv(linear)->bn->activation setup

    :param input_layer: the layer to operate on
    :param conv_params: conv2d parameters
    :param bn_params: batchnorm parameters before the conv, None to disable bn
    :param ln_params: layer normalization parameters before the conv, None to disable ln
    :param dropout_params: dropout parameters after the conv, None to disable it
    :param dropout_2d_params: dropout parameters after the conv, None to disable it
    :param conv_type: if true use depthwise convolution,

    :return: transformed input
    """
    from .backbone_blocks import squeeze_and_excite_block

    # --- argument checking
    if input_layer is None:
        raise ValueError("input_layer cannot be None")
    if conv_params is None:
        raise ValueError("conv_params cannot be None")

    # --- prepare arguments
    use_ln = ln_params is not None
    use_bn = bn_params is not None
    use_dropout = dropout_params is not None
    use_dropout_2d = dropout_2d_params is not None
    conv_params = copy.deepcopy(conv_params)
    conv_activation = conv_params.get("activation", "linear")
    conv_params["activation"] = "linear"

    if conv_params.get("use_bias", True) and \
            (conv_activation == "relu" or conv_activation == "relu6"):
        conv_params["bias_initializer"] = \
            tf.keras.initializers.Constant(DEFAULT_RELU_BIAS)

    # TODO restructure this
    if isinstance(conv_type, str):
        conv_type = ConvType.from_string(conv_type)
    if "depth_multiplier" in conv_params:
        if conv_type != ConvType.CONV2D_DEPTHWISE:
            logger.info("Changing conv_type to CONV2D_DEPTHWISE because it contains depth_multiplier argument "
                        f"[conv_params[\'depth_multiplier\']={conv_params['depth_multiplier']}]")
        conv_type = ConvType.CONV2D_DEPTHWISE
    if "dilation_rate" in conv_params:
        if conv_type != ConvType.CONV2D_TRANSPOSE:
            logger.info("Changing conv_type to CONV2D_TRANSPOSE because it contains dilation argument "
                        f"[conv_params[\'dilation_rate\']={conv_params['dilation_rate']}]")
        conv_type = ConvType.CONV2D_TRANSPOSE

    # --- set up stack of operation
    x = input_layer

    # --- convolution
    if conv_type == ConvType.CONV2D:
        x = tf.keras.layers.Conv2D(**conv_params)(x)
    elif conv_type == ConvType.CONV2D_DEPTHWISE:
        x = tf.keras.layers.DepthwiseConv2D(**conv_params)(x)
    elif conv_type == ConvType.CONV2D_TRANSPOSE:
        x = tf.keras.layers.Conv2DTranspose(**conv_params)(x)
    elif conv_type == ConvType.CONV2D_SEPARABLE:
        x = tf.keras.layers.SeparableConv2D(**conv_params)(x)
    else:
        raise ValueError(f"don't know how to handle this [{conv_type}]")

    # --- perform post convolution normalizations and activation
    if use_bn:
        x = tf.keras.layers.BatchNormalization(**bn_params)(x)
    if use_ln:
        x = tf.keras.layers.LayerNormalization(**ln_params)(x)

    # --- perform activation post normalization
    if (conv_activation is not None and
            conv_activation != "linear"):
        x = activation_wrapper(conv_activation)(x)

    # --- dropout
    if use_dropout:
        x = tf.keras.layers.Dropout(**dropout_params)(x)

    if use_dropout_2d:
        x = tf.keras.layers.SpatialDropout2D(**dropout_2d_params)(x)

    return x


# ---------------------------------------------------------------------

def activation_wrapper(
        activation: Union[tf.keras.layers.Layer, str] = "linear") -> tf.keras.layers.Layer:
    if not isinstance(activation, str):
        logger.warning("cannot wrap activation since it is already wrapper")
        return activation

    activation = activation.lower().strip()

    if activation in ["mish"]:
        # Mish: A Self Regularized Non-Monotonic Activation Function (2020)
        from .custom_layers import Mish
        x = Mish()
    elif activation in ["melu"]:
        # Mish: A Self Regularized Non-Monotonic Activation Function (2020)
        from .custom_layers import Mish
        x = Mish()
    elif activation in ["leakyrelu", "leaky_relu"]:
        # leaky relu, practically same us Relu
        # with very small negative slope to allow gradient flow
        x = tf.keras.layers.LeakyReLU(alpha=0.3)
    elif activation in ["leakyrelu_01", "leaky_relu_01"]:
        # leaky relu, practically same us Relu
        # with very small negative slope to allow gradient flow
        x = tf.keras.layers.LeakyReLU(alpha=0.1)
    elif activation in ["leaky_relu_001", "leakyrelu_001"]:
        # leaky relu, practically same us Relu
        # with very small negative slope to allow gradient flow
        x = tf.keras.layers.LeakyReLU(alpha=0.01)
    elif activation in ["prelu"]:
        # parametric Rectified Linear Unit
        constraint = \
            tf.keras.constraints.MinMaxNorm(
                min_value=0.0, max_value=1.0, rate=1.0, axis=0)
        x = tf.keras.layers.PReLU(
            alpha_initializer=0.1,
            # very small l1
            alpha_regularizer=tf.keras.regularizers.l1(1e-3),
            alpha_constraint=constraint,
            shared_axes=[1, 2])
    else:
        x = tf.keras.layers.Activation(activation)

    return x


# ---------------------------------------------------------------------

def depthwise_gaussian_kernel(
        channels: int = 3,
        kernel_size: Tuple[int, int] = (5, 5),
        nsig: Tuple[float, float] = (2.0, 2.0),
        dtype: np.dtype = np.float64):
    def gaussian_kernel(
            _kernel_size: Tuple[int, int],
            _nsig: Tuple[float, float]) -> np.ndarray:
        """
        builds a 2D Gaussian kernel array

        :param _kernel_size: size of the grid
        :param _nsig: max value out of the gaussian on the xy axis
        :return: 2d gaussian grid
        """
        assert len(_nsig) == 2
        assert len(_kernel_size) == 2
        kern1d = [
            np.linspace(
                start=-np.abs(_nsig[i]),
                stop=np.abs(_nsig[i]),
                num=_kernel_size[i],
                endpoint=True,
                dtype=np.float64)
            for i in range(2)
        ]
        x, y = np.meshgrid(kern1d[0], kern1d[1])
        d = np.sqrt(x * x + y * y)
        sigma, mu = 1.0, 0.0
        g = np.exp(-((d - mu) ** 2 / (2.0 * (sigma ** 2))))
        return g / g.sum()

    def kernel_init(shape):
        logger.info(f"building gaussian kernel with size: {shape}")
        kernel = np.zeros(shape)
        kernel_channel = \
            gaussian_kernel(
                _kernel_size=(shape[0], shape[1]),
                _nsig=nsig)
        logger.info(f"gaussian kernel: \n{kernel_channel}")

        for i in range(shape[2]):
            kernel[:, :, i, 0] = kernel_channel
        return kernel

    # [filter_height, filter_width, in_channels, channel_multiplier]
    result = kernel_init(
        shape=(kernel_size[0], kernel_size[1], channels, 1))

    return result.astype(dtype=dtype)


# ---------------------------------------------------------------------


def dense_wrapper(
        input_layer,
        dense_params: Dict,
        bn_params: Dict = None,
        elementwise_params: Dict = None):
    """
    wraps a dense layer with a preceding normalizer

    :param input_layer: the layer to operate on
    :param dense_params: dense parameters
    :param bn_params: batchnorm parameters, None to disable bn
    :param elementwise_params: if True add a learnable elementwise scaling
    :return: transformed input
    """
    # --- argument checking
    if input_layer is None:
        raise ValueError("input_layer cannot be None")
    if dense_params is None:
        raise ValueError("dense_params cannot be None")

    # --- prepare arguments
    use_bn = bn_params is not None
    use_elementwise = elementwise_params is not None

    # --- perform the transformations
    x = input_layer
    if use_bn:
        x = tf.keras.layers.BatchNormalization(**bn_params)(x)
    # ideally this should be orthonormal
    x = tf.keras.layers.Dense(**dense_params)(x)
    # learn the proper scale of the previous layer
    if use_elementwise:
        x = ChannelwiseMultiplier(**elementwise_params)(x)
    return x


# ---------------------------------------------------------------------


def sparse_block(
        input_layer: tf.Tensor,
        bn_params: Dict = None,
        threshold_sigma: float = 1.0,
        symmetrical: bool = False,
        reverse: bool = False,
        soft_sparse: bool = False) -> tf.Tensor:
    """
    create sparsity in an input layer (keeps only positive)

    :param input_layer:
    :param bn_params: batch norm parameters, leave None for disabling
    :param threshold_sigma: sparsity of the results (assuming negative values in input)
    -3 -> 0.1% sparsity
    -2 -> 2.3% sparsity
    -1 -> 15.9% sparsity
    0  -> 50% sparsity
    +1 -> 84.1% sparsity
    +2 -> 97.7% sparsity
    +3 -> 99.9% sparsity
    :param symmetrical: if True use abs values, if False cutoff negatives
    :param reverse: if True cutoff large values, if False cutoff small values
    :param soft_sparse: if True use sigmoid, if False use relu

    :return: sparse results
    """
    # --- argument checking
    if threshold_sigma < 0:
        raise ValueError("threshold_sigma must be >= 0")

    # --- set variables
    use_bn = bn_params is not None

    # --- build sparse block
    x = input_layer

    # normalize
    if use_bn:
        x_bn = \
            tf.keras.layers.BatchNormalization(
                **bn_params)(x)

    if symmetrical:
        x_bn = tf.abs(x_bn)

    # threshold based on normalization
    # keep only positive above threshold
    if soft_sparse:
        x_binary = \
            tf.nn.sigmoid(x_bn - threshold_sigma)
    else:
        x_binary = \
            tf.nn.relu(tf.sign(x_bn - threshold_sigma))

    # focus on small values
    if reverse:
        x_binary = 1.0 - x_binary

    # zero out values below the threshold
    return \
        tf.keras.layers.Multiply()([
            x_binary,
            x,
        ])


# ---------------------------------------------------------------------


def layer_denormalize(
        input_layer: tf.Tensor,
        v_min: float,
        v_max: float) -> tf.Tensor:
    """
    Convert input [-0.5, +0.5] to [v0, v1] range
    """
    y_clip = clip_normalized_tensor(input_layer)
    return (y_clip + 0.5) * (v_max - v_min) + v_min


# ---------------------------------------------------------------------


def layer_normalize(
        input_layer: tf.Tensor,
        v_min: float,
        v_max: float) -> tf.Tensor:
    """
    Convert input from [v0, v1] to [-0.5, +0.5] range
    """
    y_clip = \
        tf.clip_by_value(
            t=input_layer,
            clip_value_min=v_min,
            clip_value_max=v_max)
    return (y_clip - v_min) / (v_max - v_min) - 0.5


# ---------------------------------------------------------------------

@tf.function
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
    original_dtype = input_batch.dtype
    batch_size = shape[0]

    if shape[1] <= 0 or shape[2] <= 0:
        return \
            tf.zeros(
                shape=(no_crops_per_image, crop_size[0], crop_size[1], shape[3]),
                dtype=original_dtype)

    # computer the total number of crops
    total_crops = no_crops_per_image * batch_size

    # fill y_range, x_range based on crop size and input batch size
    if y_range is None:
        y_range = (float(crop_size[0] / shape[1]),
                   float(crop_size[0] / shape[1]))

    if x_range is None:
        x_range = (float(crop_size[1] / shape[2]),
                   float(crop_size[1] / shape[2]))

    #
    y1 = \
        tf.random.uniform(
            shape=(total_crops, 1),
            minval=0.0,
            maxval=1.0 - y_range[0],
            seed=0)
    y2 = y1 + y_range[1]
    #
    x1 = \
        tf.random.uniform(
            shape=(total_crops, 1),
            minval=0.0,
            maxval=1.0 - x_range[0],
            seed=0)
    x2 = x1 + x_range[1]
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
            dtype=tf.int32,
            seed=0)

    result = \
        tf.image.crop_and_resize(
            image=input_batch,
            boxes=boxes,
            box_indices=box_indices,
            crop_size=crop_size,
            method=interpolation_method,
            extrapolation_value=extrapolation_value)

    del boxes
    del box_indices
    del x1, y1, x2, y2
    del y_range, x_range

    # --- cast to original img dtype (no surprises principle)
    return tf.cast(result, dtype=original_dtype)


# ---------------------------------------------------------------------

def global_normalization(
        input_layer):
    x = input_layer
    x_mean = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
    x_variance = \
        tf.reduce_mean(
            tf.square(x - x_mean), axis=[1, 2], keepdims=True)
    x_sigma = tf.sqrt(x_variance + DEFAULT_EPSILON)
    x = (x - x_mean) / x_sigma
    return x


# ---------------------------------------------------------------------


def local_normalization(
        input_layer,
        pool_size: Tuple[int, int] = (16, 16)):
    x = input_layer
    x_mean = \
        tf.keras.layers.AveragePooling2D(
            strides=(1, 1),
            pool_size=pool_size,
            padding="same")(x)
    x_variance = \
        tf.keras.layers.AveragePooling2D(
            strides=(1, 1),
            pool_size=pool_size,
            padding="same")(tf.square(x - x_mean))
    x_sigma = tf.sqrt(x_variance + DEFAULT_EPSILON)
    return (x - x_mean) / x_sigma


# ---------------------------------------------------------------------


def highpass_filter(
        input_layer,
        a: float = 8.0,
        b: float = 4.0):
    x = input_layer
    x_focus = tf.math.pow(tf.nn.tanh(a * x), b)
    return x_focus * x


# ---------------------------------------------------------------------


def lowpass_filter(
        input_layer,
        a: float = 8.0,
        b: float = 4.0):
    x = input_layer
    x_focus = 1.0 - tf.math.pow(tf.nn.tanh(a * x), b)
    return x_focus * x


# ---------------------------------------------------------------------

def multiscales_generator_fn(
        shape: List[int],
        no_scales: int,
        kernel_size: Tuple[int, int] = (4, 4),
        nsig: Tuple[float, float] = (1, 1),
        clip_values: bool = False,
        round_values: bool = False,
        normalize_values: bool = False,
        concrete_functions: bool = False,
        jit_compile: bool = False):
    kernel = (
        tf.constant(
            depthwise_gaussian_kernel(
                channels=shape[-1],
                kernel_size=kernel_size,
                nsig=nsig).astype("float32")))

    def multiscale_fn(n: tf.Tensor) -> List[tf.Tensor]:
        n_scale = n
        scales = [n_scale]

        for _ in range(no_scales):
            # downsample, clip and round
            # n_scale = tf.nn.depthwise_conv2d(
            #     input=n_scale,
            #     filter=kernel,
            #     strides=(1, 2, 2, 1),
            #     data_format=None,
            #     dilations=None,
            #     padding="VALID")
            n_scale = tf.nn.avg_pool2d(input=n_scale, ksize=(2, 2), padding="VALID", strides=(2, 2))
            # clip values
            if clip_values:
                n_scale = tf.clip_by_value(n_scale,
                                           clip_value_min=0.0,
                                           clip_value_max=255.0)
            # round values
            if round_values:
                n_scale = tf.round(n_scale)
            # normalize (sum of channel dim equals 1)
            if normalize_values:
                n_scale += DEFAULT_EPSILON
                n_scale = \
                    n_scale / \
                    tf.reduce_sum(n_scale, axis=-1, keepdims=True)
            scales.append(n_scale)

        return scales

    result = tf.function(
        func=multiscale_fn,
        input_signature=[
            tf.TensorSpec(shape=shape, dtype=tf.float32),
        ],
        jit_compile=jit_compile,
        reduce_retracing=True)

    if concrete_functions:
        return result.get_concrete_function()

    return result


# ---------------------------------------------------------------------


def create_checkpoint(
        step: tf.Variable = tf.Variable(0, trainable=False, dtype=tf.dtypes.int64, name="step"),
        epoch: tf.Variable = tf.Variable(0, trainable=False, dtype=tf.dtypes.int64, name="epoch"),
        model: tf.keras.Model = None,
        path: Union[str, Path] = None) -> tf.train.Checkpoint:
    # define common checkpoint
    ckpt = \
        tf.train.Checkpoint(
            step=step,
            epoch=epoch,
            model=model)
    # if paths exists load latest
    if path is not None:
        if os.path.isdir(str(path)):
            ckpt.restore(tf.train.latest_checkpoint(str(path))).expect_partial()
    return ckpt


# ---------------------------------------------------------------------


def save_config(
        config: Union[str, Dict, Path],
        filename: Union[str, Path]) -> None:
    """
    save configuration to target filename

    :param config: dict configuration or path to json configuration
    :param filename: output filename
    :return: nothing if success, exception if failed
    """
    # --- argument checking
    config = load_config(config)
    if not filename:
        raise ValueError("filename cannot be null or empty")

    # --- log
    logger.info(f"saving configuration pipeline to [{str(filename)}]")

    # --- dump config to filename
    with open(filename, "w") as f:
        return json.dump(obj=config, fp=f, indent=4)

# ---------------------------------------------------------------------

def pad_to_power_of_2(image: tf.Tensor):
    shape = tf.shape(image)
    height = shape[1]
    width = shape[2]
    desired_height = tf.pow(2, tf.cast(tf.math.ceil(tf.math.log(tf.cast(height, dtype=tf.float32)) / tf.math.log(2.0)), tf.int32))
    desired_width = tf.pow(2, tf.cast(tf.math.ceil(tf.math.log(tf.cast(width, dtype=tf.float32)) / tf.math.log(2.0)), tf.int32))
    padding_height = desired_height - height
    padding_width = desired_width - width
    paddings = [
        [0, 0],
        [0, padding_height],
        [0, padding_width],
        [0, 0]
    ]
    padded_image = tf.pad(image, paddings, mode='CONSTANT')
    return padded_image, padding_height, padding_width

# ---------------------------------------------------------------------

def remove_padding(padded_image:tf.Tensor,
                   padding_height,
                   padding_width):
    shape = tf.shape(padded_image)
    height = shape[1]
    width = shape[2]
    return tf.slice(
        padded_image,
        [0, 0, 0, 0],
        [-1, height - padding_height, width - padding_width, -1])

# ---------------------------------------------------------------------

def find_layer_by_name(model: tf.keras.Model, layer_name: str):
    """
    Recursively finds a layer by name in a Keras model or sub-model.

    Args:
    model (tf.keras.Model): The Keras model.
    layer_name (str): The name of the layer to find.

    Returns:
    tf.keras.layers.Layer: The layer with the specified name, if found.
    None: If the layer does not exist.
    """
    # Check if the model has the 'layers' attribute
    if not hasattr(model, 'layers'):
        logger.info("The provided model does not have a 'layers' attribute.")
        return None

    # Iterate through the layers of the model
    for layer in model.layers:
        if layer.name == layer_name:
            return layer
        # If the layer is a model or a layer that may contain sub-layers, search recursively
        if isinstance(layer, tf.keras.Model) or hasattr(layer, 'layers'):
            sub_layer = find_layer_by_name(layer, layer_name)
            if sub_layer:
                return sub_layer
    return None

# ---------------------------------------------------------------------

def get_layer_output(model: tf.keras.Model, layer_name: str, input_data):
    """
    Gets the output of a specified layer in a Keras model.

    Args:
    model (tf.keras.Model): The Keras model.
    layer_name (str): The name of the layer to get the output from.
    input_data: The input data to pass through the model.

    Returns:
    np.ndarray: The output of the specified layer.
    None: If the layer does not exist.
    """
    # Find the specified layer
    layer = find_layer_by_name(model, layer_name)
    if layer is None:
        print(f"Layer with name '{layer_name}' does not exist.")
        return None

    # Create a new model that outputs the desired layer's output
    intermediate_model = tf.keras.Model(inputs=model.inputs, outputs=layer.output)

    # Get the output of the layer
    return intermediate_model.predict(input_data)

# ---------------------------------------------------------------------
