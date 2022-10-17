import abc
import tensorflow as tf
from tensorflow import keras
from collections import namedtuple
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .custom_logger import logger
from .utilities import \
    conv2d_wrapper, \
    input_shape_fixer, \
    build_normalize_model, \
    build_denormalize_model
from .backbone_unet import builder as builder_unet
from .backbone_lunet import builder as builder_lunet
from .backbone_resnet import builder as builder_resnet
from .backbone_resnet_ce import builder as builder_resnet_ce
from .pyramid import \
    build_pyramid_model, \
    build_inverse_pyramid_model
from .constants import *
from .regularizer import builder as regularizer_builder

# ---------------------------------------------------------------------

BuilderResults = namedtuple(
    "BuilderResults",
    {
        "pyramid",
        "superres",
        "denoiser",
        "backbone",
        "normalizer",
        "denormalizer",
        "superres_head",
        "denoiser_head",
        "inverse_pyramid",
    })

# ---------------------------------------------------------------------


def model_builder(
        config: Dict) -> BuilderResults:
    """
    Reads a configuration and returns 6 models,

    :param config: configuration dictionary
    :return:
        denoiser model,
        normalize model,
        denormalize model,
        pyramid model,
        denoiser_head model,
        inverse pyramid model
    """
    logger.info("building model with config [{0}]".format(config))

    # --- argument parsing
    model_type = config[TYPE_STR]
    filters = config.get("filters", 32)
    groups = config.get("groups", 4)
    no_levels = config.get("no_levels", 1)
    add_var = config.get("add_var", False)
    no_layers = config.get("no_layers", 5)
    min_value = config.get("min_value", 0)
    max_value = config.get("max_value", 255)
    use_bias = config.get("use_bias", False)
    batchnorm = config.get("batchnorm", True)
    kernel_size = config.get("kernel_size", 3)
    add_gates = config.get("add_gates", False)
    pyramid_config = config.get("pyramid", None)
    dropout_rate = config.get("dropout_rate", -1)
    activation = config.get("activation", "relu")
    clip_values = config.get("clip_values", True)
    channel_index = config.get("channel_index", 2)
    add_final_bn = config.get("add_final_bn", True)
    add_selector = config.get("add_selector", False)
    shared_model = config.get("shared_model", False)
    add_sparsity = config.get("add_sparsity", False)
    add_laplacian = config.get("add_laplacian", True)
    stop_gradient = config.get("stop_gradient", False)
    add_initial_bn = config.get("add_initial_bn", False)
    add_concat_input = config.get("add_concat_input", False)
    input_shape = config.get("input_shape", (None, None, 3))
    output_multiplier = config.get("output_multiplier", 1.0)
    final_activation = config.get("final_activation", "linear")
    kernel_regularizer = config.get("kernel_regularizer", "l1")
    backbone_activation = config.get("backbone_activation", None)
    add_skip_with_input = config.get("add_skip_with_input", True)
    add_channelwise_scaling = config.get("add_channelwise_scaling", False)
    add_sparse_features = config.get("add_sparse_features", False)
    kernel_initializer = config.get("kernel_initializer", "glorot_normal")
    add_learnable_multiplier = config.get("add_learnable_multiplier", False)
    final_kernel_regularization = config.get("final_kernel_regularization", "l1")
    add_residual_between_models = config.get("add_residual_between_models", False)
    add_mean_sigma_normalization = config.get("add_mean_sigma_normalization", False)

    use_pyramid = pyramid_config is not None
    input_shape = input_shape_fixer(input_shape)

    # --- argument checking
    if no_levels <= 0:
        raise ValueError("no_levels must be > 0")
    if filters <= 0:
        raise ValueError("filters must be > 0")
    if kernel_size <= 0:
        raise ValueError("kernel_size must be > 0")

    # regularizer for all kernels above the final ones
    kernel_regularizer = \
        regularizer_builder(kernel_regularizer)

    # regularizer for the final kernel
    final_kernel_regularization = \
        regularizer_builder(final_kernel_regularization)

    denoise_intermediate_conv_params = dict(
        groups=groups,
        kernel_size=1,
        strides=(1, 1),
        padding="same",
        use_bias=use_bias,
        activation=activation,
        filters=input_shape[channel_index] * groups,
        kernel_regularizer=kernel_regularizer,
        kernel_initializer=kernel_initializer
    )

    denoise_final_conv_params = dict(
        kernel_size=1,
        strides=(1, 1),
        padding="same",
        use_bias=use_bias,
        # this must be linear because it is capped later
        activation="linear",
        groups=input_shape[channel_index],
        filters=input_shape[channel_index],
        kernel_regularizer=final_kernel_regularization,
        kernel_initializer=kernel_initializer
    )

    residual_conv_params = dict(
        kernel_size=3,
        strides=(1, 1),
        padding="same",
        use_bias=use_bias,
        # this must be linear because it is capped later
        activation="linear",
        filters=input_shape[channel_index],
        kernel_regularizer=kernel_regularizer,
        kernel_initializer=kernel_initializer
    )

    channelwise_params = dict(
        multiplier=1.0,
        regularizer=keras.regularizers.L1(DEFAULT_CHANNELWISE_MULTIPLIER_L1),
        trainable=True,
        activation="relu"
    )

    superres_expand_conv_params = dict(
        groups=groups,
        kernel_size=3,
        padding="valid",
        use_bias=use_bias,
        dilation_rate=(2, 2),
        activation=activation,
        filters=filters * groups,
        kernel_regularizer=kernel_regularizer,
        kernel_initializer=kernel_initializer
    )

    superres_intermediate_conv_params = dict(
        groups=groups,
        kernel_size=3,
        strides=(1, 1),
        padding="same",
        use_bias=use_bias,
        activation=activation,
        filters=input_shape[channel_index] * groups,
        kernel_regularizer=kernel_regularizer,
        kernel_initializer=kernel_initializer
    )

    superres_final_conv_params = dict(
        kernel_size=1,
        strides=(1, 1),
        padding="same",
        use_bias=use_bias,
        # this must be linear because it is capped later
        activation="linear",
        groups=input_shape[channel_index],
        filters=input_shape[channel_index],
        kernel_regularizer=final_kernel_regularization,
        kernel_initializer=kernel_initializer
    )

    # --- build normalize denormalize models
    model_normalize = \
        build_normalize_model(
            input_dims=(None, None, None),
            min_value=min_value,
            max_value=max_value)

    model_denormalize = \
        build_denormalize_model(
            input_dims=(None, None, None),
            min_value=min_value,
            max_value=max_value)

    # --- build denoise model
    model_params = dict(
        add_var=add_var,
        filters=filters,
        use_bn=batchnorm,
        no_levels=no_levels,
        no_layers=no_layers,
        add_gates=add_gates,
        activation=activation,
        input_dims=input_shape,
        kernel_size=kernel_size,
        add_selector=add_selector,
        add_sparsity=add_sparsity,
        dropout_rate=dropout_rate,
        add_final_bn=add_final_bn,
        add_laplacian=add_laplacian,
        stop_gradient=stop_gradient,
        add_initial_bn=add_initial_bn,
        add_concat_input=add_concat_input,
        kernel_regularizer=kernel_regularizer,
        kernel_initializer=kernel_initializer,
        add_sparse_features=add_sparse_features,
        add_channelwise_scaling=add_channelwise_scaling,
        add_learnable_multiplier=add_learnable_multiplier,
        add_mean_sigma_normalization=add_mean_sigma_normalization,
    )

    if model_type == "unet":
        backbone_builder = builder_unet
    elif model_type == "lunet":
        backbone_builder = builder_lunet
    elif model_type == "resnet":
        backbone_builder = builder_resnet
    elif model_type == "resnet_ce":
        backbone_builder = builder_resnet_ce
    else:
        raise ValueError(
            "don't know how to build model [{0}]".format(model_type))

    # --- build pyramid / inverse pyramid
    if use_pyramid:
        logger.info(f"building pyramid: [{pyramid_config}]")
        model_pyramid = \
            build_pyramid_model(
                input_dims=input_shape,
                config=pyramid_config)

        logger.info(f"building inverse pyramid: [{pyramid_config}]")
        model_inverse_pyramid = \
            build_inverse_pyramid_model(
                input_dims=input_shape,
                config=pyramid_config)
    else:
        model_pyramid = None
        model_inverse_pyramid = None

    # --- connect the parts of the model
    # setup input
    input_layer = \
        keras.Input(
            shape=input_shape,
            name="input_tensor")
    x = input_layer

    # --- run inference
    if use_pyramid:
        x_levels = model_pyramid(x, training=False)
    else:
        x_levels = [x]

    logger.info("pyramid produces [{0}] scales".format(len(x_levels)))

    # --- shared or separate models
    if shared_model:
        logger.info("building shared model")
        backbone_model = \
            backbone_builder(
                name="level_shared",
                **model_params)
        backbone_models = [backbone_model] * len(x_levels)
    else:
        logger.info("building per scale model")
        backbone_models = [
            backbone_builder(
                name=f"level_{i}",
                **model_params)
            for i in range(len(x_levels))
        ]
    upsampling_params = \
        dict(size=(2, 2),
             interpolation="bilinear")

    if add_residual_between_models:
        previous_level = None
        for i, x_level in reversed(list(enumerate(x_levels))):
            if previous_level is None:
                current_level_output = backbone_models[i](x_level)
            else:
                previous_level = \
                    keras.layers.UpSampling2D(
                        **upsampling_params)(previous_level)
                previous_level = \
                    conv2d_wrapper(
                        input_layer=previous_level,
                        conv_params=residual_conv_params,
                        channelwise_scaling=True)
                current_level_input = \
                    keras.layers.Add()([previous_level, x_level])
                current_level_output = backbone_models[i](current_level_input)
            previous_level = current_level_output
            x_levels[i] = current_level_output
    else:
        for i, x_level in enumerate(x_levels):
            x_levels[i] = backbone_models[i](x_level)

    # --- optional multiplier to help saturation
    if output_multiplier is not None and \
            output_multiplier != 1:
        x_levels = [
            x_level * output_multiplier
            for x_level in x_levels
        ]

    # --- clip levels to [-0.5, +0.5]
    if clip_values:
        for i, x_level in enumerate(x_levels):
            x_levels[i] = \
                tf.clip_by_value(
                    t=x_level,
                    clip_value_min=-0.5,
                    clip_value_max=+0.5)

    # --- merge levels together
    if use_pyramid:
        x_backbone_output = \
            model_inverse_pyramid(x_levels)
    else:
        x_backbone_output = x_levels[0]

    if backbone_activation is not None:
        x_backbone_output = \
            tf.keras.layers.Activation(
                backbone_activation)(x_backbone_output)

    # --- keep model before projection to output
    model_backbone = \
        keras.Model(
            inputs=input_layer,
            outputs=x_backbone_output,
            name=f"{model_type}_backbone")

    # --- define denoiser here
    denoise_input_layer = \
        keras.Input(
            shape=(None, None, filters),
            name="input_tensor")
    x = denoise_input_layer

    x = \
        conv2d_wrapper(
            input_layer=x,
            bn_params=None,
            conv_params=denoise_intermediate_conv_params,
            channelwise_scaling=channelwise_params)

    x = \
        conv2d_wrapper(
            input_layer=x,
            bn_params=None,
            conv_params=denoise_final_conv_params,
            channelwise_scaling=False)

    # cap it off to limit values
    x_result = \
        tf.keras.layers.Activation(
            name="output_tensor",
            activation=final_activation)(x)

    if add_skip_with_input:
        x_result = \
            tf.keras.layers.Substract(
                name="skip_input")([x_result, input_layer])

    # wrap and name denoiser head
    model_denoiser_head = \
        keras.Model(
            inputs=denoise_input_layer,
            outputs=x_result,
            name=f"denoiser_head")

    # wrap and name denoiser
    model_denoiser = \
        keras.Model(
            inputs=input_layer,
            outputs=model_denoiser_head(x_backbone_output),
            name=f"{model_type}_denoiser")

    # --- define super resolution here
    superres_input_layer = \
        keras.Input(
            shape=(None, None, filters),
            name="input_tensor")
    x = superres_input_layer

    x = \
        conv2d_wrapper(
            input_layer=x,
            bn_params=None,
            conv_params=superres_expand_conv_params,
            channelwise_scaling=channelwise_params)

    x = \
        conv2d_wrapper(
            input_layer=x,
            bn_params=None,
            conv_params=superres_intermediate_conv_params,
            channelwise_scaling=channelwise_params)

    x = \
        conv2d_wrapper(
            input_layer=x,
            bn_params=None,
            conv_params=superres_final_conv_params,
            channelwise_scaling=channelwise_params)

    # cap it off to limit values
    x_result = \
        tf.keras.layers.Activation(
            name="output_tensor",
            activation=final_activation)(x)

    # wrap and name superres head
    model_superres_head = \
        keras.Model(
            inputs=superres_input_layer,
            outputs=x_result,
            name=f"superres_head")

    # wrap and name superres
    model_superres = \
        keras.Model(
            inputs=input_layer,
            outputs=model_superres_head(x_backbone_output),
            name=f"{model_type}_superres")

    # ---
    return \
        BuilderResults(
            pyramid=model_pyramid,
            denoiser=model_denoiser,
            superres=model_superres,
            backbone=model_backbone,
            normalizer=model_normalize,
            denormalizer=model_denormalize,
            denoiser_head=model_denoiser_head,
            superres_head=model_superres_head,
            inverse_pyramid=model_inverse_pyramid)


# ---------------------------------------------------------------------


class DenoisingInferenceModule(tf.Module, abc.ABC):
    """denoising inference module."""

    def __init__(
            self,
            model_denoise: keras.Model,
            model_normalize: keras.Model,
            model_denormalize: keras.Model,
            training_channels: int = 1,
            cast_to_uint8: bool = True):
        """
        Initializes a module for denoising.

        :param model_denoise: denoising model to use for inference.
        :param model_normalize: model that normalizes the input
        :param model_denormalize: model that denormalizes the output
        :param training_channels: how many color channels were used in training
        :param cast_to_uint8: cast output to uint8

        """
        # --- argument checking
        if model_denoise is None:
            raise ValueError("model_denoise should not be None")
        if model_normalize is None:
            raise ValueError("model_normalize should not be None")
        if model_denormalize is None:
            raise ValueError("model_denormalize should not be None")
        if training_channels <= 0:
            raise ValueError("training channels should be > 0")

        # --- setup instance variables
        self._cast_to_uint8 = cast_to_uint8
        self._model_denoise = model_denoise
        self._model_normalize = model_normalize
        self._model_denormalize = model_denormalize
        self._training_channels = training_channels

    def _run_inference_on_images(self, image):
        """
        Cast image to float and run inference.

        :param image: uint8 Tensor of shape
        :return: denoised image: uint8 Tensor of shape if the input
        """
        x = tf.cast(image, dtype=tf.float32)

        # --- normalize
        x = self._model_normalize(x)

        # --- run denoise model
        x = self._model_denoise(x)

        # --- denormalize
        x = self._model_denormalize(x)

        # --- cast to uint8
        if self._cast_to_uint8:
            x = tf.round(x)
            x = tf.cast(x, dtype=tf.uint8)

        return x

    @abc.abstractmethod
    def __call__(self, input_tensor):
        pass

# ---------------------------------------------------------------------


class DenoisingInferenceModule1Channel(DenoisingInferenceModule):
    def __init__(
            self,
            model_denoise: keras.Model = None,
            model_normalize: keras.Model = None,
            model_denormalize: keras.Model = None,
            cast_to_uint8: bool = True):
        super().__init__(
            model_denoise=model_denoise,
            model_normalize=model_normalize,
            model_denormalize=model_denormalize,
            training_channels=1,
            cast_to_uint8=cast_to_uint8)

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=[None, None, None, 1], dtype=tf.uint8)])
    def __call__(self, input_tensor):
        return self._run_inference_on_images(input_tensor)


# ---------------------------------------------------------------------


class DenoisingInferenceModule3Channel(DenoisingInferenceModule):
    def __init__(
            self,
            model_denoise: keras.Model = None,
            model_normalize: keras.Model = None,
            model_denormalize: keras.Model = None,
            cast_to_uint8: bool = True):
        super().__init__(
            model_denoise=model_denoise,
            model_normalize=model_normalize,
            model_denormalize=model_denormalize,
            training_channels=3,
            cast_to_uint8=cast_to_uint8)

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=[None, None, None, 3], dtype=tf.uint8)])
    def __call__(self, input_tensor):
        return self._run_inference_on_images(input_tensor)


# ---------------------------------------------------------------------


def module_builder(
        model_denoise: keras.Model = None,
        model_normalize: keras.Model = None,
        model_denormalize: keras.Model = None,
        cast_to_uint8: bool = True) -> DenoisingInferenceModule:
    """
    builds a module for denoising.

    :param model_denoise: denoising model to use for inference.
    :param model_normalize: model that normalizes the input
    :param model_denormalize: model that denormalizes the output
    :param cast_to_uint8: cast output to uint8

    :return: denoiser module
    """
    logger.info(
        f"building denoising module with "
        f"cast_to_uint8:{cast_to_uint8}")

    # --- argument checking
    # TODO

    training_channels = \
        model_denoise.input_shape[-1]

    if training_channels == 1:
        return \
            DenoisingInferenceModule1Channel(
                model_denoise=model_denoise,
                model_normalize=model_normalize,
                model_denormalize=model_denormalize,
                cast_to_uint8=cast_to_uint8)
    elif training_channels == 3:
        return \
            DenoisingInferenceModule3Channel(
                model_denoise=model_denoise,
                model_normalize=model_normalize,
                model_denormalize=model_denormalize,
                cast_to_uint8=cast_to_uint8)
    else:
        raise ValueError(
            "don't know how to handle training_channels:{0}".format(training_channels))

# ---------------------------------------------------------------------
