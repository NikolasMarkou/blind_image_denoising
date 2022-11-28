import abc
import keras
import tensorflow as tf
from typing import Dict, Tuple
from collections import namedtuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .constants import *
from .custom_logger import logger
from .utilities import \
    clip_normalized_tensor, \
    conv2d_wrapper, \
    input_shape_fixer, \
    build_normalize_model, \
    build_denormalize_model
from .backbone_unet import builder as builder_unet
from .backbone_lunet import builder as builder_lunet
from .backbone_resnet import builder as builder_resnet
from .pyramid import \
    build_pyramid_model, \
    build_inverse_pyramid_model
from .regularizer import \
    builder as regularizer_builder

# ---------------------------------------------------------------------

BuilderResults = namedtuple(
    "BuilderResults",
    {
        "backbone",
        "normalizer",
        "denormalizer",
        "denoiser",
        "inpaint",
        "superres",
        "hydra"
    })


def model_builder(
        config: Dict) -> BuilderResults:
    # --- build backbone
    model_backbone, model_normalizer, model_denormalizer = \
        model_backbone_builder(config=config["backbone"])

    # --- build denoiser, inpaint, superres
    model_denoiser = \
        model_denoiser_builder(config=config["denoiser"])
    model_inpaint = \
        model_inpaint_builder(config=config["inpaint"])
    model_superres = \
        model_superres_builder(config=config["superres"])

    input_shape = tf.keras.backend.int_shape(model_backbone.inputs[0])[1:]
    logger.info("input_shape: [{0}]".format(input_shape))

    # --- build hydra combined model
    input_layer = tf.keras.Input(shape=input_shape, name="input_tensor")
    input_normalized_layer = model_normalizer(input_layer, training=False)
    mask_layer = tf.keras.Input(shape=input_shape, name="mask_input_tensor")
    #
    backbone_output = model_backbone(input_normalized_layer)
    denoiser_output = model_denoiser(backbone_output)
    inpaint_output = model_inpaint([backbone_output, input_normalized_layer, mask_layer])
    superres_output = model_superres(backbone_output)
    #
    backbone_output = model_denormalizer(backbone_output, training=False)
    denoiser_output = model_denormalizer(denoiser_output, training=False)
    inpaint_output = model_denormalizer(inpaint_output, training=False)
    superres_output = model_denormalizer(superres_output, training=False)

    # wrap layers to set names
    backbone_output = tf.keras.layers.Layer(name="backbone")(backbone_output)
    denoiser_output = tf.keras.layers.Layer(name="denoiser")(denoiser_output)
    inpaint_output = tf.keras.layers.Layer(name="inpaint")(inpaint_output)
    superres_output = tf.keras.layers.Layer(name="superres")(superres_output)

    # create model
    model_hydra = \
        keras.Model(
            inputs=[input_layer, mask_layer],
            outputs=[
                backbone_output,
                denoiser_output,
                inpaint_output,
                superres_output
            ],
            name=f"hydra")

    # --- pack results
    return \
        BuilderResults(
            backbone=model_backbone,
            normalizer=model_normalizer,
            denormalizer=model_denormalizer,
            denoiser=model_denoiser,
            inpaint=model_inpaint,
            superres=model_superres,
            hydra=model_hydra
        )

# ---------------------------------------------------------------------


def model_backbone_builder(
        config: Dict) -> Tuple[tf.keras.Model,
                               tf.keras.Model,
                               tf.keras.Model]:
    """
    reads a configuration a model backbone

    :param config: configuration dictionary
    :return: backbone, normalizer, denormalizer
    """
    logger.info("building backbone model with config [{0}]".format(config))

    # --- argument parsing
    model_type = config[TYPE_STR]
    filters = config.get("filters", 32)
    no_levels = config.get("no_levels", 1)
    add_var = config.get("add_var", False)
    no_layers = config.get("no_layers", 5)
    add_gelu = config.get("add_gelu", False)
    use_bias = config.get("use_bias", False)
    batchnorm = config.get("batchnorm", True)
    kernel_size = config.get("kernel_size", 3)
    add_gates = config.get("add_gates", False)
    pyramid_config = config.get("pyramid", None)
    dropout_rate = config.get("dropout_rate", -1)
    activation = config.get("activation", "relu")
    add_squash = config.get("add_squash", False)
    clip_values = config.get("clip_values", False)
    channel_index = config.get("channel_index", 2)
    add_final_bn = config.get("add_final_bn", False)
    add_selector = config.get("add_selector", False)
    shared_model = config.get("shared_model", False)
    add_sparsity = config.get("add_sparsity", False)
    value_range = config.get("value_range", (0, 255))
    add_laplacian = config.get("add_laplacian", True)
    stop_gradient = config.get("stop_gradient", False)
    block_kernels = config.get("block_kernels", (3, 3))
    block_filters = config.get("block_filters", (32, 32))
    block_depthwise = config.get("block_depthwise", None)
    add_initial_bn = config.get("add_initial_bn", False)
    add_concat_input = config.get("add_concat_input", False)
    input_shape = config.get("input_shape", (None, None, 3))
    output_multiplier = config.get("output_multiplier", 1.0)
    kernel_regularizer = config.get("kernel_regularizer", "l1")
    backbone_activation = config.get("backbone_activation", None)
    add_sparse_features = config.get("add_sparse_features", False)
    add_channelwise_scaling = config.get("add_channelwise_scaling", False)
    kernel_initializer = config.get("kernel_initializer", "glorot_normal")
    add_learnable_multiplier = config.get("add_learnable_multiplier", False)
    add_residual_between_models = config.get("add_residual_between_models", False)
    add_mean_sigma_normalization = config.get("add_mean_sigma_normalization", False)

    min_value = value_range[0]
    max_value = value_range[1]
    use_pyramid = pyramid_config is not None
    input_shape = input_shape_fixer(input_shape)

    # --- argument checking
    if no_levels <= 0:
        raise ValueError("no_levels must be > 0")
    if filters <= 0:
        raise ValueError("filters must be > 0")
    if kernel_size <= 0:
        raise ValueError("kernel_size must be > 0")

    # regularizer for all kernels in the backbone
    kernel_regularizer = \
        regularizer_builder(kernel_regularizer)

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
        add_gelu=add_gelu,
        no_levels=no_levels,
        no_layers=no_layers,
        add_gates=add_gates,
        activation=activation,
        add_squash=add_squash,
        input_dims=input_shape,
        kernel_size=kernel_size,
        add_selector=add_selector,
        add_sparsity=add_sparsity,
        dropout_rate=dropout_rate,
        add_final_bn=add_final_bn,
        block_kernels=block_kernels,
        block_filters=block_filters,
        block_depthwise=block_depthwise,
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

    # --- residual between model
    if add_residual_between_models:
        upsampling_params = \
            dict(size=(2, 2),
                 interpolation="nearest")
        bn_params = None
        if batchnorm:
            bn_params = dict(
                scale=True,
                center=False,
                momentum=DEFAULT_BN_MOMENTUM,
                epsilon=DEFAULT_BN_EPSILON)
        residual_conv_params = dict(
            kernel_size=3,
            padding="same",
            strides=(1, 1),
            use_bias=use_bias,
            activation="tanh",
            filters=input_shape[channel_index],
            kernel_regularizer=kernel_regularizer,
            kernel_initializer=kernel_initializer)
        previous_level = None
        for i, x_level in reversed(list(enumerate(x_levels))):
            if previous_level is None:
                current_level_output = backbone_models[i](x_level)
            else:
                # NOTE
                # based on https://distill.pub/2016/deconv-checkerboard/
                # it is better to upsample with nearest neighbor and then conv2d
                if batchnorm:
                    previous_level = \
                        tf.keras.layers.BatchNormalization(**bn_params)(previous_level)
                previous_level = \
                    tf.keras.layers.UpSampling2D(
                        **upsampling_params)(previous_level)
                previous_level = \
                    tf.keras.layers.Concatenate()([previous_level, x_level])
                previous_level = \
                    conv2d_wrapper(
                        input_layer=previous_level,
                        conv_params=residual_conv_params,
                        channelwise_scaling=True,
                        multiplier_scaling=False)
                previous_level = clip_normalized_tensor(previous_level)
                previous_level = \
                    tf.keras.layers.Add()([previous_level, x_level])
                current_level_input = clip_normalized_tensor(previous_level)
                current_level_output = backbone_models[i](current_level_input)
            previous_level = current_level_output
            x_levels[i] = current_level_output
    else:
        for i, x_level in enumerate(x_levels):
            x_levels[i] = backbone_models[i](x_level)

    # --- optional multiplier to help saturation
    if output_multiplier is not None and \
            output_multiplier != 1.0:
        x_levels = [
            x_level * output_multiplier
            for x_level in x_levels
        ]

    # --- clip levels
    if clip_values:
        for i, x_level in enumerate(x_levels):
            x_levels[i] = clip_normalized_tensor(x_level)

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

    # ---
    return \
        model_backbone, \
        model_normalize, \
        model_denormalize


# ---------------------------------------------------------------------


def model_denoiser_builder(
        config: Dict,
        **kwargs):
    """
    builds the denoiser model on top of the backbone layer

    :param config: dictionary with the denoiser configuration

    :return: denoiser head model
    """
    # --- argument checking
    logger.info(f"building denoiser model with [{config}]")
    logger.info(f"unused parameters [{kwargs}]")

    # --- set configuration
    filters = config.get("filters", 32)
    use_bias = config.get("use_bias", False)
    batchnorm = config.get("batchnorm", True)
    kernel_size = config.get("kernel_size", 5)
    activation = config.get("activation", "relu")
    output_channels = config.get("output_channels", 3)
    input_shape = input_shape_fixer(config.get("input_shape"))
    final_activation = config.get("final_activation", "tanh")
    kernel_initializer = config.get("kernel_initializer", "glorot_normal")
    kernel_regularizer = regularizer_builder(config.get("kernel_regularizer", "l2"))

    bn_params = dict(
        scale=True,
        center=use_bias,
        momentum=DEFAULT_BN_MOMENTUM,
        epsilon=DEFAULT_BN_EPSILON)

    start_conv_params = dict(
        kernel_size=1,
        strides=(1, 1),
        padding="same",
        filters=filters,
        use_bias=use_bias,
        activation=activation,
        kernel_regularizer=kernel_regularizer,
        kernel_initializer=kernel_initializer
    )

    middle_conv_params = dict(
        kernel_size=kernel_size,
        strides=(1, 1),
        padding="same",
        filters=filters,
        use_bias=use_bias,
        activation=activation,
        kernel_regularizer=kernel_regularizer,
        kernel_initializer=kernel_initializer
    )

    final_conv_params = dict(
        kernel_size=1,
        strides=(1, 1),
        padding="same",
        use_bias=use_bias,
        filters=output_channels,
        activation=final_activation,
        kernel_regularizer=kernel_regularizer,
        kernel_initializer=kernel_initializer
    )

    # --- define superres head here
    model_input_layer = \
        tf.keras.Input(
            shape=input_shape,
            name="input_tensor")

    x = model_input_layer

    x = \
        conv2d_wrapper(
            input_layer=x,
            bn_params=None,
            conv_params=start_conv_params,
            channelwise_scaling=False,
            multiplier_scaling=False)

    if batchnorm:
        x = \
            tf.keras.layers.BatchNormalization(
                **bn_params)(x)

    x = \
        conv2d_wrapper(
            input_layer=x,
            bn_params=None,
            conv_params=middle_conv_params,
            channelwise_scaling=False,
            multiplier_scaling=False)

    x = \
        conv2d_wrapper(
            input_layer=x,
            bn_params=None,
            conv_params=final_conv_params,
            channelwise_scaling=False,
            multiplier_scaling=False)

    x_result = \
        tf.keras.layers.Layer(
            name="output_tensor")(x)

    model_head = \
        tf.keras.Model(
            inputs=model_input_layer,
            outputs=x_result,
            name=f"denoiser_head")

    return model_head


# ---------------------------------------------------------------------


def model_inpaint_builder(
        config: Dict,
        **kwargs):
    """
    builds the inpaint model on top of the backbone layer

    :param config: dictionary with the inpaint configuration

    :return: inpaint head model
    """
    # --- argument checking
    logger.info(f"building inpaint model with [{config}]")
    logger.info(f"unused parameters [{kwargs}]")

    # --- set configuration
    filters = config.get("filters", 32)
    use_bias = config.get("use_bias", False)
    batchnorm = config.get("batchnorm", True)
    kernel_size = config.get("kernel_size", 5)
    activation = config.get("activation", "relu")
    output_channels = config.get("output_channels", 3)
    input_shape = input_shape_fixer(config.get("input_shape"))
    final_activation = config.get("final_activation", "tanh")
    kernel_initializer = config.get("kernel_initializer", "glorot_normal")
    kernel_regularizer = regularizer_builder(config.get("kernel_regularizer", "l2"))

    bn_params = dict(
        scale=True,
        center=use_bias,
        momentum=DEFAULT_BN_MOMENTUM,
        epsilon=DEFAULT_BN_EPSILON)

    start_conv_params = dict(
        kernel_size=1,
        strides=(1, 1),
        padding="same",
        filters=filters,
        use_bias=use_bias,
        activation=activation,
        kernel_regularizer=kernel_regularizer,
        kernel_initializer=kernel_initializer
    )

    middle_conv_params = dict(
        kernel_size=kernel_size,
        strides=(1, 1),
        padding="same",
        filters=filters,
        use_bias=use_bias,
        activation=activation,
        kernel_regularizer=kernel_regularizer,
        kernel_initializer=kernel_initializer
    )

    final_conv_params = dict(
        kernel_size=1,
        strides=(1, 1),
        padding="same",
        use_bias=use_bias,
        filters=output_channels,
        activation=final_activation,
        kernel_regularizer=kernel_regularizer,
        kernel_initializer=kernel_initializer
    )

    # --- define superres head here
    model_input_layer = \
        tf.keras.Input(
            shape=input_shape,
            name="input_tensor")
    original_input_layer = \
        tf.keras.Input(
            shape=(None, None, output_channels),
            name="original_input_tensor")
    mask_input_layer = \
        tf.keras.Input(
            shape=(None, None, output_channels),
            name="mask_input_tensor")

    x = model_input_layer

    x = \
        conv2d_wrapper(
            input_layer=x,
            bn_params=None,
            conv_params=start_conv_params,
            channelwise_scaling=False,
            multiplier_scaling=False)

    if batchnorm:
        x = \
            tf.keras.layers.BatchNormalization(
                **bn_params)(x)

    x = \
        conv2d_wrapper(
            input_layer=x,
            bn_params=None,
            conv_params=middle_conv_params,
            channelwise_scaling=False,
            multiplier_scaling=False)

    x = \
        conv2d_wrapper(
            input_layer=x,
            bn_params=None,
            conv_params=final_conv_params,
            channelwise_scaling=False,
            multiplier_scaling=False)

    x = \
        tf.keras.layers.Multiply()([x, 1.0 - mask_input_layer]) + \
        tf.keras.layers.Multiply()([original_input_layer, mask_input_layer])

    x_result = \
        tf.keras.layers.Layer(
            name="output_tensor")(x)

    model_head = \
        tf.keras.Model(
            inputs=[
                model_input_layer,
                original_input_layer,
                mask_input_layer
            ],
            outputs=x_result,
            name=f"inpaint_head")

    return model_head


# ---------------------------------------------------------------------


def model_superres_builder(
        config: Dict,
        **kwargs) -> tf.keras.Model:
    """
    builds the superres model on top of the backbone layer

    :param config: dictionary with the superres configuration

    :return: superres head model
    """
    # --- argument checking
    logger.info(f"building superres model with [{config}]")
    logger.info(f"unused parameters [{kwargs}]")

    # --- set configuration
    filters = config.get("filters", 32)
    use_bias = config.get("use_bias", False)
    batchnorm = config.get("batchnorm", True)
    kernel_size = config.get("kernel_size", 5)
    activation = config.get("activation", "relu")
    output_channels = config.get("output_channels", 3)
    input_shape = input_shape_fixer(config.get("input_shape"))
    final_activation = config.get("final_activation", "tanh")
    kernel_initializer = config.get("kernel_initializer", "glorot_normal")
    kernel_regularizer = regularizer_builder(config.get("kernel_regularizer", "l2"))

    # --- set network parameters
    bn_params = dict(
        scale=True,
        center=use_bias,
        momentum=DEFAULT_BN_MOMENTUM,
        epsilon=DEFAULT_BN_EPSILON)

    upsampling_params = dict(
        size=(2, 2),
        interpolation="nearest")

    start_conv_params = dict(
        kernel_size=1,
        strides=(1, 1),
        padding="same",
        filters=filters,
        use_bias=use_bias,
        activation=activation,
        kernel_regularizer=kernel_regularizer,
        kernel_initializer=kernel_initializer
    )

    middle_conv_params = dict(
        kernel_size=kernel_size,
        strides=(1, 1),
        padding="same",
        filters=filters,
        use_bias=use_bias,
        activation=activation,
        kernel_regularizer=kernel_regularizer,
        kernel_initializer=kernel_initializer
    )

    final_conv_params = dict(
        kernel_size=1,
        strides=(1, 1),
        padding="same",
        use_bias=use_bias,
        filters=output_channels,
        activation=final_activation,
        kernel_regularizer=kernel_regularizer,
        kernel_initializer=kernel_initializer
    )

    # --- define superres head here
    model_input_layer = \
        tf.keras.Input(
            shape=input_shape,
            name="input_tensor")

    x = model_input_layer

    x = \
        conv2d_wrapper(
            input_layer=x,
            bn_params=None,
            conv_params=start_conv_params,
            channelwise_scaling=False,
            multiplier_scaling=False)

    if batchnorm:
        x = \
            tf.keras.layers.BatchNormalization(
                **bn_params)(x)

    x = \
        tf.keras.layers.UpSampling2D(
            **upsampling_params)(x)

    x = \
        conv2d_wrapper(
            input_layer=x,
            bn_params=None,
            conv_params=middle_conv_params,
            channelwise_scaling=False,
            multiplier_scaling=False)

    x = \
        conv2d_wrapper(
            input_layer=x,
            bn_params=None,
            conv_params=final_conv_params,
            channelwise_scaling=False,
            multiplier_scaling=False)

    x_result = \
        tf.keras.layers.Layer(
            name="output_tensor")(x)

    model_head = \
        tf.keras.Model(
            inputs=model_input_layer,
            outputs=x_result,
            name=f"superres_head")

    return model_head

# ---------------------------------------------------------------------


class DenoiserModule(tf.Module):
    """denoising inference module."""

    def __init__(
            self,
            model_backbone: keras.Model,
            model_denoise: keras.Model,
            model_normalize: keras.Model,
            model_denormalize: keras.Model,
            training_channels: int = 1,
            cast_to_uint8: bool = True):
        """
        Initializes a module for denoising.

        :param model_backbone: backbone model to use for inference
        :param model_denoise: denoising model to use for inference.
        :param model_normalize: model that normalizes the input
        :param model_denormalize: model that denormalizes the output
        :param training_channels: how many color channels were used in training
        :param cast_to_uint8: cast output to uint8

        """
        # --- argument checking
        if model_backbone is None:
            raise ValueError("model_denoise should not be None")
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
        self._model_backbone = model_backbone
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

        # --- run backbone
        x = self._model_backbone(x)

        # --- run denoise model
        x = self._model_denoise(x)

        # --- denormalize
        x = self._model_denormalize(x)

        # --- cast to uint8
        if self._cast_to_uint8:
            x = tf.round(x)
            x = tf.cast(x, dtype=tf.uint8)

        return x

    def __call__(self, input_tensor):
        return tf.function(
            func=self._run_inference_on_images,
            input_signature=[
                tf.TensorSpec(shape=[None,
                                     None,
                                     None,
                                     self._training_channels],
                              dtype=tf.uint8)])(input_tensor)

# ---------------------------------------------------------------------


def module_builder_denoise(
        model_backbone: keras.Model = None,
        model_denoise: keras.Model = None,
        model_normalize: keras.Model = None,
        model_denormalize: keras.Model = None,
        cast_to_uint8: bool = True) -> DenoiserModule:
    """
    builds a module for denoising.

    :param model_backbone: backbone model
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
        model_backbone.input_shape[-1]

    return \
        DenoiserModule(
            model_backbone=model_backbone,
            model_denoise=model_denoise,
            model_normalize=model_normalize,
            model_denormalize=model_denormalize,
            cast_to_uint8=cast_to_uint8,
            training_channels=training_channels)

# ---------------------------------------------------------------------
