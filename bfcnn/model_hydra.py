import copy
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
    build_denormalize_model, \
    expected_uncertainty_head
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
        "superres",
        "hydra"
    })


def model_builder(
        config: Dict) -> BuilderResults:
    # --- argument checking
    # TODO

    # --- build backbone
    model_backbone_low, model_normalizer, model_denormalizer = \
        model_backbone_builder(config=config[BACKBONE_STR])

    # --- build denoiser, superres
    model_denoiser = model_denoiser_builder(config=config[DENOISER_STR])
    model_superres = model_superres_builder(config=config[SUPERRES_STR])

    input_shape = tf.keras.backend.int_shape(model_backbone_low.inputs[0])[1:]
    logger.info("input_shape: [{0}]".format(input_shape))

    # --- build hydra combined model
    input_layer = tf.keras.Input(shape=input_shape, name="input_tensor")
    input_normalized_layer = model_normalizer(input_layer, training=False)

    # common backbone low level
    backbone_low_level = model_backbone_low(input_normalized_layer)

    # low level heads
    denoiser_mid, denoiser_uq_mid = model_denoiser(backbone_low_level)
    superres_mid, superres_uq_mid = model_superres(backbone_low_level)

    # denormalize
    denoiser_output = model_denormalizer(denoiser_mid, training=False)
    superres_output = model_denormalizer(superres_mid, training=False)
    denoiser_uq_output = denoiser_uq_mid
    superres_uq_output = superres_uq_mid

    # wrap layers to set names
    denoiser_output = tf.keras.layers.Layer(name=DENOISER_STR)(denoiser_output)
    superres_output = tf.keras.layers.Layer(name=SUPERRES_STR)(superres_output)
    denoiser_uq_output = tf.keras.layers.Layer(name=DENOISER_UQ_STR)(denoiser_uq_output)
    superres_uq_output = tf.keras.layers.Layer(name=SUPERRES_UQ_STR)(superres_uq_output)

    # create model
    model_hydra = \
        tf.keras.Model(
            inputs=[
                input_layer
            ],
            outputs=[
                denoiser_output,
                denoiser_uq_output,
                superres_output,
                superres_uq_output
            ],
            name=f"hydra")

    # --- pack results
    return \
        BuilderResults(
            backbone=model_backbone_low,
            normalizer=model_normalizer,
            denormalizer=model_denormalizer,
            denoiser=model_denoiser,
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

    block_groups = config.get("block_groups", None)
    block_kernels = config.get("block_kernels", (3, 3))
    block_filters = config.get("block_filters", (32, 32))
    block_depthwise = config.get("block_depthwise", None)
    add_initial_bn = config.get("add_initial_bn", False)
    base_conv_params = config.get("base_conv_params", None)
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
        block_groups=block_groups,
        block_kernels=block_kernels,
        block_filters=block_filters,
        block_depthwise=block_depthwise,
        add_laplacian=add_laplacian,
        stop_gradient=stop_gradient,
        add_initial_bn=add_initial_bn,
        add_concat_input=add_concat_input,
        base_conv_params=base_conv_params,
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
        tf.keras.Input(
            shape=input_shape,
            name="input_tensor")
    x = input_layer

    # --- run inference
    if use_pyramid:
        x_levels = model_pyramid(x, training=False)
    else:
        x_levels = [x]

    if len(x_levels) > 1:
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
        if len(x_levels) > 1:
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
        residual_conv_params = dict(
            kernel_size=1,
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
                previous_level = \
                    conv2d_wrapper(
                        input_layer=previous_level,
                        conv_params=residual_conv_params,
                        channelwise_scaling=False,
                        multiplier_scaling=False)
                previous_level = \
                    tf.keras.layers.UpSampling2D(
                        **upsampling_params)(previous_level)
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
        tf.keras.Model(
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
    """
    builds the denoiser model on top of the backbone layer

    :param config: dictionary with the denoiser configuration

    :return: denoiser head model
    """
    # --- argument checking
    logger.info(f"building denoiser model with [{config}]")
    if kwargs:
        logger.info(f"unused parameters [{kwargs}]")

    # --- set configuration
    use_bias = config.get("use_bias", False)
    output_channels = config.get("output_channels", 3)
    input_shape = input_shape_fixer(config.get("input_shape"))
    lin_start = config.get("lin_start", -0.5)
    lin_stop = config.get("lin_stop", +0.5)

    uncertainty_kernel_regularizer = "l1"
    uncertainty_kernel_initializer = "glorot_normal"
    uncertainty_buckets = config.get("uncertainty_buckets", 16)
    uncertainty_threshold = config.get("uncertainty_threshold", None)
    uncertainty_activation = config.get("uncertainty_activation", "linear")

    # --- set network parameters
    uncertainty_conv_params = \
        dict(
            kernel_size=1,
            strides=(1, 1),
            padding="same",
            use_bias=use_bias,
            filters=uncertainty_buckets,
            activation=uncertainty_activation,
            kernel_regularizer=uncertainty_kernel_regularizer,
            kernel_initializer=uncertainty_kernel_initializer)

    # --- define superres network here
    model_input_layer = \
        tf.keras.Input(
            shape=input_shape,
            name="input_tensor")

    x = model_input_layer

    backbone, _, _ = model_backbone_builder(config)
    x = backbone(x)

    x_expected, x_uncertainty = \
        expected_uncertainty_head(
            input_layer=x,
            conv_parameters=uncertainty_conv_params,
            output_channels=output_channels,
            probability_threshold=uncertainty_threshold,
            linspace_start_stop=(lin_start, lin_stop))

    x_expected = \
        tf.keras.layers.Layer(
            name="output_tensor_expected")(x_expected)

    x_uncertainty = \
        tf.keras.layers.Layer(
            name="output_tensor_uncertainty")(x_uncertainty)

    model_head = \
        tf.keras.Model(
            inputs=model_input_layer,
            outputs=[x_expected, x_uncertainty],
            name=f"denoiser_head")

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
    if kwargs:
        logger.info(f"unused parameters [{kwargs}]")

    # --- set configuration
    use_bias = config.get("use_bias", False)
    lin_start = config.get("lin_start", -0.5)
    lin_stop = config.get("lin_stop", +0.5)
    output_channels = config.get("output_channels", 3)
    input_shape = input_shape_fixer(config.get("input_shape"))
    upscale_type = config.get("upscale_type", "nearest").strip().lower()
    kernel_initializer = config.get("kernel_initializer", "glorot_normal")
    kernel_regularizer = regularizer_builder(config.get("kernel_regularizer", "l2"))

    uncertainty_kernel_regularizer = "l1"
    uncertainty_kernel_initializer = "glorot_normal"
    uncertainty_buckets = config.get("uncertainty_buckets", 16)
    uncertainty_threshold = config.get("uncertainty_threshold", None)
    uncertainty_activation = config.get("uncertainty_activation", "linear")

    # --- set network parameters
    uncertainty_conv_params = \
        dict(
            kernel_size=1,
            strides=(1, 1),
            padding="same",
            use_bias=use_bias,
            filters=uncertainty_buckets,
            activation=uncertainty_activation,
            kernel_regularizer=uncertainty_kernel_regularizer,
            kernel_initializer=uncertainty_kernel_initializer)

    # --- define superres network here
    model_input_layer = \
        tf.keras.Input(
            shape=input_shape,
            name="input_tensor")
    x = model_input_layer

    # NOTE
    # nearest / bilinear -> conv2d (no artifacts)
    # conv2dTranspose (no artifacts)
    # conv2d -> conv2dTranspose (artifacts)
    if upscale_type == "nearest":
        x = \
            tf.keras.layers.UpSampling2D(
                size=(2, 2), interpolation="nearest")(x)
    elif upscale_type == "bilinear":
        x = \
            tf.keras.layers.UpSampling2D(
                size=(2, 2), interpolation="bilinear")(x)
    elif upscale_type == "dilate":
        config["base_conv_params"] = \
            dict(
                kernel_size=config["kernel_size"],
                filters=config["filters"],
                strides=(2, 2),
                padding="same",
                use_bias=use_bias,
                activation="linear",
                dilation_rate=(1, 1),
                kernel_regularizer=kernel_regularizer,
                kernel_initializer=kernel_initializer
            )
    else:
        raise ValueError(f"don't know how to handle upscale_type: [{upscale_type}]")

    backbone, _, _ = model_backbone_builder(config)
    x = backbone(x)

    x_expected, x_uncertainty = \
        expected_uncertainty_head(
            input_layer=x,
            conv_parameters=uncertainty_conv_params,
            output_channels=output_channels,
            probability_threshold=uncertainty_threshold,
            linspace_start_stop=(lin_start, lin_stop))

    x_expected = \
        tf.keras.layers.Layer(
            name="output_tensor_expected")(x_expected)

    x_uncertainty = \
        tf.keras.layers.Layer(
            name="output_tensor_uncertainty")(x_uncertainty)

    model_head = \
        tf.keras.Model(
            inputs=model_input_layer,
            outputs=[
                x_expected,
                x_uncertainty
            ],
            name=f"superres_head")

    return model_head

# ---------------------------------------------------------------------
