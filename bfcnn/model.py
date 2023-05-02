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
    layer_normalize, layer_denormalize
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
        "hydra",
        "options"
    })


def model_builder(
        config: Dict) -> BuilderResults:
    # --- argument checking
    # TODO

    # --- build backbone
    model_backbone, model_normalizer, model_denormalizer = \
        model_backbone_builder(config=config[BACKBONE_STR])

    input_shape = tf.keras.backend.int_shape(model_backbone.inputs[0])[1:]
    logger.info("input_shape: [{0}]".format(input_shape))

    # --- build hydra combined model
    input_layer = tf.keras.Input(shape=input_shape, name="input_tensor")
    input_normalized_layer = model_normalizer(input_layer, training=False)

    # common backbone low level
    backbone = model_backbone(input_normalized_layer)

    denoiser_input_channels = \
        tf.keras.backend.int_shape(
            model_backbone.outputs[0])[-1]
    denoiser_shape = copy.deepcopy(config[BACKBONE_STR][INPUT_SHAPE_STR])
    denoiser_shape[-1] = denoiser_input_channels
    config[DENOISER_STR][INPUT_SHAPE_STR] = denoiser_shape

    # --- build denoiser
    model_denoiser, options = \
        model_denoiser_builder(config=config[DENOISER_STR])

    options = dict(num_outputs=3, has_uncertainty=False)
    denoiser_output = model_denoiser(backbone)

    # denormalize
    denoiser_output = \
        model_denormalizer(denoiser_output, training=False)

    # wrap layers to set names
    denoiser_output = \
        tf.keras.layers.Layer(name=DENOISER_STR)(denoiser_output)

    # create model
    model_hydra = \
        tf.keras.Model(
            inputs=[
                input_layer
            ],
            outputs=[
                denoiser_output,
            ],
            name=f"hydra")

    # --- pack results
    return \
        BuilderResults(
            backbone=model_backbone,
            normalizer=model_normalizer,
            denormalizer=model_denormalizer,
            denoiser=model_denoiser,
            hydra=model_hydra,
            options=options
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
        **kwargs) -> Tuple[tf.keras.Model, Dict]:
    """
    builds the denoiser model on top of the backbone layer

    :param config: dictionary with the denoiser configuration

    :return: denoiser head model
    """
    """
    builds the denoiser model on top of the backbone layer

    :param config: dictionary with the denoiser configuration

    :return: denoiser head model, options
    """
    # --- argument checking
    logger.info(f"building denoiser model with [{config}]")
    if kwargs:
        logger.info(f"unused parameters [{kwargs}]")

    # --- set configuration
    use_bias = config.get("use_bias", False)
    output_channels = config.get("output_channels", 3)
    input_shape = input_shape_fixer(config.get("input_shape"))

    # --- config uncertainty or point estimation
    conv_params = \
        dict(
            kernel_size=1,
            strides=(1, 1),
            padding="same",
            use_bias=use_bias,
            filters=output_channels,
            activation="linear",
            kernel_regularizer="l2",
            kernel_initializer="glorot_normal")

    # --- define network here
    model_input_layer = \
        tf.keras.Input(
            shape=input_shape,
            name="input_tensor")

    x = model_input_layer

    # regression / point sample estimation
    options = \
        dict(num_outputs=1,
             has_uncertainty=False)
    x_expected = \
        conv2d_wrapper(x, conv_params=conv_params)
    # squash to [-1, +1] and then to [-0.51, +0.51]
    x_expected = tf.nn.tanh(x_expected) / 1.9
    x_expected = \
        tf.keras.layers.Layer(
            name="output_tensor")(x_expected)
    model_head = \
        tf.keras.Model(
            inputs=model_input_layer,
            outputs=[
                x_expected,
            ],
            name=f"denoiser_head")

    return \
        model_head, \
        options

# ---------------------------------------------------------------------


def build_normalize_model(
        input_dims,
        min_value: float = 0.0,
        max_value: float = 255.0,
        name: str = "normalize") -> tf.keras.Model:
    """
    Wrap a normalize layer in a model

    :param input_dims: Models input dimensions
    :param min_value: Minimum value
    :param max_value: Maximum value
    :param name: name of the model

    :return: normalization model
    """
    model_input = tf.keras.Input(shape=input_dims)

    # --- normalize input
    # from [min_value, max_value] to [-0.5, +0.5]
    model_output = \
        layer_normalize(
            input_layer=model_input,
            v_min=float(min_value),
            v_max=float(max_value))

    # --- wrap model
    return tf.keras.Model(
        name=name,
        trainable=False,
        inputs=model_input,
        outputs=model_output)

# ---------------------------------------------------------------------


def build_denormalize_model(
        input_dims,
        min_value: float = 0.0,
        max_value: float = 255.0,
        name: str = "denormalize") -> tf.keras.Model:
    """
    Wrap a denormalize layer in a model

    :param input_dims: Models input dimensions
    :param min_value: Minimum value
    :param max_value: Maximum value
    :param name: name of the model

    :return: denormalization model
    """
    model_input = tf.keras.Input(shape=input_dims)

    # --- denormalize input
    # from [-0.5, +0.5] to [v0, v1] range
    model_output = \
        layer_denormalize(
            input_layer=model_input,
            v_min=float(min_value),
            v_max=float(max_value))

    # --- wrap model
    return \
        tf.keras.Model(
            name=name,
            trainable=False,
            inputs=model_input,
            outputs=model_output)

# ---------------------------------------------------------------------

