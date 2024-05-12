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
    conv2d_wrapper, \
    input_shape_fixer, \
    layer_normalize, layer_denormalize
from .backbone_unet import builder as builder_unet
from .backbone_resnet import builder as builder_resnet
from .backbone_convnext import builder as builder_convnext
from .backbone_unet_laplacian import builder as builder_unet_laplacian
from .regularizers import \
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

# ---------------------------------------------------------------------

BackboneBuilderResults = namedtuple(
    "BackboneBuilderResults",
    {
        "backbone",
        "normalizer",
        "denormalizer"
    })

# ---------------------------------------------------------------------

DenoiserBuilderResults = namedtuple(
    "DenoiserBuilderResults",
    {
        "denoiser",
        "options"
    })

# ---------------------------------------------------------------------


def model_builder(
        config: Dict) -> BuilderResults:
    # --- get configs
    config_denoiser = config[DENOISER_STR]
    config_backbone = config[BACKBONE_STR]
    batch_size = config.get(BATCH_SIZE_STR, None)

    # --- build backbone
    backbone_builder_results = model_backbone_builder(config=config_backbone)
    model_backbone = backbone_builder_results.backbone
    model_normalizer = backbone_builder_results.normalizer
    model_denormalizer = backbone_builder_results.denormalizer

    backbone_no_outputs = len(model_backbone.outputs)
    logger.warning(
        f"Backbone model has [{backbone_no_outputs}] outputs, "
        f"probably of different scale or depth")

    denoiser_input_channels = \
        tf.keras.backend.int_shape(
            model_backbone.outputs[0])[-1]
    denoiser_shape = copy.deepcopy(config_backbone[INPUT_SHAPE_STR])
    denoiser_shape[-1] = denoiser_input_channels
    config_denoiser[INPUT_SHAPE_STR] = denoiser_shape

    # --- build denoiser and segmentation networks
    denoiser_builder_results = model_denoiser_builder(config=config_denoiser)
    model_denoiser = denoiser_builder_results.denoiser

    input_image_shape = tf.keras.backend.int_shape(model_backbone.inputs[0])[1:]
    logger.info("input_shape: [{0}]".format(input_image_shape))

    # --- build hydra combined model
    input_layer = \
        tf.keras.Input(
            shape=input_image_shape,
            dtype="float32",
            sparse=False,
            ragged=False,
            batch_size=batch_size,
            name=INPUT_TENSOR_STR)

    input_normalized_layer = \
        model_normalizer(
            input_layer, training=False)

    # common backbone low level
    backbone = \
        model_backbone(input_normalized_layer)

    if len(model_backbone.outputs) == 1:
        denoiser_mid = \
            model_denoiser(backbone)

        output_layers = [
            denoiser_mid,
        ]
    else:
        config_denoisers = []

        for i in range(backbone_no_outputs):
            denoiser_input_channels = \
                tf.keras.backend.int_shape(model_backbone.outputs[i])[-1]
            denoiser_shape = copy.deepcopy(config_backbone[INPUT_SHAPE_STR])
            denoiser_shape[-1] = denoiser_input_channels
            tmp_config_denoiser = copy.deepcopy(config_denoiser)
            tmp_config_denoiser[INPUT_SHAPE_STR] = copy.deepcopy(denoiser_shape)
            config_denoisers.append(tmp_config_denoiser)

        # --- denoiser heads
        model_denoisers = [
            model_denoiser_builder(
                config=config_denoisers[i],
                name=f"denoiser_head_{i}")
            for i in range(backbone_no_outputs)
        ]
        denoisers_mid = [
            model_denormalizer(
                model_denoisers[i].denoiser(backbone[i]), training=False)
            for i in range(backbone_no_outputs)
        ]
        output_layers = \
            denoisers_mid

    # create model
    model_hydra = \
        tf.keras.Model(
            inputs=[
                input_layer
            ],
            outputs=output_layers,
            name=f"hydra")

    # --- pack results
    return \
        BuilderResults(
            backbone=model_backbone,
            normalizer=model_normalizer,
            denormalizer=model_denormalizer,
            denoiser=model_denoiser,
            hydra=model_hydra,
            options={}
        )


# ---------------------------------------------------------------------


def model_backbone_builder(
        config: Dict,
        name_str: str = None) -> BackboneBuilderResults:
    """
    reads a configuration a model backbone

    :param config: configuration dictionary
    :param name_str: name of the backbone

    :return: backbone, normalizer, denormalizer
    """
    logger.info("building backbone model with config [{0}]".format(config))

    # --- argument parsing
    model_type = config[TYPE_STR].strip().lower()
    value_range = config.get("value_range", (0, 255))
    input_shape = config.get(INPUT_SHAPE_STR, (None, None, 1))
    min_value = value_range[0]
    max_value = value_range[1]
    input_shape = input_shape_fixer(input_shape)
    if name_str is None or len(name_str) <= 0:
        name_str = f"{model_type}_backbone"

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

    if model_type == "resnet":
        backbone_builder = builder_resnet
    elif model_type == "unet":
        backbone_builder = builder_unet
    elif model_type in ["unet_laplacian"]:
        backbone_builder = builder_unet_laplacian
    elif model_type in ["convnext"]:
        backbone_builder = builder_convnext
    elif model_type == "efficientnet":
        raise NotImplementedError("efficientnet not implemented yet")
    else:
        raise ValueError(
            "don't know how to build model [{0}]".format(model_type))

    backbone_model = \
        backbone_builder(
            input_dims=input_shape, **config)

    # --- connect the parts of the model
    # setup input
    input_layer = \
        tf.keras.Input(
            shape=input_shape,
            name="input_tensor")
    x = input_layer

    x = backbone_model(x)

    model_backbone = \
        tf.keras.Model(
            inputs=input_layer,
            outputs=x,
            name=name_str)

    # ---
    return (
        BackboneBuilderResults(
            backbone=model_backbone,
            normalizer=model_normalize,
            denormalizer=model_denormalize))


# ---------------------------------------------------------------------


def model_denoiser_builder(
        config: Dict,
        **kwargs) -> DenoiserBuilderResults:
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
    use_bias = config.get(USE_BIAS, False)
    output_channels = config.get("output_channels", 3)
    input_shape = input_shape_fixer(config.get("input_shape"))
    kernel_regularizer = config.get(KERNEL_REGULARIZER, "l2")
    kernel_initializer = config.get(KERNEL_INITIALIZER, "glorot_normal")

    # --- config uncertainty or point estimation
    conv_params = \
        dict(
            kernel_size=1,
            strides=(1, 1),
            padding="same",
            use_bias=use_bias,
            activation="linear",
            filters=output_channels,
            kernel_regularizer=kernel_regularizer,
            kernel_initializer=kernel_initializer)

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
    x_expected = tf.nn.tanh(x_expected) * 0.51
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
        DenoiserBuilderResults(
            denoiser=model_head,
            options=options)

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

