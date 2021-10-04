import os
import math
import copy
import keras
import pathlib
import numpy as np
from typing import List, Tuple, Union, Dict

# ---------------------------------------------------------------------

from .utilities import *
from .custom_logger import logger

# ---------------------------------------------------------------------


def model_builder(
        config: Dict) -> Tuple[keras.Model, keras.Model, keras.Model]:
    """
    Reads a configuration and returns 3 models,

    :param config: configuration dictionary
    :return: denoiser model, normalize model, denormalize model
    """
    logger.info("building model with config [{0}]".format(config))

    # --- argument parsing
    model_type = config["type"]
    input_shape = config["input_shape"]
    filters = config.get("filters", 32)
    no_layers = config.get("no_layers", 5)
    min_value = config.get("min_value", 0)
    max_value = config.get("max_value", 255)
    batchnorm = config.get("batchnorm", True)
    kernel_size = config.get("kernel_size", 3)
    output_multiplier = config.get("output_multiplier", 1.0)
    final_activation = config.get("final_activation", "linear")
    kernel_regularizer = config.get("kernel_regularizer", "l1")
    normalize_denormalize = config.get("normalize_denormalize", False)
    kernel_initializer = config.get("kernel_initializer", "glorot_normal")

    # --- build normalize denormalize models
    model_normalize = \
        build_normalize_model(
            input_dims=input_shape,
            min_value=min_value,
            max_value=max_value)

    model_denormalize = \
        build_denormalize_model(
            input_dims=input_shape,
            min_value=min_value,
            max_value=max_value)

    if model_type == "resnet":
        model_denoise = \
            build_resnet_model(
                use_bn=batchnorm,
                filters=filters,
                no_layers=no_layers,
                input_dims=input_shape,
                kernel_size=kernel_size,
                final_activation=final_activation,
                kernel_regularizer=kernel_regularizer,
                kernel_initializer=kernel_initializer)
    elif model_type == "sparse_resnet":
        model_denoise = \
            build_sparse_resnet_model(
                filters=filters,
                no_layers=no_layers,
                input_dims=input_shape,
                kernel_size=kernel_size,
                final_activation=final_activation,
                kernel_regularizer=kernel_regularizer,
                kernel_initializer=kernel_initializer)
    elif model_type == "gatenet":
        model_denoise = \
            build_gatenet_model(
                use_bn=batchnorm,
                filters=filters,
                no_layers=no_layers,
                input_dims=input_shape,
                kernel_size=kernel_size,
                final_activation=final_activation,
                kernel_regularizer=kernel_regularizer,
                kernel_initializer=kernel_initializer)
    else:
        raise ValueError(
            "don't know how to build model [{0}]".format(model_type))

    # --- connect the parts of the model
    # setup input
    model_input = \
        keras.Input(
            shape=input_shape,
            name="input_tensor")
    x = model_input
    # add normalize cap
    if normalize_denormalize:
        x = model_normalize(x)

    # denoise image
    x = model_denoise(x)

    # uplift a bit because of tanh saturation
    if output_multiplier != 1.0:
        x = x * output_multiplier

    # add denormalize cap
    if normalize_denormalize:
        x = model_denormalize(x)

    # --- wrap model
    model_denoise = \
        keras.Model(
            inputs=model_input,
            outputs=x)

    return \
        model_denoise, \
        model_normalize, \
        model_denormalize

# ---------------------------------------------------------------------
