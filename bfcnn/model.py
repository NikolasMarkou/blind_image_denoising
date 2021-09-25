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
        config: Dict) -> keras.Model:
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

    # --- connect the parts of the model
    # setup input
    model_input = keras.Input(shape=input_shape)
    x = model_input
    # add normalize cap
    if normalize_denormalize:
        x = model_normalize(x)

    # add model
    if model_type == "resnet":
        model = \
            build_resnet_model(
                use_bn=batchnorm,
                filters=filters,
                no_layers=no_layers,
                input_dims=input_shape,
                kernel_size=kernel_size,
                final_activation="tanh",
                kernel_regularizer=kernel_regularizer,
                kernel_initializer=kernel_initializer)
    elif model_type == "gatenet":
        model = \
            build_gatenet_model(
                use_bn=batchnorm,
                filters=filters,
                no_layers=no_layers,
                input_dims=input_shape,
                kernel_size=kernel_size,
                final_activation="tanh",
                kernel_regularizer=kernel_regularizer,
                kernel_initializer=kernel_initializer)
    else:
        raise ValueError(
            "don't know how to build model [{0}]".format(model_type))
    x = model(x)
    # uplift a bit because of tanh saturation
    x = x * 1.5

    # add denormalize cap
    if normalize_denormalize:
        x = model_denormalize(x)

    # --- wrap model
    return \
        keras.Model(
            inputs=model_input,
            outputs=x)

# ---------------------------------------------------------------------
