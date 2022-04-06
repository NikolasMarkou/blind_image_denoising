# ---------------------------------------------------------------------

__author__ = "Nikolas Markou"
__version__ = "1.0.0"
__license__ = "MIT"

# ---------------------------------------------------------------------

import abc
import tensorflow as tf
from tensorflow import keras

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .utilities import *
from .custom_logger import logger

# ---------------------------------------------------------------------


def model_builder(
        config: Dict) -> keras.Model:
    """
    Reads a configuration and returns a model

    :param config: configuration dictionary
    :return: noise estimation model
    """
    logger.info("building model with config [{0}]".format(config))

    # --- argument parsing
    filters = config.get("filters", 32)
    no_layers = config.get("no_layers", 5)
    batchnorm = config.get("batchnorm", True)
    kernel_size = config.get("kernel_size", 3)
    activation = config.get("activation", "relu")
    final_activation = config.get("final_activation", "linear")
    kernel_regularizer = config.get("kernel_regularizer", "l1")
    kernel_initializer = config.get("kernel_initializer", "glorot_normal")
    input_shape = input_shape_fixer(config.get("input_shape", (None, None, 3)))

    # --- build denoise model
    model_params = dict(
        add_gates=False,
        add_var=False,
        filters=filters,
        use_bn=batchnorm,
        add_sparsity=False,
        no_layers=no_layers,
        activation=activation,
        input_dims=input_shape,
        kernel_size=kernel_size,
        final_activation=final_activation,
        kernel_regularizer=kernel_regularizer,
        kernel_initializer=kernel_initializer,
        add_skip_with_input=False,
        add_projection_to_input=False,
        add_intermediate_results=False,
        add_learnable_multiplier=False
    )

    final_conv_params = dict(
        kernel_size=1,
        strides=(1, 1),
        padding="same",
        use_bias=False,
        filters=filters,
        activation=activation,
        kernel_regularizer=kernel_regularizer,
        kernel_initializer=kernel_initializer
    )

    # --- connect the parts of the model
    # setup input
    input_layer = \
        keras.Input(
            shape=input_shape,
            name="input_tensor")
    x = input_layer
    x = \
        build_model_resnet(
            name="resnet_noise_estimation",
            **model_params)(x)
    x = keras.layers.Conv2D(**final_conv_params)(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Flatten()(x)

    output_layer = \
        keras.layers.Dense(
            units=1,
            use_bias=False,
            activation=final_activation,
            kernel_regularizer=kernel_regularizer,
            kernel_initializer=kernel_initializer)(x)

    # --- wrap and name model
    model_noise_estimation = \
        keras.Model(
            inputs=input_layer,
            outputs=output_layer,
            name=f"noise_estimation")

    return model_noise_estimation

# ---------------------------------------------------------------------


def noise_estimation_mixer(
        model_noise_estimation: keras.Model,
        x0_input_layer,
        x1_input_layer):
    x0_noise = model_noise_estimation(x0_input_layer)
    x1_noise = model_noise_estimation(x1_input_layer)
    x = keras.layers.Concatenate(axis=-1)([x0_noise, x1_noise])
    x = keras.layers.Softmax(axis=-1)(x)
    x_coeffs = tf.unstack(x, axis=1)
    x0_coeff = tf.reshape(x_coeffs[0], shape=(-1, 1, 1, 1))
    x1_coeff = tf.reshape(x_coeffs[1], shape=(-1, 1, 1, 1))
    x0_result_mix = x0_input_layer * x0_coeff
    x1_result_mix = x1_input_layer * x1_coeff
    return x0_result_mix + x1_result_mix

# ---------------------------------------------------------------------

