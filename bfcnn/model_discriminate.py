# ---------------------------------------------------------------------

__author__ = "Nikolas Markou"
__version__ = "1.0.0"
__license__ = "MIT"

# ---------------------------------------------------------------------

import abc
from tensorflow import keras

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .utilities import *
from .custom_logger import logger
from .pyramid import build_pyramid_model, build_inverse_pyramid_model


# ---------------------------------------------------------------------


def model_builder(
        config: Dict) -> Tuple[keras.Model, keras.Model, keras.Model]:
    """
    Reads a configuration and returns 3 models,

    :param config: configuration dictionary
    :return: discriminator model, normalize model, denormalize model
    """
    logger.info("building model with config [{0}]".format(config))

    # --- argument parsing
    model_type = config["type"]
    levels = config.get("levels", 1)
    filters = config.get("filters", 32)
    no_layers = config.get("no_layers", 5)
    min_value = config.get("min_value", 0)
    max_value = config.get("max_value", 255)
    batchnorm = config.get("batchnorm", True)
    kernel_size = config.get("kernel_size", 3)
    pyramid_config = config.get("pyramid", None)
    clip_values = config.get("clip_values", False)
    shared_model = config.get("shared_model", False)
    input_shape = config.get("input_shape", (None, None, 3))
    output_multiplier = config.get("output_multiplier", 1.0)
    local_normalization = config.get("local_normalization", -1)
    final_activation = config.get("final_activation", "linear")
    kernel_regularizer = config.get("kernel_regularizer", "l1")
    inverse_pyramid_config = config.get("inverse_pyramid", None)
    add_intermediate_results = config.get("intermediate_results", False)
    kernel_initializer = config.get("kernel_initializer", "glorot_normal")
    use_local_normalization = local_normalization > 0
    use_global_normalization = local_normalization == 0
    use_normalization = use_local_normalization or use_global_normalization
    local_normalization_kernel = [local_normalization, local_normalization]
    for i in range(len(input_shape)):
        if input_shape[i] == "?" or \
                input_shape[i] == "" or \
                input_shape[i] == "-1":
            input_shape[i] = None

    # --- argument checking
    if levels <= 0:
        raise ValueError("levels must be > 0")
    if filters <= 0:
        raise ValueError("filters must be > 0")
    if kernel_size <= 0:
        raise ValueError("kernel_size must be > 0")

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

    # --- build denoise model
    model_params = dict(
        add_gates=False,
        filters=filters,
        use_bn=batchnorm,
        add_sparsity=False,
        no_layers=no_layers,
        input_dims=input_shape,
        kernel_size=kernel_size,
        final_activation=final_activation,
        kernel_regularizer=kernel_regularizer,
        kernel_initializer=kernel_initializer,
        add_intermediate_results=add_intermediate_results
    )

    if model_type == "resnet":
        pass
    elif model_type == "sparse_resnet":
        model_params["add_sparsity"] = True
    elif model_type == "gatenet":
        model_params["add_gates"] = True
    else:
        raise ValueError(
            "don't know how to build model [{0}]".format(model_type))

    def func_sigma_norm(args):
        y, mean_y, sigma_y = args
        return (y - mean_y) / sigma_y

    def func_sigma_denorm(args):
        y, mean_y, sigma_y = args
        return (y * sigma_y) + mean_y

    # --- connect the parts of the model
    # setup input
    input_layer = \
        keras.Input(
            shape=input_shape,
            name="input_tensor")
    x = input_layer

    logger.info("building model with multiscale pyramid")
    # build pyramid
    model_pyramid = \
        build_pyramid_model(
            input_dims=input_shape,
            config=pyramid_config)
    # build inverse pyramid
    model_inverse_pyramid = \
        build_inverse_pyramid_model(
            input_dims=input_shape[:-1] + [2],
            config=inverse_pyramid_config)
    # define normalization/denormalization layers
    local_normalization_layer = \
        keras.layers.Lambda(
            name="local_normalization",
            function=func_sigma_norm,
            trainable=False)
    local_denormalization_layer = \
        keras.layers.Lambda(
            name="local_denormalization",
            function=func_sigma_denorm,
            trainable=False)
    global_normalization_layer = \
        keras.layers.Lambda(
            name="global_normalization",
            function=func_sigma_norm,
            trainable=False)
    global_denormalization_layer = \
        keras.layers.Lambda(
            name="global_denormalization",
            function=func_sigma_denorm,
            trainable=False)

    # --- run inference
    x_levels = model_pyramid(x)

    means = []
    sigmas = []

    # local/global normalization cap
    if use_normalization:
        for i, x_level in enumerate(x_levels):
            mean, sigma = None, None
            if use_local_normalization:
                mean, sigma = \
                    mean_sigma_local(
                        input_layer=x_level,
                        kernel_size=local_normalization_kernel)
                x_level = \
                    local_normalization_layer(
                        [x_level, mean, sigma])
            if use_global_normalization:
                mean, sigma = \
                    mean_sigma_global(
                        input_layer=x_level,
                        axis=[1, 2])
                x_level = \
                    global_normalization_layer(
                        [x_level, mean, sigma])
            means.append(mean)
            sigmas.append(sigma)
            x_levels[i] = x_level

    # --- shared or separate models
    if shared_model:
        logger.info("building shared model")
        resnet_model = \
            build_model_discriminate_resnet(
                name="level_shared",
                **model_params)
        x_levels = [
            resnet_model(x_level)
            for i, x_level in enumerate(x_levels)
        ]
    else:
        logger.info("building per scale model")
        x_levels = [
            build_model_discriminate_resnet(
                name=f"level_{i}",
                **model_params)(x_level)
            for i, x_level in enumerate(x_levels)
        ]

    # --- split intermediate results and actual results
    x_levels_intermediate = []
    if add_intermediate_results:
        for i, x_level in enumerate(x_levels):
            x_levels_intermediate += x_level[1::]
        x_levels = [
            x_level[0]
            for i, x_level in enumerate(x_levels)
        ]

    # --- optional multiplier to help saturation
    if output_multiplier is not None and \
            output_multiplier != 1:
        x_levels = [
            x_level * output_multiplier
            for x_level in x_levels
        ]

    # --- local/global denormalization cap
    if use_normalization:
        for i, x_level in enumerate(x_levels):
            if use_local_normalization:
                x_level = \
                    local_denormalization_layer(
                        [x_level, means[i], sigmas[i]])
            if use_global_normalization:
                x_level = \
                    global_denormalization_layer(
                        [x_level, means[i], sigmas[i]])
            x_levels[i] = x_level

    # --- merge levels together
    x_result = \
        model_inverse_pyramid(x_levels)

    # clip to [-0.5, +0.5]
    if clip_values:
        x_result = \
            keras.backend.clip(
                x_result,
                min_value=-0.5,
                max_value=+0.5)

    # name output
    output_layer = \
        keras.layers.Layer(name="output_tensor")(
            x_result)

    # add intermediate results
    output_layers = [output_layer]
    if add_intermediate_results:
        x_levels_intermediate = [
            keras.layers.Layer(
                name=f"intermediate_tensor_{i}")(x_level_intermediate)
            for i, x_level_intermediate in enumerate(x_levels_intermediate)
        ]
        output_layers = output_layers + x_levels_intermediate

    # --- wrap and name model
    model_discriminate = \
        keras.Model(
            inputs=input_layer,
            outputs=output_layers,
            name=f"{model_type}_discriminate")

    return \
        model_discriminate, \
        model_normalize, \
        model_denormalize

# ---------------------------------------------------------------------


def build_model_discriminate_resnet(
        input_dims,
        no_layers: int,
        kernel_size: int,
        filters: int,
        activation: str = "relu",
        final_activation: str = "softmax",
        use_bn: bool = True,
        use_bias: bool = False,
        kernel_regularizer="l1",
        kernel_initializer="glorot_normal",
        add_sparsity: bool = False,
        add_gates: bool = False,
        add_intermediate_results: bool = False,
        add_learnable_multiplier: bool = True,
        name="resnet") -> keras.Model:
    """
    builds a resnet model

    :param input_dims: Models input dimensions
    :param no_layers: Number of resnet layers
    :param kernel_size: kernel size of the conv layers
    :param filters: number of filters per convolutional layer
    :param activation: intermediate activation
    :param final_activation: activation of the final layer
    :param use_bn: Use Batch Normalization
    :param use_bias: use bias
    :param kernel_regularizer: Kernel weight regularizer
    :param kernel_initializer: Kernel weight initializer
    :param add_sparsity: if true add sparsity layer
    :param add_gates: if true add gate layer
    :param add_intermediate_results: if true output results before projection
    :param add_learnable_multiplier:
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

    conv_params = dict(
        filters=filters,
        strides=(1, 1),
        padding="same",
        use_bias=use_bias,
        activation=activation,
        kernel_size=kernel_size,
        kernel_regularizer=kernel_regularizer,
        kernel_initializer=kernel_initializer
    )

    gate_params = dict(
        filters=filters,
        strides=(1, 1),
        padding="same",
        use_bias=use_bias,
        activation="linear",
        kernel_size=1,
        kernel_regularizer=kernel_regularizer,
        kernel_initializer=kernel_initializer
    )

    depth_conv_params = dict(
        kernel_size=3,
        filters=filters * 2,
        strides=(1, 1),
        padding="same",
        use_bias=use_bias,
        activation=activation,
        kernel_regularizer=kernel_regularizer,
        kernel_initializer=kernel_initializer
    )

    intermediate_conv_params = dict(
        kernel_size=1,
        filters=filters,
        strides=(1, 1),
        padding="same",
        use_bias=use_bias,
        activation=activation,
        kernel_regularizer=kernel_regularizer,
        kernel_initializer=kernel_initializer
    )

    final_conv_params = dict(
        filters=2,
        kernel_size=1,
        strides=(1, 1),
        padding="same",
        use_bias=use_bias,
        name="output_tensor",
        activation=final_activation,
        kernel_regularizer=kernel_regularizer,
        kernel_initializer=kernel_initializer
    )

    resnet_params = dict(
        no_layers=no_layers,
        bn_params=bn_params,
        depth_conv_params=depth_conv_params,
        intermediate_conv_params=intermediate_conv_params
    )

    # make it linear so it gets sparse afterwards
    if add_sparsity:
        conv_params["activation"] = "linear"

    if add_gates:
        resnet_params["gate_params"] = gate_params

    # --- build model
    # set input
    input_layer = \
        keras.Input(
            name="input_tensor",
            shape=input_dims)
    x = input_layer

    # optional batch norm
    if use_bn:
        x = keras.layers.BatchNormalization(**bn_params)(x)

    # add base layer
    x = keras.layers.Conv2D(**conv_params)(x)

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

    # learnable multiplier
    if add_learnable_multiplier:
        x = \
            learnable_multiplier_layer(
                input_layer=x,
                trainable=True,
                multiplier=1.0,
                activation="linear")

    # output to probability
    output_layer = \
        keras.layers.Conv2D(**final_conv_params)(x)

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

