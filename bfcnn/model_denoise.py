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
        config: Dict) -> Tuple[keras.Model, keras.Model,
                               keras.Model, keras.Model, keras.Model]:
    """
    Reads a configuration and returns 5 models,

    :param config: configuration dictionary
    :return: denoiser model, normalize model,
            denormalize model, pyramid model, inverse pyramid model
    """
    logger.info("building model with config [{0}]".format(config))

    # --- argument parsing
    model_type = config["type"]
    levels = config.get("levels", 1)
    filters = config.get("filters", 32)
    add_var = config.get("add_var", False)
    no_layers = config.get("no_layers", 5)
    min_value = config.get("min_value", 0)
    max_value = config.get("max_value", 255)
    batchnorm = config.get("batchnorm", True)
    kernel_size = config.get("kernel_size", 3)
    pyramid_config = config.get("pyramid", None)
    activation = config.get("activation", "relu")
    clip_values = config.get("clip_values", False)
    shared_model = config.get("shared_model", False)
    stop_gradient = config.get("stop_gradient", False)
    input_shape = config.get("input_shape", (None, None, 3))
    output_multiplier = config.get("output_multiplier", 1.0)
    local_normalization = config.get("local_normalization", -1)
    final_activation = config.get("final_activation", "linear")
    kernel_regularizer = config.get("kernel_regularizer", "l1")
    inverse_pyramid_config = config.get("inverse_pyramid", None)
    add_skip_with_input = config.get("add_skip_with_input", True)
    add_intermediate_results = config.get("intermediate_results", False)
    kernel_initializer = config.get("kernel_initializer", "glorot_normal")
    add_residual_between_models = config.get("add_residual_between_models", False)
    use_local_normalization = local_normalization > 0
    use_global_normalization = local_normalization == 0
    use_normalization = use_local_normalization or use_global_normalization
    local_normalization_kernel = [local_normalization, local_normalization]
    input_shape = input_shape_fixer(input_shape)

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
        add_var=add_var,
        filters=filters,
        use_bn=batchnorm,
        add_sparsity=False,
        no_layers=no_layers,
        activation=activation,
        input_dims=input_shape,
        kernel_size=kernel_size,
        stop_gradient=stop_gradient,
        final_activation=final_activation,
        kernel_regularizer=kernel_regularizer,
        kernel_initializer=kernel_initializer,
        add_skip_with_input=add_skip_with_input,
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
            input_dims=input_shape,
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

    logger.info("pyramid produces [{0}] levels".format(len(x_levels)))

    # --- shared or separate models
    if shared_model:
        logger.info("building shared model")
        resnet_model = \
            build_model_denoise_resnet(
                name="level_shared",
                **model_params)
        resnet_models = [resnet_model] * len(x_levels)
    else:
        logger.info("building per scale model")
        resnet_models = [
            build_model_denoise_resnet(
                name=f"level_{i}",
                **model_params)
            for i in range(len(x_levels))
        ]

    # --- add residual between models
    # speeds up training a lot, and better results
    if add_residual_between_models:
        # basic mixer
        # tmp_level = None
        # for i, x_level in reversed(list(enumerate(x_levels))):
        #     if tmp_level is None:
        #         tmp_level = resnet_models[i](x_level)
        #     else:
        #         tmp_level = \
        #             upscale_2x2_block(
        #                 input_layer=tmp_level)
        #         tmp_level = \
        #             keras.layers.Add(
        #                 name="level_{0}_to_{1}".format(i+1, i))(
        #                 [tmp_level, x_level]) * 0.5
        #         tmp_level = resnet_models[i](tmp_level)
        #     x_levels[i] = tmp_level
        # better mixer
        tmp_level = None
        for i, x_level in reversed(list(enumerate(x_levels))):
            if tmp_level is None:
                tmp_level = resnet_models[i](x_level)
                x_levels[i] = tmp_level
            else:
                tmp_level_2 = \
                    upscale_2x2_block(
                        input_layer=tmp_level)
                level_new_to_previous = \
                    keras.layers.Add(
                        name="level_{0}_to_{1}".format(i+1, i))(
                        [tmp_level_2, x_level]) * 0.5
                current_level = \
                    resnet_models[i](level_new_to_previous)
                tmp_level = (tmp_level_2 + current_level) * 0.5
                x_levels[i] = current_level
    else:
        for i, x_level in enumerate(x_levels):
            x_levels[i] = resnet_models[i](x_level)

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

    # --- clip levels to [-0.5, +0.5]
    if clip_values:
        for i, x_level in enumerate(x_levels):
            x_levels[i] = \
                tf.clip_by_value(
                    t=x_level,
                    clip_value_min=-0.5,
                    clip_value_max=+0.5)

    # --- merge levels together
    x_result = \
        model_inverse_pyramid(x_levels)

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
    model_denoise = \
        keras.Model(
            inputs=input_layer,
            outputs=output_layers,
            name=f"{model_type}_denoiser")

    return \
        model_denoise, \
        model_normalize, \
        model_denormalize, \
        model_pyramid, \
        model_inverse_pyramid

# ---------------------------------------------------------------------


def build_model_denoise_resnet(
        input_dims,
        no_layers: int,
        kernel_size: int,
        filters: int,
        activation: str = "relu",
        final_activation: str = "linear",
        use_bn: bool = True,
        use_bias: bool = False,
        kernel_regularizer="l1",
        kernel_initializer="glorot_normal",
        channel_index: int = 2,
        stop_gradient: bool = False,
        add_skip_with_input: bool = True,
        add_sparsity: bool = False,
        add_gates: bool = False,
        add_var: bool = False,
        add_intermediate_results: bool = False,
        add_learnable_multiplier: bool = True,
        add_projection_to_input: bool = True,
        name="resnet") -> keras.Model:
    """
    builds a resnet model

    :param input_dims: Models input dimensions
    :param no_layers: Number of resnet layers
    :param kernel_size: kernel size of the conv layers
    :param filters: number of filters per convolutional layer
    :param activation: intermediate activation
    :param final_activation: activation of the final layer
    :param channel_index: Index of the channel in dimensions
    :param use_bn: Use Batch Normalization
    :param use_bias: use bias
    :param kernel_regularizer: Kernel weight regularizer
    :param kernel_initializer: Kernel weight initializer
    :param stop_gradient: if true stop gradient at each resnet block
    :param add_skip_with_input: if true skip with input
    :param add_sparsity: if true add sparsity layer
    :param add_gates: if true add gate layer
    :param add_var: if true add variance for each block
    :param add_intermediate_results: if true output results before projection
    :param add_learnable_multiplier:
    :param add_projection_to_input: if true project to input tensor channel number
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
        kernel_size=1,
        filters=filters,
        strides=(1, 1),
        padding="same",
        use_bias=use_bias,
        activation="linear",
        kernel_regularizer=kernel_regularizer,
        kernel_initializer=kernel_initializer
    )

    first_conv_params = dict(
        kernel_size=1,
        filters=filters,
        strides=(1, 1),
        padding="same",
        use_bias=use_bias,
        activation=activation,
        kernel_regularizer=kernel_regularizer,
        kernel_initializer=kernel_initializer
    )

    second_conv_params = dict(
        kernel_size=3,
        filters=filters * 2,
        strides=(1, 1),
        padding="same",
        use_bias=use_bias,
        activation=activation,
        kernel_regularizer=kernel_regularizer,
        kernel_initializer=kernel_initializer
    )

    third_conv_params = dict(
        groups=2,
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
        kernel_size=1,
        strides=(1, 1),
        padding="same",
        use_bias=use_bias,
        activation=final_activation,
        filters=input_dims[channel_index],
        kernel_regularizer=kernel_regularizer,
        kernel_initializer=kernel_initializer
    )

    resnet_params = dict(
        no_layers=no_layers,
        bn_params=bn_params,
        gate_params=gate_params,
        stop_gradient=stop_gradient,
        first_conv_params=first_conv_params,
        second_conv_params=second_conv_params,
        third_conv_params=third_conv_params,
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
    y = input_layer

    # # optional batch norm
    # if use_bn:
    #     x = keras.layers.BatchNormalization(**bn_params)(x)

    if add_var:
        _, x_var = \
            mean_variance_local(
                input_layer=x,
                kernel_size=(5, 5))
        x = keras.layers.Concatenate()([x, x_var])

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

    # --- output layer branches here,
    # to allow space for intermediate results
    output_layer = x

    # learnable multiplier
    if add_learnable_multiplier:
        output_layer = \
            learnable_multiplier_layer(
                input_layer=output_layer,
                trainable=True,
                multiplier=1.0)

    # --- output to original channels / projection
    if add_projection_to_input:
        output_layer = \
            keras.layers.Conv2D(**final_conv_params)(output_layer)

    # --- skip with input layer
    if add_skip_with_input:
        output_layer = \
            keras.layers.Add()([output_layer, y])

    output_layer = \
        keras.layers.Layer(name="output_tensor")(output_layer)

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


class DenoisingInferenceModule(tf.Module, abc.ABC):
    """denoising inference module."""

    def __init__(
            self,
            model_denoise: keras.Model = None,
            model_normalize: keras.Model = None,
            model_denormalize: keras.Model = None,
            training_channels: int = 1,
            iterations: int = 1,
            cast_to_uint8: bool = True):
        """
        Initializes a module for denoising.

        :param model_denoise: denoising model to use for inference.
        :param model_normalize: model that normalizes the input
        :param model_denormalize: model that denormalizes the output
        :param training_channels: how many color channels were used in training
        :param iterations: how many times to run the model
        :param cast_to_uint8: cast output to uint8

        """
        # --- argument checking
        if model_denoise is None:
            raise ValueError("model_denoise should not be None")
        if iterations <= 0:
            raise ValueError("iterations should be > 0")
        if training_channels <= 0:
            raise ValueError("training channels should be > 0")

        # --- setup instance variables
        self._iterations = iterations
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
        # --- argument checking
        # --- argument checking
        if image is None:
            raise ValueError("input image cannot be empty")

        x = tf.cast(image, dtype=tf.float32)

        # --- normalize
        if self._model_normalize is not None:
            x = self._model_normalize(x)

        # --- run denoise model as many times as required
        for i in range(self._iterations):
            x = self._model_denoise(x)
            x = tf.clip_by_value(x, clip_value_min=-0.5, clip_value_max=+0.5)

        # --- denormalize
        if self._model_denormalize is not None:
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
            iterations: int = 1,
            cast_to_uint8: bool = True):
        super().__init__(
            model_denoise=model_denoise,
            model_normalize=model_normalize,
            model_denormalize=model_denormalize,
            training_channels=1,
            iterations=iterations,
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
            iterations: int = 1,
            cast_to_uint8: bool = True):
        super().__init__(
            model_denoise=model_denoise,
            model_normalize=model_normalize,
            model_denormalize=model_denormalize,
            training_channels=3,
            iterations=iterations,
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
        training_channels: int = 1,
        iterations: int = 1,
        cast_to_uint8: bool = True) -> DenoisingInferenceModule:
    """
    builds a module for denoising.

    :param model_denoise: denoising model to use for inference.
    :param model_normalize: model that normalizes the input
    :param model_denormalize: model that denormalizes the output
    :param training_channels: how many color channels were used in training
    :param iterations: how many times to run the model
    :param cast_to_uint8: cast output to uint8

    :return: denoiser module
    """
    logger.info(
        f"building denoising module with "
        f"iterations:{iterations}, "
        f"training_channels:{training_channels}, "
        f"cast_to_uint8:{cast_to_uint8}")

    if training_channels == 1:
        return \
            DenoisingInferenceModule1Channel(
                model_denoise=model_denoise,
                model_normalize=model_normalize,
                model_denormalize=model_denormalize,
                iterations=iterations,
                cast_to_uint8=cast_to_uint8)
    elif training_channels == 3:
        return \
            DenoisingInferenceModule3Channel(
                model_denoise=model_denoise,
                model_normalize=model_normalize,
                model_denormalize=model_denormalize,
                iterations=iterations,
                cast_to_uint8=cast_to_uint8)
    else:
        raise ValueError("don't know how to handle training_channels:{0}".format(training_channels))

# ---------------------------------------------------------------------
