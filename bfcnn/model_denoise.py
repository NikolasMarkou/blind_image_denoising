# ---------------------------------------------------------------------

__author__ = "Nikolas Markou"
__version__ = "1.0.0"
__license__ = "MIT"

# ---------------------------------------------------------------------

import abc
from tensorflow import keras
from collections import namedtuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .utilities import *
from .custom_logger import logger
from .pyramid import \
    upscale_2x2_block, \
    build_pyramid_model, \
    build_inverse_pyramid_model
from .model_noise_estimation import \
    model_builder as model_noise_estimation_builder, \
    noise_estimation_mixer
from .regularizer import builder as regularizer_builder

# ---------------------------------------------------------------------

BuilderResults = namedtuple(
    "BuilderResults",
    {
        "denoiser",
        "denoiser_decomposition",
        "normalizer",
        "denormalizer",
        "pyramid",
        "inverse_pyramid"
     })

# ---------------------------------------------------------------------


def model_builder(
        config: Dict) -> BuilderResults:
    """
    Reads a configuration and returns 5 models,

    :param config: configuration dictionary
    :return:
        denoiser model,
        normalize model,
        denormalize model,
        pyramid model,
        inverse pyramid model
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
    dropout_rate = config.get("dropout_rate", -1)
    activation = config.get("activation", "relu")
    clip_values = config.get("clip_values", False)
    shared_model = config.get("shared_model", False)
    add_concat_input = config.get("add_concat_input", False)
    input_shape = config.get("input_shape", (None, None, 3))
    output_multiplier = config.get("output_multiplier", 1.0)
    local_normalization = config.get("local_normalization", -1)
    final_activation = config.get("final_activation", "linear")
    kernel_regularizer = config.get("kernel_regularizer", "l1")
    inverse_pyramid_config = config.get("inverse_pyramid", None)
    add_skip_with_input = config.get("add_skip_with_input", True)
    add_intermediate_results = config.get("intermediate_results", False)
    kernel_initializer = config.get("kernel_initializer", "glorot_normal")
    add_learnable_multiplier = config.get("add_learnable_multiplier", False)
    noise_estimation_mixer_config = config.get("noise_estimation_mixer", None)
    add_residual_between_models = config.get("add_residual_between_models", False)

    use_pyramid = pyramid_config is not None
    use_inverse_pyramid = inverse_pyramid_config is not None
    use_local_normalization = local_normalization > 0
    use_global_normalization = local_normalization == 0
    use_normalization = use_local_normalization or use_global_normalization
    use_noise_estimation_mixer = noise_estimation_mixer_config is not None
    local_normalization_kernel = [local_normalization, local_normalization]
    input_shape = input_shape_fixer(input_shape)

    # --- argument checking
    if levels <= 0:
        raise ValueError("levels must be > 0")
    if filters <= 0:
        raise ValueError("filters must be > 0")
    if kernel_size <= 0:
        raise ValueError("kernel_size must be > 0")

    kernel_regularizer = \
        regularizer_builder(kernel_regularizer)

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
        dropout_rate=dropout_rate,
        add_concat_input=add_concat_input,
        final_activation=final_activation,
        kernel_regularizer=kernel_regularizer,
        kernel_initializer=kernel_initializer,
        add_skip_with_input=add_skip_with_input,
        add_intermediate_results=add_intermediate_results,
        add_learnable_multiplier=add_learnable_multiplier,
    )

    if model_type == "resnet":
        model_params["add_gates"] = False
        model_params["add_sparsity"] = False
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

    # build pyramid / inverse pyramid
    if use_pyramid:
        logger.info(f"building pyramid: [{pyramid_config}]")
        model_pyramid = \
            build_pyramid_model(
                input_dims=input_shape,
                config=pyramid_config)

        logger.info(f"building inverse pyramid: [{inverse_pyramid_config}]")
        model_inverse_pyramid = \
            build_inverse_pyramid_model(
                input_dims=input_shape,
                config=inverse_pyramid_config)
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
    if use_pyramid:
        x_levels = model_pyramid(x, training=False)
    else:
        x_levels = [x]

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

    logger.info("pyramid produces [{0}] scales".format(len(x_levels)))

    # --- shared or separate models
    if shared_model:
        logger.info("building shared model")
        resnet_model = \
            build_model_resnet(
                name="level_shared",
                **model_params)
        resnet_models = [resnet_model] * len(x_levels)
    else:
        logger.info("building per scale model")
        resnet_models = [
            build_model_resnet(
                name=f"level_{i}",
                **model_params)
            for i in range(len(x_levels))
        ]

    # --- add residual between models
    # speeds up training a lot, and better results
    if add_residual_between_models:
        if use_noise_estimation_mixer:
            model_noise_estimation = \
                model_noise_estimation_builder(
                    noise_estimation_mixer_config)
        else:
            model_noise_estimation = None

        tmp_level = None
        for i, x_level in reversed(list(enumerate(x_levels))):
            if tmp_level is None:
                tmp_level = resnet_models[i](x_level)
            else:
                tmp_level = \
                    upscale_2x2_block(
                        input_layer=tmp_level)
                if use_noise_estimation_mixer:
                    # learnable mixer
                    tmp_level = \
                        noise_estimation_mixer(
                            model_noise_estimation=model_noise_estimation,
                            x0_input_layer=tmp_level,
                            x1_input_layer=x_level)
                    tmp_level = \
                        keras.layers.Layer(
                            name="level_{0}_to_{1}".format(i+1, i))(tmp_level)
                else:
                    # basic mixer
                    tmp_level = \
                        keras.layers.Add(
                            name="level_{0}_to_{1}".format(i+1, i))(
                            [tmp_level, x_level]) * 0.5
                tmp_level = resnet_models[i](tmp_level)
            x_levels[i] = tmp_level
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

    # --- keep model before merging (this is for better training)
    model_denoise_decomposition = \
        keras.Model(
            inputs=input_layer,
            outputs=x_levels,
            name=f"{model_type}_denoiser_decomposition")

    # --- merge levels together
    if use_pyramid:
        x_result = \
            model_inverse_pyramid(x_levels)
    else:
        x_result = x_levels[0]

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

    return BuilderResults(
        denoiser=model_denoise,
        denoiser_decomposition=model_denoise_decomposition,
        normalizer=model_normalize,
        denormalizer=model_denormalize,
        pyramid=model_pyramid,
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

        # --- run denoise model as many times as required
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
        training_channels: int = 1,
        cast_to_uint8: bool = True) -> DenoisingInferenceModule:
    """
    builds a module for denoising.

    :param model_denoise: denoising model to use for inference.
    :param model_normalize: model that normalizes the input
    :param model_denormalize: model that denormalizes the output
    :param training_channels: how many color channels were used in training
    :param cast_to_uint8: cast output to uint8

    :return: denoiser module
    """
    logger.info(
        f"building denoising module with "
        f"training_channels:{training_channels}, "
        f"cast_to_uint8:{cast_to_uint8}")

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
        raise ValueError("don't know how to handle training_channels:{0}".format(training_channels))

# ---------------------------------------------------------------------
