import copy
from enum import Enum

import numpy as np
import tensorflow as tf
from typing import Dict, Tuple, Union, List, Any

# ---------------------------------------------------------------------

from .constants import *
from .custom_logger import logger
from .regularizers import \
    SoftOrthogonalConstraintRegularizer, \
    SoftOrthonormalConstraintRegularizer


# ---------------------------------------------------------------------

@tf.keras.utils.register_keras_serializable()
class Mish(tf.keras.layers.Layer):
    """
    Mish: A Self Regularized Non-Monotonic Neural Activation Function
    https://arxiv.org/abs/1908.08681v1

    x = x * tanh(softplus(x))
    """

    def __init__(self,
                 **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        return inputs * tf.math.tanh(tf.math.softplus(inputs))

# ---------------------------------------------------------------------


@tf.keras.utils.register_keras_serializable()
class ScaledMish(tf.keras.layers.Layer):
    """
    Scaled Mish: A variant of the Mish activation function that smoothly saturates at a positive value alpha.
    """

    def __init__(self, alpha=1.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha

    def call(self, inputs):
        mish_value = inputs * tf.nn.tanh(tf.nn.softplus(inputs))
        scaled_mish = self.alpha * tf.nn.tanh(mish_value / self.alpha)
        return scaled_mish

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'alpha': self.alpha
        })
        return config

# ---------------------------------------------------------------------

@tf.keras.utils.register_keras_serializable()
class GatedMLP(tf.keras.layers.Layer):
    def __init__(self,
                 filters: int,
                 use_bias: bool = False,
                 use_orthogonal_regularization: bool = False,
                 use_soft_orthonormal_regularization: bool = False,
                 attention_activation: str = "relu",
                 output_activation: str = "linear",
                 **kwargs):
        super().__init__(**kwargs)

        if use_orthogonal_regularization and use_soft_orthonormal_regularization:
            raise ValueError("Cannot have both enabled")
        self.filters = filters
        self.conv_gate = None
        self.conv_up = None
        self.conv_down = None
        self.use_bias = use_bias
        self.attention_activation = attention_activation
        self.output_activation = output_activation
        self.attention_activation_fn = None
        self.output_activation_fn = None
        self.use_orthogonal_regularization = use_orthogonal_regularization
        self.use_soft_orthonormal_regularization = use_soft_orthonormal_regularization

    def build(self, input_shape):
        from .utilities import activation_wrapper

        conv_params = dict(
            filters=self._filters,
            kernel_size=(1, 1),
            strides=(1, 1),
            activation=None,
            use_bias=self._use_bias,
            kernel_initializer="glorot_normal",
            kernel_regularizer="l2",
        )
        if self.use_soft_orthogonal_regularization:
            conv_params["kernel_regularizer"] = \
                SoftOrthogonalConstraintRegularizer(
                    lambda_coefficient=0.01, l1_coefficient=1e-4, l2_coefficient=0.0)
        if self.use_soft_orthonormal_regularization:
            conv_params["kernel_regularizer"] = \
                SoftOrthonormalConstraintRegularizer(
                    lambda_coefficient=0.01, l1_coefficient=1e-4, l2_coefficient=0.0)

        conv_params_out = copy.deepcopy(conv_params)
        conv_params_out["filters"] = input_shape[-1]
        self.conv_gate = tf.keras.layers.Conv2d(**conv_params)
        self.conv_up = tf.keras.layers.Conv2d(**conv_params)
        self.conv_down = tf.keras.layers.Conv2d(**conv_params_out)
        self.attention_activation_fn = activation_wrapper(activation=self.attention_activation)
        self.output_activation_fn = activation_wrapper(activation=self.output_activation)

    def call(self, inputs, training=None):
        x = inputs
        x_gate = self.attention_activation_fn(self.conv_gate(x, training=training))
        x_up = self.attention_activation_fn(self.conv_up(x, training=training))
        x_combined = x_gate * x_up
        x_gated_mlp = self.conv_down(x_combined, training=training)
        return self.output_activation_fn(x_gated_mlp)

    def compute_output_shape(self, input_shape):
        shape = copy.deepcopy(input_shape)
        return shape

# ---------------------------------------------------------------------


@tf.keras.utils.register_keras_serializable()
class GaussianFilter(tf.keras.layers.Layer):
    def __init__(
            self,
            kernel_size: Tuple[int, int] = (5, 5),
            strides: Tuple[int, int] = (1, 1),
            **kwargs):
        super().__init__(**kwargs)
        if len(kernel_size) != 2:
            raise ValueError("kernel size must be length 2")
        if len(strides) == 2:
            strides = [1] + list(strides) + [1]
        self._kernel_size = kernel_size
        self._strides = strides
        self._sigma = ((kernel_size[0] - 1) / 2, (kernel_size[1] - 1) / 2)
        self._kernel = None

    def build(self, input_shape):
        from .utilities import depthwise_gaussian_kernel
        self._kernel = \
            depthwise_gaussian_kernel(
                channels=input_shape[-1],
                kernel_size=self._kernel_size,
                nsig=self._sigma,
                dtype=np.float32)
        self._kernel = \
            tf.constant(self._kernel, dtype=self.compute_dtype)

    def call(self, inputs, training=False):
        return \
            tf.nn.depthwise_conv2d(
                input=inputs,
                filter=self._kernel,
                strides=self._strides,
                data_format=None,
                dilations=None,
                padding="SAME")


# ---------------------------------------------------------------------

@tf.keras.utils.register_keras_serializable()
class StochasticDepth(tf.keras.layers.Layer):
    """Stochastic Depth module.

    It performs batch-wise dropping rather than sample-wise. In libraries like
    `timm`, it's similar to `DropPath` layers that drops residual paths
    sample-wise.

    References:
      - https://github.com/rwightman/pytorch-image-models

    Args:
      drop_path_rate (float): Probability of dropping paths. Should be within
        [0, 1].

    Returns:
      Tensor either with the residual path dropped or kept.
    """

    def __init__(self, drop_path_rate: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        if drop_path_rate < 0.0 or drop_path_rate > 1.0:
            raise ValueError("drop_path_rate must be between 0.0 and 1.0")
        self.drop_path_rate = drop_path_rate
        self.dropout = None

    def build(self, input_shape):
        dims = len(input_shape)
        noise_shape = (input_shape[0], ) + (dims-1) * (1,)
        self.dropout = (
            tf.keras.layers.Dropout(
                rate=self.drop_path_rate,
                noise_shape=noise_shape)
        )

    def call(self, x, training=None):
        return self.dropout(x, training=training)

    def compute_output_shape(self, input_shape):
        return input_shape

# ---------------------------------------------------------------------

class MultiplierType(Enum):
    Global = 0

    Channel = 1

    @staticmethod
    def from_string(type_str: Union[str, "MultiplierType"]) -> "MultiplierType":
        # --- argument checking
        if type_str is None:
            raise ValueError("type_str must not be null")
        if isinstance(type_str, MultiplierType):
            return type_str
        if not isinstance(type_str, str):
            raise ValueError("type_str must be string")
        # --- clean string and get
        type_str = type_str.strip().upper()
        if len(type_str) <= 0:
            raise ValueError("stripped type_str must not be empty")
        return MultiplierType[type_str]

    def to_string(self) -> str:
        return self.name

# ---------------------------------------------------------------------


@tf.keras.utils.register_keras_serializable()
class LearnableMultiplier(tf.keras.layers.Layer):
    def __init__(self,
                 multiplier_type: str,
                 capped:bool = True,
                 initializer=tf.keras.initializers.truncated_normal(mean=0.0, stddev=0.01),
                 regularizer=tf.keras.regularizers.l1(1e-6),
                 **kwargs):
        """
        Per channel or global learnable multiplier

        :param initializer: initializes the values, should keep them close to 0.0
        :param regularizer: keeps the values near 0.0, so it becomes 1
        :param kwargs:
        """
        super().__init__(**kwargs)
        self.capped = capped
        self.w_multiplier = None
        self.initializer = initializer
        self.regularizer = regularizer
        self.multiplier_type = MultiplierType.from_string(multiplier_type)

    def build(self, input_shape):
        """
        Initializes the LearnableMultiplier layer.

        Parameters:
        initializer (tf.keras.initializers.Initializer): Initializer for the multiplier weights.
        regularizer (tf.keras.regularizers.Regularizer): Regularizer for the multiplier weights.
        kwargs: Additional keyword arguments.
        """
        new_shape = [1, ] * len(input_shape)
        if self.multiplier_type == MultiplierType.Channel:
            new_shape[-1] = input_shape[-1]

        self.w_multiplier = self.add_weight(
            shape=new_shape,
            name="w_multiplier",
            initializer=self.initializer,
            regularizer=self.regularizer,
            trainable=True,
        )

    def call(self, inputs, training=None):
        """
        Applies the multipliers to the input tensor.
        relu makes sure we don't have sign reversal
        multiplier starts from 1 and moves away from there
        y=\frac{\left(\tanh\left(4\cdot x+1.5\right)+1\right)}{2}\cdot1

        Parameters:
        inputs (tf.Tensor): Input tensor.
        training (bool, optional): Whether the call is in training mode or not.

        Returns:
        tf.Tensor: Output tensor with the multipliers applied.
        """
        if self.capped:
            return tf.multiply(
                tf.nn.tanh((4 * self.w_multiplier + 1.5) + 1) * 0.5,
                inputs)
        return tf.multiply(
            tf.nn.tanh((4 * self.w_multiplier + 1.5) + 1) * 0.5 * (1.0 + self.w_multiplier),
            inputs)


    def get_config(self):
        config = super().get_config().copy()
        config.update({
            "regularizer": self.regularizer,
            "initializer": self.initializer,
            "w_multiplier": self.w_multiplier.numpy(),
            "multiplier_type": self.multiplier_type.to_string()
        })
        return config

    def compute_output_shape(self, input_shape):
        return input_shape

# ---------------------------------------------------------------------

@tf.keras.utils.register_keras_serializable()
class SqueezeExcitation(tf.keras.layers.Layer):
    """
    Squeeze-and-Excitation Networks (2019)
    https://arxiv.org/abs/1709.01507

    General squeeze and excite block,
    has some differences from keras build-in

    smaller regularization than default

    based on the:
        https://github.com/lvpeiqing/SAR-U-Net-liver-segmentation/blob/master/models/core/modules.py
    """

    def __init__(self,
                 r_ratio: float = 0.25,
                 use_bias: bool = False,
                 kernel_initializer: str = "glorot_normal",
                 **kwargs):
        super().__init__(**kwargs)
        if r_ratio <= 0.0 or r_ratio > 1.0:
            raise ValueError(f"reduction [{r_ratio}] must be > 0 and <= 1")
        self.r_ratio = r_ratio
        self.use_bias = use_bias
        self.channels = -1
        self.channels_squeeze = -1
        self.conv_0 = None
        self.conv_1 = None
        self.scale = None
        self.scale = None
        self.kernel_initializer = kernel_initializer.strip().lower()
        self.pool = tf.keras.layers.GlobalAvgPool2D(keepdims=True)

    def build(self, input_shape):
        self.channels = input_shape[-1]
        self.channels_squeeze = max(1, int(round(self.channels * self.r_ratio)))
        self.scale = LearnableMultiplier(capped=True, multiplier_type=MultiplierType.Channel)
        kernel_initializer = copy.deepcopy(self.kernel_initializer)

        if kernel_initializer in ["trunc_normal", "truncated_normal"]:
            # https://github.com/facebookresearch/ConvNeXt/blob/048efcea897d999aed302f2639b6270aedf8d4c8/models/convnext.py#L105
            kernel_initializer = tf.keras.initializers.truncated_normal(mean=0.0, stddev=0.02)

        self.conv_0 = \
            tf.keras.layers.Conv2D(
                kernel_size=(1, 1),
                filters=self.channels_squeeze,
                activation="linear",
                use_bias=self.use_bias,
                kernel_regularizer=SoftOrthonormalConstraintRegularizer(
                    lambda_coefficient=0.01, l1_coefficient=1e-4, l2_coefficient=0.0),
                kernel_initializer=kernel_initializer)
        self.conv_1 = \
            tf.keras.layers.Conv2D(
                kernel_size=(1, 1),
                filters=self.channels,
                activation="linear",
                use_bias=self.use_bias,
                kernel_regularizer=SoftOrthonormalConstraintRegularizer(
                    lambda_coefficient=0.01, l1_coefficient=1e-4, l2_coefficient=0.0),
                kernel_initializer=kernel_initializer)

    def call(self, x, training):
        y = x
        y = self.pool(y)
        y = self.conv_0(y)
        # --- replaced relu with leaky relu and low alpha to allow some gradient flow
        o = tf.nn.gelu(features=y, approximate=True)
        o = self.conv_1(o)
        o = self.scale(o)
        o = tf.nn.sigmoid(o)
        return tf.math.multiply(x, o)


# ---------------------------------------------------------------------

@tf.keras.utils.register_keras_serializable()
class AdditiveAttentionGate(tf.keras.layers.Layer):
    """
    Convolutional Additive Attention Gate

    This layer implements a convolutional additive attention mechanism as described in the paper:
    https://www.researchgate.net/figure/Schematic-of-the-additive-Attention-Gate_fig2_358867978

    Parameters
    ----------
    attention_channels : int
        The number of channels for the attention mechanism.
    use_bias : bool, optional
        Whether to use bias in the convolutional layers. Default is False.
    use_bn : bool, optional
        Whether to use Batch Normalization. Default is False.
    use_ln : bool, optional
        Whether to use Layer Normalization. Default is False.
    use_soft_orthonormal_regularization : bool, optional
        Whether to use soft orthonormal regularization. Default is False.
    use_soft_orthogonal_regularization : bool, optional
        Whether to use soft orthogonal regularization. Default is False.
    kernel_initializer : str, optional
        Initializer for the kernel weights matrix. Default is "glorot_normal".
    **kwargs :
        Additional keyword arguments for the layer, passed to the base `tf.keras.layers.Layer` class.

    Attributes
    ----------
    conv_x : tf.keras.layers.Conv2D
        Convolutional layer for upsampled signal.
    bn_x : tf.keras.layers.BatchNormalization or None
        Batch normalization layer for upsampled signal (if use_bn is True).
    ln_x : tf.keras.layers.LayerNormalization or None
        Layer normalization layer for upsampled signal (if use_ln is True).
    conv_y : tf.keras.layers.Conv2D
        Convolutional layer for encoder feature.
    bn_y : tf.keras.layers.BatchNormalization or None
        Batch normalization layer for encoder feature (if use_bn is True).
    ln_y : tf.keras.layers.LayerNormalization or None
        Layer normalization layer for encoder feature (if use_ln is True).
    conv_o : tf.keras.layers.Conv2D
        Convolutional layer for the output signal.
    bn_o : tf.keras.layers.BatchNormalization or None
        Batch normalization layer for the output signal (if use_bn is True).
    ln_o : tf.keras.layers.LayerNormalization or None
        Layer normalization layer for the output signal (if use_ln is True).
    scale_o : LearnableMultiplier
        Channel learnable multiplier layer for scaling the output.

    Methods
    -------
    build(input_shapes)
        Creates the weights of the layer based on the input shapes.
    call(inputs, training=None)
        Applies the attention mechanism to the input tensors.

    Examples
    --------
    ```python
    import tensorflow as tf
    from your_module import AdditiveAttentionLayer

    # Example usage in a model
    encoder_feature = tf.keras.Input(shape=(32, 32, 64))
    upsample_signal = tf.keras.Input(shape=(32, 32, 64))
    x = AdditiveAttentionLayer(attention_channels=32)([encoder_feature, upsample_signal])
    model = tf.keras.Model(inputs=[encoder_feature, upsample_signal], outputs=x)

    model.summary()
    ```

    Notes
    -----
    - This layer is particularly useful in image segmentation models, where attention mechanisms can improve performance by focusing on relevant regions.
    - Ensure that the attention_channels parameter is set appropriately to balance model complexity and performance.
    - This implementation uses a combination of convolutional layers, optional normalization layers, and a custom scaling layer to achieve the attention effect.
    """

    def __init__(self,
                 attention_channels: int,
                 use_bias: bool = False,
                 use_bn: bool = False,
                 use_ln: bool = False,
                 use_soft_orthonormal_regularization: bool = False,
                 use_soft_orthogonal_regularization: bool = False,
                 kernel_initializer: str = "glorot_normal",
                 **kwargs):
        """
        Initializes the AdditiveAttentionLayer.

        attention_channels (int): The number of channels for the attention mechanism.
        use_bias (bool, optional): Whether to use bias in the convolutional layers. Default is False.
        use_bn (bool, optional): Whether to use Batch Normalization. Default is False.
        use_ln (bool, optional): Whether to use Layer Normalization. Default is False.
        use_soft_orthonormal_regularization (bool, optional): Whether to use soft orthonormal regularization. Default is False.
        use_soft_orthogonal_regularization (bool, optional): Whether to use soft orthogonal regularization. Default is False.
        kernel_initializer (str, optional): Initializer for the kernel weights matrix. Default is "glorot_normal".
        kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)

        # --- argument parsing
        if attention_channels <= 0:
            raise ValueError("Attention channels must be > 0")
        if use_ln and use_bn:
            raise ValueError(
                "cannot have enabled use_ln and use_bn at the same time")
        if use_soft_orthonormal_regularization and use_soft_orthogonal_regularization:
            raise ValueError(
                "cannot have enabled use_soft_orthonormal_regularization and "
                "use_soft_orthogonal_regularization at the same time")

        # ---
        self.attention_channels = attention_channels
        self.use_soft_orthonormal_regularization = use_soft_orthonormal_regularization
        self.use_soft_orthogonal_regularization = use_soft_orthogonal_regularization
        self.kernel_initializer = kernel_initializer.strip().lower()
        self.use_bias = use_bias
        self.use_bn = use_bn
        self.use_ln = use_ln
        self.conv_x = None
        self.bn_x = None
        self.ln_x = None
        self.conv_y = None
        self.bn_y = None
        self.ln_y = None
        self.conv_o = None
        self.bn_o = None
        self.ln_o = None
        self.scale_o = None

    def build(self, input_shapes):
        """
        Creates the weights of the layer based on the input shapes.

        Parameters:
        input_shapes (tuple): Shape of the input tensors, expected to be a list or tuple of two shapes.
        """
        encoder_feature, upsample_signal = input_shapes
        output_channels = encoder_feature[-1]

        # ---
        kernel_regularizer = tf.keras.regularizers.l2(1e-4)
        if self.use_soft_orthogonal_regularization:
            kernel_regularizer = \
                SoftOrthogonalConstraintRegularizer(
                    lambda_coefficient=DEFAULT_SOFTORTHOGONAL_LAMBDA,
                    l1_coefficient=DEFAULT_SOFTORTHOGONAL_L1,
                    l2_coefficient=DEFAULT_SOFTORTHOGONAL_L2)
        if self.use_soft_orthonormal_regularization:
            kernel_regularizer = \
                SoftOrthonormalConstraintRegularizer(
                    lambda_coefficient=DEFAULT_SOFTORTHONORMAL_LAMBDA,
                    l1_coefficient=DEFAULT_SOFTORTHONORMAL_L1,
                    l2_coefficient=DEFAULT_SOFTORTHONORMAL_L2)
        # ---
        kernel_initializer = copy.deepcopy(self.kernel_initializer)
        if kernel_initializer in ["trunc_normal", "truncated_normal"]:
            # https://github.com/facebookresearch/ConvNeXt/blob/048efcea897d999aed302f2639b6270aedf8d4c8/models/convnext.py#L105
            kernel_initializer = tf.keras.initializers.truncated_normal(mean=0.0, stddev=0.02)

        # ---
        self.conv_x = (
            tf.keras.layers.Conv2D(
                padding="same",
                filters=self.attention_channels,
                kernel_size=(1, 1),
                activation="linear",
                use_bias=self.use_bias,
                kernel_regularizer=copy.deepcopy(kernel_regularizer),
                kernel_initializer=kernel_initializer))

        if self.use_bn:
            self.bn_x = tf.keras.layers.BatchNormalization(center=self.use_bias)
        if self.use_ln:
            self.ln_x = tf.keras.layers.LayerNormalization(center=self.use_bias)

        # ---
        self.conv_y = (
            tf.keras.layers.Conv2D(
                padding="same",
                filters=self.attention_channels,
                kernel_size=(1, 1),
                activation="linear",
                use_bias=self.use_bias,
                kernel_regularizer=copy.deepcopy(kernel_regularizer),
                kernel_initializer=kernel_initializer))

        if self.use_bn:
            self.bn_y = tf.keras.layers.BatchNormalization(center=self.use_bias)
        if self.use_ln:
            self.ln_y = tf.keras.layers.LayerNormalization(center=self.use_bias)

        # ---
        self.conv_o = (
            tf.keras.layers.Conv2D(
                padding="same",
                filters=output_channels,
                kernel_size=(1, 1),
                activation="linear",
                use_bias=self.use_bias,
                kernel_regularizer=copy.deepcopy(kernel_regularizer),
                kernel_initializer=kernel_initializer))
        # ---
        self.scale_o = LearnableMultiplier(capped=False)

    def call(self, inputs, training=None):
        """
        Applies the attention mechanism to the input tensors.

        Parameters:
        inputs (list of tf.Tensor): List containing two input tensors - encoder feature and upsample signal.
        training (bool, optional): Whether the call is in training mode or not.

        Returns:
        tf.Tensor: Output tensor after applying the attention mechanism.
        """
        encoder_feature, upsample_signal = \
            inputs

        # --- encoder signal
        y = encoder_feature
        if self.use_bn:
            y = self.bn_y(y, training=training)
        if self.use_ln:
            y = self.ln_y(y, training=training)
        y = self.conv_y(y, training=training)

        # --- upsample signal
        x = upsample_signal
        if self.use_bn:
            x = self.bn_x(x, training=training)
        if self.use_ln:
            x = self.ln_x(x, training=training)
        x = self.conv_x(x, training=training)

        # --- replaced relu with leaky_relu, so we have some gradient flow
        o = tf.nn.leaky_relu(x + y, alpha=0.1)
        o = self.conv_o(o, training=training)
        o = self.scale_o(o, training=training)
        o = tf.nn.sigmoid(4.0 * o)

        return \
            tf.math.multiply(
                x=encoder_feature,
                y=o)

# ---------------------------------------------------------------------

@tf.keras.utils.register_keras_serializable()
class ConvNextBlock(tf.keras.layers.Layer):
    def __init__(self,
                 conv_params_1: Dict,
                 conv_params_2: Dict,
                 conv_params_3: Dict,
                 ln_params: Dict = None,
                 bn_params: Dict = None,
                 dropout_params: Dict = None,
                 dropout_2d_params: Dict = None,
                 use_gamma: bool = True,
                 use_soft_orthogonal_regularization: bool = False,
                 use_soft_orthonormal_regularization: bool = False,
                 **kwargs):
        """

        :param kwargs:
        """

        super().__init__(**kwargs)

        # conv params
        self.conv_params_1 = conv_params_1
        self.conv_params_2 = conv_params_2
        self.conv_params_3 = conv_params_3

        self.conv_1 = None
        self.conv_2 = None
        self.conv_3 = None

        self.activation_1 = None
        self.activation_2 = None
        self.activation_3 = None

        # ln params
        self.ln = None
        self.use_ln = False
        self.ln_params = ln_params
        if ln_params is not None:
            self.use_ln = True
            self.ln = tf.keras.layers.LayerNormalization(**ln_params)

        # bn params
        self.bn = None
        self.use_bn = False
        self.bn_params = bn_params
        if bn_params is not None:
            self.use_bn = True
            self.bn = tf.keras.layers.BatchNormalization(**bn_params)

        # dropout params
        if dropout_params is not None:
            self.use_dropout = True
            self.dropout_params = dropout_params
            self.dropout = None
        else:
            self.use_dropout = False
            self.dropout_params = {}

        # dropout 2d params
        if dropout_2d_params is not None:
            self.use_dropout_2d = True
            self.dropout_2d_params = dropout_2d_params
            self.dropout_2d = None
        else:
            self.use_dropout_2d = False
            self.dropout_2d_params = {}

        # learnable multiplier
        # https://github.com/facebookresearch/ConvNeXt/blob/048efcea897d999aed302f2639b6270aedf8d4c8/models/convnext.py#L45
        self.gamma = None
        self.use_gamma = use_gamma

        self.soft_gamma = None

        # regularizer
        self.use_soft_orthogonal_regularization = use_soft_orthogonal_regularization
        self.use_soft_orthonormal_regularization = use_soft_orthonormal_regularization

    def build(self, input_shape):
        from .utilities import activation_wrapper

        if self.use_dropout:
            self.dropout = tf.keras.layers.Dropout(input_shape=input_shape, **self.dropout_params)
        if self.use_dropout_2d:
            self.dropout_2d = tf.keras.layers.SpatialDropout2D(**self.dropout_2d_params)

        # activations
        self.activation_1 = activation_wrapper(activation=self.conv_params_1["activation"])
        self.activation_2 = activation_wrapper(activation=self.conv_params_2["activation"])
        self.activation_3 = activation_wrapper(activation=self.conv_params_3["activation"])

        # conv 1
        params = copy.deepcopy(self.conv_params_1)
        params["activation"] = "linear"
        # https://github.com/facebookresearch/ConvNeXt/blob/048efcea897d999aed302f2639b6270aedf8d4c8/models/convnext.py#L105
        params["depthwise_initializer"] = \
            tf.keras.initializers.truncated_normal(
                    mean=0.0,
                    stddev=DEFAULT_SOFTORTHONORMAL_STDDEV)
        self.conv_1 = tf.keras.layers.DepthwiseConv2D(**params)

        # conv 2
        params = copy.deepcopy(self.conv_params_2)
        params["activation"] = "linear"
        # https://github.com/facebookresearch/ConvNeXt/blob/048efcea897d999aed302f2639b6270aedf8d4c8/models/convnext.py#L105
        params["kernel_initializer"] = \
            tf.keras.initializers.truncated_normal(
                    mean=0.0,
                    stddev=DEFAULT_SOFTORTHONORMAL_STDDEV)
        if self.use_soft_orthogonal_regularization:
            params["kernel_regularizer"] = \
                SoftOrthogonalConstraintRegularizer(
                    lambda_coefficient=DEFAULT_SOFTORTHOGONAL_LAMBDA,
                    l1_coefficient=DEFAULT_SOFTORTHOGONAL_L1,
                    l2_coefficient=DEFAULT_SOFTORTHOGONAL_L2)
        if self.use_soft_orthonormal_regularization:
            params["kernel_regularizer"] = \
                SoftOrthonormalConstraintRegularizer(
                    lambda_coefficient=DEFAULT_SOFTORTHONORMAL_LAMBDA,
                    l1_coefficient=DEFAULT_SOFTORTHONORMAL_L1,
                    l2_coefficient=DEFAULT_SOFTORTHONORMAL_L2)
        self.conv_2 = tf.keras.layers.Conv2D(**params)

        # conv 3
        params = copy.deepcopy(self.conv_params_3)
        params["activation"] = "linear"
        params["kernel_initializer"] = \
            tf.keras.initializers.truncated_normal(
                    mean=0.0,
                    stddev=DEFAULT_SOFTORTHONORMAL_STDDEV)
        if self.use_soft_orthogonal_regularization:
            params["kernel_regularizer"] = \
                SoftOrthogonalConstraintRegularizer(
                    lambda_coefficient=DEFAULT_SOFTORTHOGONAL_LAMBDA,
                    l1_coefficient=DEFAULT_SOFTORTHOGONAL_L1,
                    l2_coefficient=DEFAULT_SOFTORTHOGONAL_L2)
        if self.use_soft_orthonormal_regularization:
            params["kernel_regularizer"] = \
                SoftOrthonormalConstraintRegularizer(
                    lambda_coefficient=DEFAULT_SOFTORTHONORMAL_LAMBDA,
                    l1_coefficient=DEFAULT_SOFTORTHONORMAL_L1,
                    l2_coefficient=DEFAULT_SOFTORTHONORMAL_L2)
        self.conv_3 = tf.keras.layers.Conv2D(**params)

        # gamma
        if self.use_gamma:
            self.gamma = (
                LearnableMultiplier(
                    name="gamma",
                    capped=False,
                    regularizer=tf.keras.regularizers.l2(1e-2),
                    multiplier_type=MultiplierType.Channel)
            )

    def call(self, inputs, training=None):
        x = inputs

        # --- 1st part
        # normalize
        if self.use_bn:
            x = self.bn(x, training=training)
        if self.use_ln:
            x = self.ln(x, training=training)
        x = self.conv_1(x, training=training)
        if self.conv_params_1["activation"] != "linear":
            x = self.activation_1(x)

        # --- 2nd part
        x = self.conv_2(x, training=training)
        if self.conv_params_2["activation"] != "linear":
            x = self.activation_2(x)

        if self.use_dropout:
            x = self.dropout(x, training=training)
        if self.use_dropout_2d:
            x = self.dropout_2d(x, training=training)

        # --- 3rd part
        x = self.conv_3(x, training=training)
        if self.conv_params_3["activation"] != "linear":
            x = self.activation_3(x)

        # --- gamma
        if self.use_gamma:
            x = self.gamma(x, training=training)

        return x

    def compute_output_shape(self, input_shape):
        shape = copy.deepcopy(input_shape)
        shape[-1] = self.conv_params_3["filters"]
        return shape

# ---------------------------------------------------------------------

@tf.keras.utils.register_keras_serializable()
class LogitNorm(tf.keras.layers.Layer):
    """
    implementation of logit_norm based on

    Mitigating Neural Network Overconfidence with Logit Normalization

    https://proceedings.mlr.press/v162/wei22d.html
    https://arxiv.org/abs/2205.09310
    https://github.com/hongxin001/logitnorm_ood

    """

    def __init__(self,
                 constant: float = 1.0,
                 axis: Union[int, Tuple[int, int]] = -1,
                 **kwargs):
        super().__init__(**kwargs)
        self._axis = axis
        self._constant = \
            tf.constant(constant, dtype=self.compute_dtype)

    def call(self, inputs, training):
        x = inputs
        x_denominator = tf.square(x)
        x_denominator = tf.reduce_sum(x_denominator, axis=self._axis, keepdims=True)
        x_denominator = tf.sqrt(x_denominator + 1e-7)
        return (
            tf.divide(x, x_denominator) / self._constant,
            x_denominator)

    def compute_output_shape(self, input_shape):
        return input_shape


# ---------------------------------------------------------------------

@tf.keras.utils.register_keras_serializable()
class ConvolutionalSelfAttention(tf.keras.layers.Layer):
    """
    Self attention layer

    implements self-attention block as described in
    Non-local Neural Networks (2018) by Facebook AI research

    A spacetime non-local block. The feature maps are
    shown as the shape of their tensors, e.g., T ×H×W ×1024 for
    1024 channels (proper reshaping is performed when noted). “⊗”
    denotes matrix multiplication, and “⊕” denotes element-wise sum.
    The softmax operation is performed on each row.
    Here we show the embedded Gaussian
    version, with a bottleneck of 512 channels. The vanilla Gaussian
    version can be done by removing θ and φ, and the dot-product
    version can be done by replacing softmax with scaling by 1/N .

    """

    def __init__(self,
                 attention_channels: int,
                 use_bias: bool = False,
                 bn_params: Dict = None,
                 ln_params: Dict = None,
                 use_gamma: bool = True,
                 attention_activation: str = "linear",
                 output_activation: str = "linear",
                 use_soft_orthonormal_regularization: bool = False,
                 use_soft_orthogonal_regularization: bool = False,
                 dropout: float = 0.0,
                 attention_resolution: Tuple[int, int] = (16, 16),
                 **kwargs):
        super().__init__(**kwargs)
        if attention_channels is None or attention_channels <= 0:
            raise ValueError("attention_channels should be > 0")
        self.attention_channels = attention_channels
        self.use_bias = use_bias
        self.output_activation = output_activation
        self.attention_activation = attention_activation
        self.attention_resolution = attention_resolution
        self.dropout = dropout
        self.query_conv = None
        self.key_conv = None
        self.value_conv = None
        self.attention = None
        self.output_fn = None

        self.use_bn = False
        if bn_params is not None:
            self.use_bn = True
            self.bn_0 = tf.keras.layers.BatchNormalization(**bn_params)

        self.use_ln = False
        if ln_params is not None:
            self.use_ln = True
            self.ln_0 = tf.keras.layers.LayerNormalization(**ln_params)

        # learnable multiplier
        self.gamma = None
        self.use_gamma = use_gamma

        # regularizer
        self.use_soft_orthogonal_regularization = use_soft_orthogonal_regularization
        self.use_soft_orthonormal_regularization = use_soft_orthonormal_regularization

        self.reshape_attention = None
        self.reshape_output = None
    def build(self, input_shape):
        qvk_conv_params = dict(
            filters=self.attention_channels,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="same",
            use_bias=self.use_bias,
            activation=self.attention_activation,
            kernel_regularizer=tf.keras.regularizers.L2(l2=1e-4),
            kernel_initializer="glorot_normal"
        )
        # query key value convolution
        params = copy.deepcopy(qvk_conv_params)
        if self.use_soft_orthogonal_regularization:
            params["kernel_regularizer"] = \
                SoftOrthogonalConstraintRegularizer(
                    lambda_coefficient=DEFAULT_SOFTORTHOGONAL_LAMBDA,
                    l1_coefficient=DEFAULT_SOFTORTHOGONAL_L1,
                    l2_coefficient=DEFAULT_SOFTORTHOGONAL_L2)
        if self.use_soft_orthonormal_regularization:
            params["kernel_regularizer"] = \
                SoftOrthonormalConstraintRegularizer(
                    lambda_coefficient=DEFAULT_SOFTORTHONORMAL_LAMBDA,
                    l1_coefficient=DEFAULT_SOFTORTHONORMAL_L1,
                    l2_coefficient=DEFAULT_SOFTORTHONORMAL_L2)
        self.key_conv = tf.keras.layers.Conv2D(**copy.deepcopy(params))
        self.query_conv = tf.keras.layers.Conv2D(**copy.deepcopy(params))
        self.value_conv = tf.keras.layers.Conv2D(**copy.deepcopy(params))

        # output convolution
        params = copy.deepcopy(qvk_conv_params)
        params["filters"] = input_shape[-1]
        params["activation"] = self.output_activation
        if self.use_soft_orthogonal_regularization:
            params["kernel_regularizer"] = \
                SoftOrthogonalConstraintRegularizer(
                    lambda_coefficient=DEFAULT_SOFTORTHOGONAL_LAMBDA,
                    l1_coefficient=DEFAULT_SOFTORTHOGONAL_L1,
                    l2_coefficient=DEFAULT_SOFTORTHOGONAL_L2)
        if self.use_soft_orthonormal_regularization:
            params["kernel_regularizer"] = \
                SoftOrthonormalConstraintRegularizer(
                    lambda_coefficient=DEFAULT_SOFTORTHONORMAL_LAMBDA,
                    l1_coefficient=DEFAULT_SOFTORTHONORMAL_L1,
                    l2_coefficient=DEFAULT_SOFTORTHONORMAL_L2)
        self.output_fn = tf.keras.layers.Conv2D(**copy.deepcopy(params))
        self.reshape_attention = (
            tf.keras.layers.Reshape(target_shape=(-1, self.attention_channels)))
        self.reshape_output = (
            tf.keras.layers.Reshape(
                target_shape=(self.attention_resolution[0], self.attention_resolution[1], -1)))

        # attention
        self.attention = (
            tf.keras.layers.Attention(
                use_scale=False,
                score_mode="dot",
                dropout=self.dropout))

        # gamma
        if self.use_gamma:
            self.gamma = (
                LearnableMultiplier(
                    capped=True,
                    multiplier_type=MultiplierType.Channel)
            )

    def call(self, inputs, training=None):
        shape_x = tf.shape(inputs)

        x = tf.image.resize(
            inputs,
            size=self.attention_resolution,
            method=tf.image.ResizeMethod.BILINEAR,
            preserve_aspect_ratio=False,
            antialias=False
        )

        # --- normalize
        if self.use_bn:
            x = self.bn_0(x, training=training)
        if self.use_ln:
            x = self.ln_0(x, training=training)

        # --- compute query, key, value
        q_x = self.reshape_attention(self.query_conv(x, training=training))
        v_x = self.reshape_attention(self.value_conv(x, training=training))
        k_x = self.reshape_attention(self.key_conv(x, training=training))

        # --- compute attention
        x = self.attention([q_x, v_x, k_x], training=training)
        x = self.reshape_output(x)

        # --- compute output conv
        x = tf.image.resize(
            x,
            size=shape_x[1:3],
            method=tf.image.ResizeMethod.BILINEAR,
            preserve_aspect_ratio=False,
            antialias=False
        )
        x = self.output_fn(x, training=training)

        # --- gamma
        if self.use_gamma:
            x = self.gamma(x, training=training)

        return x

    def compute_output_shape(self, input_shape):
        shape = copy.deepcopy(input_shape)
        return shape

# ---------------------------------------------------------------------
