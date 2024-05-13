import copy
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
class RandomOnOff(tf.keras.layers.Layer):
    """randomly drops the whole connection"""

    def __init__(self,
                 rate: float = 0.5,
                 **kwargs):
        super().__init__(**kwargs)
        self._rate = rate
        self._dropout = None

    def build(self, input_shape):
        new_shape = [1, ] * len(input_shape)
        new_shape[0] = input_shape[0]
        self._dropout = (
            tf.keras.layers.Dropout(
                rate=self._rate,
                noise_shape=new_shape))

    def call(self, inputs, training):
        return self._dropout(inputs, training=training)


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
                nsig=self._sigma).astype("float32")

    def call(self, inputs, training):
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
        self.dropout = (
            tf.keras.layers.Dropout(
                rate=self.drop_path_rate,
                noise_shape=(input_shape[0], 1, 1, 1)))

    def call(self, x, training=None):
        if training:
            keep_prob = 1.0 - self.drop_path_rate
            return self.dropout(x / keep_prob, training=training)

        return x

    def compute_output_shape(self, input_shape):
        return input_shape

# ---------------------------------------------------------------------

@tf.keras.utils.register_keras_serializable()
class ChannelLearnableMultiplier(tf.keras.layers.Layer):
    def __init__(self,
                 initializer=tf.keras.initializers.truncated_normal(mean=0.0, stddev=0.1),
                 regularizer=tf.keras.regularizers.l1(1e-4),
                 **kwargs):
        """
        Per channel learnable multiplier

        :param initializer: initializes the values, should keep them close to 0.0
        :param regularizer: keeps the values near 0.0 so it becomes 1
        :param kwargs:
        """
        super().__init__(**kwargs)
        self.w_multiplier = None
        self.initializer = initializer
        self.regularizer = regularizer

    def build(self, input_shape):
        new_shape = [1, ] * len(input_shape)
        new_shape[-1] = input_shape[-1]
        self.w_multiplier = self.add_weight(
            shape=new_shape,
            name="w_multiplier",
            initializer=self.initializer,
            regularizer=self.regularizer,
            trainable=True,
        )

    def call(self, inputs, training=None):
        # relu makes sure we don't have sign reversal
        # multiplier starts from 1 and moves away from there
        return tf.keras.activations.relu(1.0 + self.w_multiplier) * inputs


# ---------------------------------------------------------------------


@tf.keras.utils.register_keras_serializable()
class SmoothChannelLearnableMultiplier(tf.keras.layers.Layer):
    def __init__(self,
                 initializer=tf.keras.initializers.truncated_normal(mean=0.0, stddev=0.01),
                 regularizer=tf.keras.regularizers.l1(1e-4),
                 **kwargs):
        """
        Per channel learnable multiplier

        :param initializer: initializes the values, should keep them close to 0.0
        :param regularizer: keeps the values near 0.0 so it becomes 1
        :param kwargs:
        """
        super().__init__(**kwargs)
        self.w_multiplier = None
        self.initializer = initializer
        self.regularizer = regularizer

    def build(self, input_shape):
        new_shape = [1, ] * len(input_shape)
        new_shape[-1] = input_shape[-1]
        self.w_multiplier = self.add_weight(
            shape=new_shape,
            name="w_multiplier",
            initializer=self.initializer,
            regularizer=self.regularizer,
            trainable=True,
        )

    def call(self, inputs, training=None):
        # y=\frac{\left(\tanh\left(2\cdot x\right)+1\right)}{2}
        # sigmoid is smooth on all values and keeps it between 0.0 and 1.0
        # multiplier starts from 1 and moves away from there,
        # so it is learning to turn off
        return tf.keras.activations.sigmoid(2.5 + self.w_multiplier) * inputs

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
        self.scale = ChannelLearnableMultiplier()
        self.kernel_initializer = kernel_initializer.strip().lower()
        self.pool = tf.keras.layers.GlobalAvgPool2D(keepdims=True)

    def build(self, input_shape):
        self.channels = input_shape[-1]
        self.channels_squeeze = max(1, int(round(self.channels * self.r_ratio)))
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
        o = tf.nn.leaky_relu(features=y, alpha=0.1)
        o = self.conv_1(o)
        o = self.scale(o)
        o = tf.nn.sigmoid(o)
        return tf.math.multiply(x, o)


# ---------------------------------------------------------------------

@tf.keras.utils.register_keras_serializable()
class AttentionGate(tf.keras.layers.Layer):
    """
    Convolutional Additive Attention gate

    https://www.researchgate.net/figure/Schematic-of-the-additive-Attention-Gate_fig2_358867978
    """

    def __init__(self,
                 attention_channels: int,
                 use_bias: bool = False,
                 use_bn: bool = True,
                 use_soft_orthonormal_regularization: bool = False,
                 use_soft_orthogonal_regularization: bool = False,
                 kernel_initializer: str = "glorot_normal",
                 **kwargs):
        super().__init__(**kwargs)
        self.attention_channels = attention_channels
        self.use_soft_orthonormal_regularization = use_soft_orthonormal_regularization
        self.use_soft_orthogonal_regularization = use_soft_orthogonal_regularization
        self.kernel_initializer = kernel_initializer.strip().lower()
        self.use_bias = use_bias
        self.use_bn = use_bn
        self.conv_x = None
        self.bn_x = None
        self.conv_y = None
        self.bn_y = None
        self.conv_o = None
        self.bn_o = None
        self.scale_o = None

    def build(self, input_shapes):
        encoder_feature, upsample_signal = input_shapes
        output_channels = encoder_feature[-1]

        # ---
        kernel_regularizer = tf.keras.regularizers.l2(1e-4)
        if self.use_soft_orthogonal_regularization:
            kernel_regularizer = \
                SoftOrthogonalConstraintRegularizer(
                    lambda_coefficient=0.01, l1_coefficient=1e-4, l2_coefficient=0.0)
        if self.use_soft_orthonormal_regularization:
            kernel_regularizer = \
                SoftOrthonormalConstraintRegularizer(
                    lambda_coefficient=0.01, l1_coefficient=1e-4, l2_coefficient=0.0)
        # ---
        kernel_initializer = copy.deepcopy(self.kernel_initializer)
        if kernel_initializer in ["trunc_normal", "truncated_normal"]:
            # https://github.com/facebookresearch/ConvNeXt/blob/048efcea897d999aed302f2639b6270aedf8d4c8/models/convnext.py#L105
            kernel_initializer = tf.keras.initializers.truncated_normal(mean=0.0, stddev=0.02)

        # ---
        self.conv_x = (
            tf.keras.layers.Conv2D(
                filters=self.attention_channels,
                kernel_size=(1, 1),
                activation="linear",
                use_bias=self.use_bias,
                kernel_regularizer=copy.deepcopy(kernel_regularizer),
                kernel_initializer=kernel_initializer))

        if self.use_bn:
            self.bn_x = tf.keras.layers.BatchNormalization(center=self.use_bias)

        # ---
        self.conv_y = (
            tf.keras.layers.Conv2D(
                filters=self.attention_channels,
                kernel_size=(1, 1),
                activation="linear",
                use_bias=self.use_bias,
                kernel_regularizer=copy.deepcopy(kernel_regularizer),
                kernel_initializer=kernel_initializer))

        if self.use_bn:
            self.bn_y = tf.keras.layers.BatchNormalization(center=self.use_bias)

        # ---
        self.conv_o = (
            tf.keras.layers.Conv2D(
                filters=output_channels,
                kernel_size=(1, 1),
                activation="linear",
                use_bias=self.use_bias,
                kernel_regularizer=copy.deepcopy(kernel_regularizer),
                kernel_initializer=kernel_initializer))
        if self.use_bn:
            self.bn_o = tf.keras.layers.BatchNormalization(center=self.use_bias)

        # ---
        self.scale_o = ChannelLearnableMultiplier()

    def call(self, inputs, training=None):
        encoder_feature, upsample_signal = inputs
        # --- upsample signal
        x = self.conv_x(upsample_signal)
        if self.use_bn:
            x = self.bn_x(x, training=training)
        # --- encoder signal
        y = self.conv_y(encoder_feature)
        if self.use_bn:
            y = self.bn_y(y, training=training)
        # --- replaced relu with leaky relu and low alpha to allow some gradient flow
        o = tf.nn.leaky_relu(x + y, alpha=0.1)
        # ---
        o = self.conv_o(o)
        if self.use_bn:
            o = self.bn_o(o, training=training)
        o = self.scale_o(o)
        o = tf.nn.sigmoid(o)
        return tf.math.multiply(encoder_feature, o)


# ---------------------------------------------------------------------

@tf.keras.utils.register_keras_serializable()
class GlobalLearnableMultiplier(tf.keras.layers.Layer):
    def __init__(self,
                 initializer=tf.keras.initializers.truncated_normal(mean=0.0, stddev=0.1, seed=0),
                 regularizer=tf.keras.regularizers.l1(1e-4),
                 **kwargs):
        """
        Global learnable multiplier

        :param kwargs:
        """

        super().__init__(**kwargs)
        self.w_multiplier = None
        self.initializer = initializer
        self.regularizer = regularizer

    def build(self, input_shape):
        new_shape = [1, ] * len(input_shape)
        self.w_multiplier = self.add_weight(
            shape=new_shape,
            initializer=self.initializer,
            regularizer=self.regularizer,
            trainable=True,
        )

    def call(self, inputs, training):
        return tf.keras.activations.relu(1.0 + self.w_multiplier) * inputs


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
                 drop_path_rate: float = 0.0,
                 use_gamma: bool = True,
                 use_soft_gamma: bool = False,
                 use_soft_orthogonal_regularization: bool = False,
                 use_soft_orthonormal_regularization: bool = False,
                 **kwargs):
        """

        :param kwargs:
        """

        super().__init__(**kwargs)
        if use_gamma and use_soft_gamma:
            raise ValueError("use_gamma and use_soft_gamma should not be both activated")

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

        # stochastic depth
        self.drop_path_rate = drop_path_rate
        self.use_stochastic_depth = drop_path_rate > 0.0
        self.stochastic_depth = StochasticDepth(drop_path_rate=self.drop_path_rate)

        # learnable multiplier
        # https://github.com/facebookresearch/ConvNeXt/blob/048efcea897d999aed302f2639b6270aedf8d4c8/models/convnext.py#L45
        self.gamma = None
        self.use_gamma = use_gamma

        self.soft_gamma = None
        self.use_soft_gamma = use_soft_gamma

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
        if params.get("depthwise_initializer", "glorot_normal") in ["trunc_normal", "truncated_normal"]:
            # https://github.com/facebookresearch/ConvNeXt/blob/048efcea897d999aed302f2639b6270aedf8d4c8/models/convnext.py#L105
            params["depthwise_initializer"] = (
                tf.keras.initializers.truncated_normal(mean=0.0, stddev=0.02))
        self.conv_1 = tf.keras.layers.DepthwiseConv2D(**params)

        # conv 2
        params = copy.deepcopy(self.conv_params_2)
        params["activation"] = "linear"
        if params.get("kernel_initializer", "glorot_normal") in ["trunc_normal", "truncated_normal"]:
            # https://github.com/facebookresearch/ConvNeXt/blob/048efcea897d999aed302f2639b6270aedf8d4c8/models/convnext.py#L105
            params["kernel_initializer"] = (
                tf.keras.initializers.truncated_normal(mean=0.0, stddev=0.02))
        if self.use_soft_orthogonal_regularization:
            params["kernel_regularizer"] = \
                SoftOrthogonalConstraintRegularizer(
                    lambda_coefficient=0.01, l1_coefficient=0.0, l2_coefficient=1e-4)
        if self.use_soft_orthonormal_regularization:
            params["kernel_regularizer"] = \
                SoftOrthonormalConstraintRegularizer(
                    lambda_coefficient=0.01, l1_coefficient=0.0, l2_coefficient=1e-4)
        self.conv_2 = tf.keras.layers.Conv2D(**params)

        # conv 3
        params = copy.deepcopy(self.conv_params_3)
        params["activation"] = "linear"
        if params.get("kernel_initializer", "glorot_normal") in ["trunc_normal", "truncated_normal"]:
            # https://github.com/facebookresearch/ConvNeXt/blob/048efcea897d999aed302f2639b6270aedf8d4c8/models/convnext.py#L105
            params["kernel_initializer"] = (
                tf.keras.initializers.truncated_normal(mean=0.0, stddev=0.02))
        if self.use_soft_orthogonal_regularization:
            params["kernel_regularizer"] = \
                SoftOrthogonalConstraintRegularizer(
                    lambda_coefficient=0.01, l1_coefficient=0.0, l2_coefficient=1e-4)
        if self.use_soft_orthonormal_regularization:
            params["kernel_regularizer"] = \
                SoftOrthonormalConstraintRegularizer(
                    lambda_coefficient=0.01, l1_coefficient=0.0, l2_coefficient=1e-4)
        self.conv_3 = tf.keras.layers.Conv2D(**params)

        # gamma
        if self.use_gamma:
            self.gamma = ChannelLearnableMultiplier()

        if self.use_soft_gamma:
            self.soft_gamma = SmoothChannelLearnableMultiplier()

    def call(self, inputs, training=None):
        x = inputs

        # --- 1st part
        x = self.conv_1(x)

        if self.use_bn:
            x = self.bn(x, training=training)
        if self.use_ln:
            x = self.ln(x, training=training)

        if self.conv_params_1["activation"] != "linear":
            x = self.activation_1(x)

        # --- 2nd part
        x = self.conv_2(x)
        if self.conv_params_2["activation"] != "linear":
            x = self.activation_2(x)
        if self.use_dropout:
            x = self.dropout(x, training=training)
        if self.use_dropout_2d:
            x = self.dropout_2d(x, training=training)

        # --- 3rd part
        x = self.conv_3(x)
        if self.conv_params_3["activation"] != "linear":
            x = self.activation_3(x)

        # --- gamma
        if self.use_gamma:
            x = self.gamma(x)
        elif self.use_soft_gamma:
            x = self.soft_gamma(x)

        # --- stochastic depth
        if self.use_stochastic_depth:
            x = self.stochastic_depth(x, training=training)

        return x

    def compute_output_shape(self, input_shape):
        shape = copy.deepcopy(input_shape)
        shape[-1] = self.conv_params_3["filters"]
        return shape

# ---------------------------------------------------------------------

# ---------------------------------------------------------------------


class Multiplier(tf.keras.layers.Layer):
    def __init__(self,
                 multiplier: float = 1.0,
                 regularizer: Any = None,
                 trainable: bool = True,
                 activation: Any = "linear",
                 name=None,
                 **kwargs):
        super(Multiplier, self).__init__(
            trainable=trainable,
            name=name,
            **kwargs)
        self._w0 = None
        self._w1 = None
        self._activation = None
        self._multiplier = multiplier
        self._activation_type = activation
        self._regularizer = keras.regularizers.get(regularizer)

    def build(self, input_shape):
        def init_w0_fn(shape, dtype):
            return np.zeros(shape, dtype=np.float32)

        self._w0 = \
            self.add_weight(
                shape=[1],
                trainable=True,
                name="w0",
                initializer=init_w0_fn,
                regularizer=self._regularizer)

        def init_w1_fn(shape, dtype):
            return np.ones(shape, dtype=np.float32) * self._multiplier

        self._w1 = \
            self.add_weight(
                shape=[1],
                trainable=False,
                name="w1",
                initializer=init_w1_fn,
                regularizer=self._regularizer)

        self._activation = keras.layers.Activation(self._activation_type)
        super(Multiplier, self).build(input_shape)

    def call(self, inputs):
        return self._activation(self._w0 + self._w1) * inputs

    def get_config(self):
        return {
            "w0": self._w0.numpy(),
            "w1": self._w1.numpy(),
            "regularizer": self._regularizer,
            "activation": self._activation_type
        }

    def compute_output_shape(self, input_shape):
        return input_shape


# ---------------------------------------------------------------------


class ChannelwiseMultiplier(tf.keras.layers.Layer):
    """
    learns a scaling multiplier for each channel (no bias) independently
    works always on the last tensor dim (channel)

    if (batch, filters) input then it learns to scale the filters
    if (batch, x, y, channels) input then it learns to scale the channels
    """

    def __init__(self,
                 multiplier: float = 1.0,
                 regularizer: Any = None,
                 trainable: bool = True,
                 activation: Any = "linear",
                 name=None,
                 **kwargs):
        super(ChannelwiseMultiplier, self).__init__(
            trainable=trainable,
            name=name,
            **kwargs)
        self._w0 = None
        self._w1 = None
        self._activation = None
        self._multiplier = multiplier
        self._activation_type = activation
        self._regularizer = keras.regularizers.get(regularizer)

    def build(self, input_shape):
        def init_w0_fn(shape, dtype):
            return np.zeros(shape, dtype=np.float32)

        self._w0 = \
            self.add_weight(
                shape=input_shape[-1],
                trainable=True,
                name="w0",
                initializer=init_w0_fn,
                regularizer=self._regularizer)

        def init_w1_fn(shape, dtype):
            return np.ones(shape, dtype=np.float32) * self._multiplier

        self._w1 = \
            self.add_weight(
                shape=[1],
                trainable=False,
                name="w1",
                initializer=init_w1_fn,
                regularizer=None)

        self._activation = keras.layers.Activation(self._activation_type)
        super(ChannelwiseMultiplier, self).build(input_shape)

    def call(self, inputs):
        return self._activation(self._w0 + self._w1) * inputs

    def get_config(self):
        return {
            "w0": self._w0.numpy(),
            "w1": self._w1.numpy(),
            "regularizer": self._regularizer,
            "activation": self._activation_type
        }

    def compute_output_shape(self, input_shape):
        return input_shape

# ---------------------------------------------------------------------