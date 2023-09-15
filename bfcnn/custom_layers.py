import numpy as np
import tensorflow as tf
from typing import Any, List
from tensorflow import keras


# ---------------------------------------------------------------------


class Mish(tf.keras.layers.Layer):
    """
    Mish: A Self Regularized Non-Monotonic Neural Activation Function
    https://arxiv.org/abs/1908.08681v1

    x = x * tanh(softplus(x))
    """
    def __init__(self,
                 name=None,
                 **kwargs):
        super(Mish, self).__init__(
            trainable=False,
            name=name,
            **kwargs)

    def build(self, input_shape):
        super(Mish, self).build(input_shape)

    def call(self, inputs):
        return inputs * tf.math.tanh(tf.math.softplus(inputs))

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        return {}

# ---------------------------------------------------------------------


class BinaryEntropyActivityRegularizer(tf.keras.layers.Layer):
    def __init__(self,
                 threshold: float = 0.5,
                 rate: float = 1e-2):
        super(BinaryEntropyActivityRegularizer, self).__init__()
        self._rate = rate
        self._threshold = threshold

    def call(self, inputs, training):
        # We use `add_loss` to create a regularization loss
        # that depends on the inputs.

        # --- threshold
        threshold_to_binary = \
            tf.round(tf.nn.sigmoid(inputs - self._threshold))

        # --- per variable calculation
        p_per_variable = \
            tf.reduce_mean(
                threshold_to_binary,
                axis=[0],
                keepdims=False)
        total_mean_entropy_per_variable = \
            tf.reduce_mean(
                -p_per_variable * tf.math.log(p_per_variable) - \
                (1.0 - p_per_variable) * tf.math.log(1.0 - p_per_variable),
                axis=None,
                keepdims=False
            )

        # --- add loss
        self.add_loss(
            self._rate *
            total_mean_entropy_per_variable
        )

        return inputs

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        return {
            "rate": self._rate,
            "threshold": self._threshold
        }


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


class RandomOnOff(tf.keras.layers.Layer):
    def __init__(self,
                 rate: float = 0.5,
                 trainable: bool = True,
                 name=None,
                 **kwargs):
        super(RandomOnOff, self).__init__(
            trainable=trainable,
            name=name,
            **kwargs)
        self._w0 = None
        self._dropout = None
        self._rate = rate

    def build(self, input_shape):
        def init_w0_fn(shape, dtype):
            return np.ones(shape, dtype=np.float32)

        self._w0 = \
            self.add_weight(
                shape=[1],
                trainable=False,
                regularizer=None,
                name="placeholder",
                initializer=init_w0_fn)
        self._dropout = keras.layers.Dropout(rate=self._rate)
        super(RandomOnOff, self).build(input_shape)

    def call(self, inputs, training):
        return self._dropout(
            self._w0, training=training) * inputs

    def get_config(self):
        return {
            "placeholder": self._w0.numpy(),
            "rate": self._rate
        }

    def compute_output_shape(self, input_shape):
        return input_shape


# ---------------------------------------------------------------------


class GeluLayer(tf.keras.layers.Layer):
    def __init__(self,
                 trainable: bool = True,
                 regularizer: Any = None,
                 name=None,
                 **kwargs):
        super(GeluLayer, self).__init__(
            trainable=trainable,
            name=name,
            **kwargs)
        self._offset = None
        self._multiplier = None
        self._regularizer = keras.regularizers.get(regularizer)

    def build(self, input_shape):
        def init_offset_fn(shape, dtype):
            return np.zeros(shape, dtype=np.float32)

        def init_multiplier_fn(shape, dtype):
            return np.ones(shape, dtype=np.float32)

        self._offset = \
            self.add_weight(
                shape=[1],
                trainable=True,
                regularizer=self._regularizer,
                name="offset",
                initializer=init_offset_fn)
        self._multiplier = \
            self.add_weight(
                shape=[1],
                trainable=True,
                regularizer=self._regularizer,
                name="multiplier",
                initializer=init_multiplier_fn)
        super(GeluLayer, self).build(input_shape)

    def call(self, inputs, training):
        x = inputs
        x = (x * self._multiplier) + self._offset
        x = tf.keras.activations.sigmoid(x * 1.702)
        return tf.keras.layers.Multiply()([x, inputs])

    def get_config(self):
        return {
            "offset": self._offset.numpy(),
            "multiplier": self._multiplier.numpy(),
            "regularizer": self._regularizer,
        }

    def compute_output_shape(self, input_shape):
        return input_shape


# ---------------------------------------------------------------------


class DifferentiableReluLayer(tf.keras.layers.Layer):
    def __init__(self,
                 trainable: bool = True,
                 threshold: float = 0.0,
                 max_value: float = 6.0,
                 multiplier: float = 10.0,
                 regularizer: Any = None,
                 name=None,
                 **kwargs):
        """
        Creates a differentiable relu layer

        :param threshold: lower bound value before zeroing
        :param max_value: max allowed value
        :param multiplier: controls steepness
        :result: activation layer
        """
        super(DifferentiableReluLayer, self).__init__(
            trainable=trainable,
            name=name,
            **kwargs)
        self._multiplier = multiplier
        self._threshold = threshold
        self._max_value = max_value
        self._regularizer = keras.regularizers.get(regularizer)

    def build(self, input_shape):
        def init_threshold_fn(shape, dtype):
            return np.zeros(shape, dtype=np.float32) + self._threshold

        def init_max_value_fn(shape, dtype):
            return np.zeros(shape, dtype=np.float32) + self._max_value

        def init_multiplier_fn(shape, dtype):
            return np.ones(shape, dtype=np.float32) * self._multiplier

        self._threshold = \
            self.add_weight(
                shape=[1],
                trainable=True,
                regularizer=self._regularizer,
                name="threshold",
                initializer=init_threshold_fn)
        self._max_value = \
            self.add_weight(
                shape=[1],
                trainable=False,
                regularizer=None,
                name="max_value",
                initializer=init_max_value_fn)
        self._multiplier = \
            self.add_weight(
                shape=[1],
                trainable=True,
                regularizer=self._regularizer,
                name="multiplier",
                initializer=init_multiplier_fn)
        super(DifferentiableReluLayer, self).build(input_shape)

    def call(self, inputs, training):
        x = inputs
        step_threshold = tf.math.sigmoid(self._multiplier * (x - self._threshold))
        step_max_value = tf.math.sigmoid(self._multiplier * (x - self._max_value))
        result = \
            (
                    (step_max_value * self._max_value) +
                    ((1.0 - step_max_value) * x)
            ) * step_threshold
        return result

    def get_config(self):
        return {
            "regularizer": self._regularizer,
            "threshold": self._threshold.numpy(),
            "max_value": self._max_value.numpy(),
            "multiplier": self._multiplier.numpy(),
        }

    def compute_output_shape(self, input_shape):
        return input_shape

# ---------------------------------------------------------------------


class DifferentiableGateLayer(tf.keras.layers.Layer):
    def __init__(self,
                 multiplier: float = 1.0,
                 min_value: float = 0.001,
                 regularizer: Any = "l1",
                 trainable: bool = True,
                 name=None,
                 **kwargs):
        """
        GELU with a twist to force it to become binary
        Creates a differentiable gate layer, it multiplies the input with a learnable sigmoid gate
        The gate starts as a normal sigmoid and it is being squeezed from the regularizer to become binary
        as the multiplier goes to zero

        :param multiplier: controls the initial steepness
        :param min_value: controls the maximum steepness
        :param regularizer: regularizer required to push multiplier down and increase the steepness

        :result: activation layer
        """
        super(DifferentiableGateLayer, self).__init__(
            trainable=trainable,
            name=name,
            **kwargs)
        self._multiplier = multiplier
        self._min_value = min_value
        self._regularizer = keras.regularizers.get(regularizer)

    def build(self, input_shape):

        def init_min_value(shape, dtype):
            return np.zeros(shape, dtype=np.float32) + self._min_value

        def init_multiplier_fn(shape, dtype):
            return np.ones(shape, dtype=np.float32) * self._multiplier

        self._min_value = \
            self.add_weight(
                shape=[1],
                trainable=False,
                regularizer=None,
                name="min_value",
                initializer=init_min_value)
        self._multiplier = \
            self.add_weight(
                shape=[1],
                trainable=True,
                regularizer=self._regularizer,
                name="multiplier",
                initializer=init_multiplier_fn)
        super(DifferentiableGateLayer, self).build(input_shape)

    def call(self, inputs, training):
        # minimum value of 1 gives a max multiplier of 1.0 / 1.0 = 1
        # minimum value of 0.1 gives a max multiplier of 1.0 / 0.1 = 10
        # minimum value of 0.01 gives a max multiplier of 1.0 / 0.01 = 100
        # minimum value of 0.001 gives a max multiplier of 1.0 / 0.001 = 1000
        x = tf.math.sigmoid(inputs * (1.0 / (tf.nn.relu(self._multiplier) + self._min_value)))
        return tf.keras.layers.Multiply()([x, inputs])

    def get_config(self):
        return {
            "regularizer": self._regularizer,
            "min_value": self._min_value.numpy(),
            "multiplier": self._multiplier.numpy(),
        }

    def compute_output_shape(self, input_shape):
        return input_shape

# ---------------------------------------------------------------------


class Patches(tf.keras.layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self._patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = \
            tf.image.extract_patches(
                images=images,
                sizes=[1, self._patch_size, self._patch_size, 1],
                strides=[1, self._patch_size, self._patch_size, 1],
                rates=[1, 1, 1, 1],
                padding="valid")
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

    def get_config(self):
        return {
            "patch_size": self._patch_size
        }

# ---------------------------------------------------------------------


class Mish(tf.keras.layers.Layer):
    """
    Mish: A Self Regularized Non-Monotonic Neural Activation Function
    https://arxiv.org/abs/1908.08681v1

    x = x * tanh(softplus(x))
    """
    def __init__(self,
                 name=None,
                 **kwargs):
        super(Mish, self).__init__(
            trainable=False,
            name=name,
            **kwargs)

    def build(self, input_shape):
        super(Mish, self).build(input_shape)

    def call(self, inputs):
        return inputs * tf.math.tanh(tf.math.softplus(inputs))

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        return {}

# ---------------------------------------------------------------------


class GradientBottleneck(tf.keras.layers.Layer):
    """
    limit gradient statically, percentage wise
    """
    def __init__(self,
                 name=None,
                 pass_through_percentage: float = 1.0,
                 **kwargs):
        super(GradientBottleneck, self).__init__(
            trainable=False,
            name=name,
            **kwargs)
        if pass_through_percentage > 1.0 or pass_through_percentage < 0.0:
            raise ValueError("multiplier must be in [0,1] range")
        self._pass_through_percentage = pass_through_percentage

    def build(self, input_shape):
        super(GradientBottleneck, self).build(input_shape)

    def call(self, inputs, training=None):
        if training is None:
            training = tf.keras.backend.learning_phase()
        if training:
            return \
                self._pass_through_percentage * inputs + \
                (1.0 - self._pass_through_percentage) * tf.stop_gradient(inputs)
        return inputs

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        return {
            "pass_through_percentage": self._pass_through_percentage
        }

# ---------------------------------------------------------------------


class GradientDropout(tf.keras.layers.Layer):
    """
    stop gradient randomly
    """
    def __init__(self,
                 name=None,
                 probability_off: float = 0.5,
                 **kwargs):
        super(GradientDropout, self).__init__(
            trainable=False,
            name=name,
            **kwargs)

        if probability_off > 1.0 or probability_off < 0.0:
            raise ValueError("probability_off must be in [0,1] range")
        self._probability_off = probability_off

    def build(self, input_shape):
        super(GradientDropout, self).build(input_shape)

    def call(self, inputs, training=None):
        if training is None:
            training = tf.keras.backend.learning_phase()
        if training:
            pass_through_percentage = \
                tf.random.uniform(
                    shape=(1,),
                    seed=0,
                    minval=0.0,
                    maxval=1.0,
                    dtype=tf.float32)
            return tf.cond(
                pred=pass_through_percentage < self._probability_off,
                true_fn=lambda: tf.stop_gradient(inputs),
                false_fn=lambda: inputs
            )
        return inputs

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        return {
            "probability_off": self._probability_off
        }

# ---------------------------------------------------------------------
