import numpy as np
import tensorflow as tf
from typing import Any, List
from tensorflow import keras

# ---------------------------------------------------------------------

from .regularizer import builder, REGULARIZER_ALLOWED_TYPES


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
                 regularizer: REGULARIZER_ALLOWED_TYPES = None,
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
                 regularizer: REGULARIZER_ALLOWED_TYPES = None,
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
                shape=input_shape[-1],
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
                 regularizer: REGULARIZER_ALLOWED_TYPES = None,
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
                 regularizer: REGULARIZER_ALLOWED_TYPES = None,
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
