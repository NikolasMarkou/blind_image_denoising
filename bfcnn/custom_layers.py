import numpy as np
from typing import Any
import tensorflow as tf
from tensorflow import keras


# ---------------------------------------------------------------------


class TrainableMultiplier(tf.keras.layers.Layer):
    def __init__(self,
                 multiplier: float = 1.0,
                 regularizer=None,
                 trainable: bool = True,
                 activation: Any = "linear",
                 name=None,
                 **kwargs):
        super(TrainableMultiplier, self).__init__(
            trainable=trainable,
            name=name,
            **kwargs)
        self._w0 = None
        self._activation = None
        self._multiplier = multiplier
        self._activation_type = activation
        self._regularizer = keras.regularizers.get(regularizer)

    def build(self, input_shape):
        def init_w0_fn(shape, dtype):
            return np.ones(shape, dtype=np.float32) * self._multiplier

        self._w0 = \
            self.add_weight(
                shape=[1],
                trainable=True,
                name="multiplier",
                initializer=init_w0_fn,
                regularizer=self._regularizer)
        self._activation = keras.layers.Activation(self._activation_type)
        super(TrainableMultiplier, self).build(input_shape)

    def call(self, inputs):
        return self._activation(self._w0 * inputs)

    def get_config(self):
        return {
            "multiplier": self._w0.numpy(),
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
                 name=None,
                 **kwargs):
        super(GeluLayer, self).__init__(
            trainable=trainable,
            name=name,
            **kwargs)
        self._offset = None
        self._multiplier = None

    def build(self, input_shape):
        def init_offset_fn(shape, dtype):
            return np.zeros(shape, dtype=np.float32)

        def init_multiplier_fn(shape, dtype):
            return np.ones(shape, dtype=np.float32)

        self._offset = \
            self.add_weight(
                shape=[1],
                trainable=True,
                regularizer=None,
                name="offset",
                initializer=init_offset_fn)
        self._multiplier = \
            self.add_weight(
                shape=[1],
                trainable=True,
                regularizer=None,
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
            "multiplier": self._multiplier.numpy()
        }

    def compute_output_shape(self, input_shape):
        return input_shape

# ---------------------------------------------------------------------
