# ---------------------------------------------------------------------

__author__ = "Nikolas Markou"
__version__ = "1.0.0"
__license__ = "MIT"

# ---------------------------------------------------------------------

import numpy as np
from typing import Any
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers

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
        self._w1 = None
        self._activation_w1 = None
        self._multiplier = multiplier
        self._activation = activation
        self._weight_regularizer = regularizers.get(regularizer)

    def build(self, input_shape):
        def init_w0_fn(shape, dtype):
            return np.ones(shape, dtype=np.float32) * self._multiplier

        def init_w1_fn(shape, dtype):
            return np.zeros(shape, dtype=np.float32)

        self._w0 = \
            self.add_weight(
                name="constant",
                shape=[1],
                regularizer=None,
                initializer=init_w0_fn,
                trainable=False)
        self._w1 = \
            self.add_weight(
                name="multiplier",
                shape=[1],
                regularizer=self._weight_regularizer,
                initializer=init_w1_fn,
                trainable=self.trainable)
        self._activation = keras.layers.Activation(self._activation)
        super(TrainableMultiplier, self).build(input_shape)

    def call(self, inputs):
        return self._activation(
            (self._w0 * (1.0 + self._w1)) * inputs)

    def get_config(self):
        return {
            "constant": self._w0.numpy(),
            "multiplier": self._w1.numpy(),
            "activation": self._activation
        }

    def compute_output_shape(self, input_shape):
        return input_shape

# ---------------------------------------------------------------------
