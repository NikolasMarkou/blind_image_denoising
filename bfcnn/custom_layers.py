# ---------------------------------------------------------------------

__author__ = "Nikolas Markou"
__version__ = "1.0.0"
__license__ = "MIT"

# ---------------------------------------------------------------------

import numpy as np
import tensorflow as tf
from tensorflow.keras import regularizers

# ---------------------------------------------------------------------


class TrainableMultiplier(tf.keras.layers.Layer):
    def __init__(self,
                 multiplier: float = 1.0,
                 regularizer=None,
                 trainable: bool = True,
                 name=None,
                 **kwargs):
        super(TrainableMultiplier, self).__init__(
            trainable=trainable,
            name=name,
            **kwargs)
        self._w1 = None
        self._multiplier = multiplier
        self._weight_regularizer = regularizers.get(regularizer)

    def build(self, input_shape):
        def init_fn(shape, dtype):
            return np.ones(shape, dtype=np.float32) * self._multiplier
        self._w1 = \
            self.add_weight(
                name="multiplier",
                shape=[1],
                regularizer=self._weight_regularizer,
                initializer=init_fn,
                trainable=self.trainable)
        super(TrainableMultiplier, self).build(input_shape)

    def call(self, inputs):
        return inputs * self._w1

    def get_config(self):
        return {
            "multiplier": self._w1.numpy()
        }

    def compute_output_shape(self, input_shape):
        return input_shape
# ---------------------------------------------------------------------
