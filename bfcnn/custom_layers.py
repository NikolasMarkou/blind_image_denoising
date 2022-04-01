# ---------------------------------------------------------------------

__author__ = "Nikolas Markou"
__version__ = "1.0.0"
__license__ = "MIT"

# ---------------------------------------------------------------------

import numpy as np
import tensorflow as tf
from typing import Any
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
        self._weight_regularizer = regularizers.get(regularizer)

        def init_fn(shape, dtype):
            return np.ones(shape, dtype=np.float32) * multiplier

        self._w1 = \
            self.add_weight(
                name="trainable_multiplier",
                shape=[1],
                regularizer=self._weight_regularizer,
                initializer=init_fn,
                trainable=True)

    def call(self, inputs):
        return inputs * self._w1

    def get_config(self):
        return {"multiplier": self._w1.numpy()}

# ---------------------------------------------------------------------
