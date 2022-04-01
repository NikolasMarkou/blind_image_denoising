# ---------------------------------------------------------------------

__author__ = "Nikolas Markou"
__version__ = "1.0.0"
__license__ = "MIT"

# ---------------------------------------------------------------------

import numpy as np
import tensorflow as tf
from typing import Any
# ---------------------------------------------------------------------


class TrainableMultiplier(tf.keras.layers.Layer):

    def __init__(self,
                 multiplier: float,
                 regularizer: Any = tf.keras.regularizers.l1(0.01),
                 **kwargs):
        super(TrainableMultiplier, self).__init__(**kwargs)

        def init_fn(shape, dtype):
            return np.ones(shape, dtype=np.float32) * multiplier

        self.w1 = \
            self.add_weight(
                shape=[1],
                regularizer=regularizer,
                initializer=init_fn,
                trainable=True)

    def call(self, inputs):
        return inputs * self.w1

    def get_config(self):
        return {"multiplier": self.w1}

# ---------------------------------------------------------------------
