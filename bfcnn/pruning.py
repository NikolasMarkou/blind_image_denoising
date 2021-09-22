import copy
import keras
import itertools
import numpy as np
from enum import Enum
from typing import List
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator

# ==============================================================================


class PruneStrategy(Enum):
    MINIMUM_THRESHOLD = 1
    MINIMUM_THRESHOLD_BIFURCATE = 2
    MINIMUM_THRESHOLD_SHRINKAGE = 3

# ==============================================================================


def prune_conv2d_weights(
        model: keras.Model,
        parameters: dict,
        strategy: PruneStrategy = PruneStrategy.MINIMUM_THRESHOLD):
    for layer in model.layers:
        layer_config = layer.get_config()
        if "layers" not in layer_config:
            continue
        for l in layer.layers:
            if not isinstance(l, keras.layers.Conv2D):
                continue
            # get layer weights
            pruned_weights = []
            layer_weights = l.get_weights()
            if strategy == PruneStrategy.MINIMUM_THRESHOLD:
                minimum_weight_threshold = parameters["minimum_threshold"]
                for x in layer_weights:
                    x[np.abs(x) < minimum_weight_threshold] = 0.0
                    pruned_weights.append(x)
            elif strategy == PruneStrategy.MINIMUM_THRESHOLD_BIFURCATE:
                minimum_weight_threshold = parameters["minimum_threshold"]
                for x in layer_weights:
                    mask = np.abs(x) < minimum_weight_threshold
                    rand = np.random.uniform(-minimum_weight_threshold * 2.0,
                                             +minimum_weight_threshold * 2.0,
                                             size=mask.shape)
                    x[mask] = rand[mask]
                    x[np.abs(x) < minimum_weight_threshold] = 0.0
                    pruned_weights.append(x)
            elif strategy == PruneStrategy.MINIMUM_THRESHOLD_SHRINKAGE:
                shrinkage = parameters["shrinkage"]
                shrinkage_threshold = parameters["shrinkage_threshold"]
                to_zero_threshold = parameters["to_zero_threshold"]
                for x in layer_weights:
                    mask = np.abs(x) < shrinkage_threshold
                    x[mask] = x[mask] * shrinkage
                    x[np.abs(x) < to_zero_threshold] = 0.0
                    pruned_weights.append(x)
            else:
                raise NotImplementedError("not implemented strategy")
            l.set_weights(pruned_weights)
    return model

# ==============================================================================
