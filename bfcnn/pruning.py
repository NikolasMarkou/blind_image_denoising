r"""build weight pruning strategies"""

# ---------------------------------------------------------------------

__author__ = "Nikolas Markou"
__version__ = "0.1.0"
__license__ = "None"

# ---------------------------------------------------------------------

import copy
import keras
import itertools
import numpy as np
from enum import Enum
from typing import List, Dict, Callable

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .custom_logger import logger

# ---------------------------------------------------------------------


class PruneStrategy(Enum):
    """
    do nothing, experimental
    """
    NONE = 0
    """
    every weight in a conv2d below a threshold becomes zero
    """
    MINIMUM_THRESHOLD = 1
    """
    every weight in a conv2d below a threshold becomes random re-assigned
    """
    MINIMUM_THRESHOLD_BIFURCATE = 2
    """
    every weight in a conv2d below a threshold gets shrunk by shrink percentage
    """
    MINIMUM_THRESHOLD_SHRINKAGE = 3

    @staticmethod
    def from_string(type_str: str):
        # --- argument checking
        if type_str is None:
            raise ValueError("type_str must not be null")
        if not isinstance(type_str, str):
            raise ValueError("type_str must be string")
        if len(type_str.strip()) <= 0:
            raise ValueError("stripped type_str must not be empty")

        # --- clean string and get
        return PruneStrategy[type_str.strip().upper()]

    def to_string(self) -> str:
        return self.name

# ---------------------------------------------------------------------


def prune_conv2d_weights(
        model: keras.Model,
        config: Dict,
        strategy: PruneStrategy = PruneStrategy.MINIMUM_THRESHOLD):
    """

    """
    for layer in model.layers:
        layer_config = layer.get_config()
        if "layers" not in layer_config:
            continue
        if not layer.trainable:
            continue
        for layer_internal in layer.layers:
            # skip non Conv2D operation
            if not isinstance(layer_internal, keras.layers.Conv2D):
                # skipping because not convolution
                continue
            layer_internal_config = layer_internal.get_config()
            if not layer_internal_config["trainable"]:
                # skipping because not trainable
                continue
            # ---
            pruned_weights = []
            # get layer weights
            layer_weights = layer_internal.get_weights()
            # ---
            if strategy == PruneStrategy.NONE:
                pruned_weights = layer_weights
            elif strategy == PruneStrategy.MINIMUM_THRESHOLD:
                minimum_weight_threshold = config["minimum_threshold"]
                for x in layer_weights:
                    x[np.abs(x) < minimum_weight_threshold] = 0.0
                    pruned_weights.append(x)
            elif strategy == PruneStrategy.MINIMUM_THRESHOLD_BIFURCATE:
                minimum_weight_threshold = config["minimum_threshold"]
                for x in layer_weights:
                    mask = np.abs(x) < minimum_weight_threshold
                    rand = \
                        np.random.uniform(
                            -minimum_weight_threshold * 2.0,
                            +minimum_weight_threshold * 2.0,
                            size=mask.shape)
                    x[mask] = rand[mask]
                    x[np.abs(x) < minimum_weight_threshold] = 0.0
                    pruned_weights.append(x)
            elif strategy == PruneStrategy.MINIMUM_THRESHOLD_SHRINKAGE:
                shrinkage = config["shrinkage"]
                shrinkage_threshold = config["shrinkage_threshold"]
                minimum_threshold = config["minimum_threshold"]
                for x in layer_weights:
                    mask = np.abs(x) < shrinkage_threshold
                    x[mask] = x[mask] * shrinkage
                    x[np.abs(x) < minimum_threshold] = 0.0
                    pruned_weights.append(x)
            else:
                raise NotImplementedError("not implemented strategy")
            layer_internal.set_weights(pruned_weights)
    return model

# ==============================================================================


def prune_function_builder(
        config: Dict) -> Callable:
    """
    Constructs a pruning function
    """
    strategy = PruneStrategy.from_string(config["strategy"])

    def prune(model: keras.Model) -> keras.Model:
        return \
            prune_conv2d_weights(
                model=model,
                config=config,
                strategy=strategy)

    return prune

# ==============================================================================
