r"""build weight pruning strategies"""

# ---------------------------------------------------------------------

__author__ = "Nikolas Markou"
__version__ = "1.0.0"
__license__ = "MIT"

# ---------------------------------------------------------------------

import keras
import numpy as np
from enum import Enum
from typing import Dict, Callable, List, Tuple
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .custom_logger import logger


# ---------------------------------------------------------------------


def reshape_to_4d_to_2d(
        x: np.ndarray) -> Tuple[np.ndarray, Tuple]:
    """
    Reshapes tensor from 4d to 2d

    :param x:
    :return: reshaped array and transposed shape
    """
    # --- argument checking
    if x is None:
        raise ValueError("input cannot be empty")
    if len(x.shape) != 4:
        raise ValueError("only 4-dimensional tensors allowed")

    # --- reshape to 2d matrix
    x_t = np.transpose(x, axes=(3, 0, 1, 2))
    x_r = np.reshape(x_t, newshape=(x_t.shape[0], -1))
    return x_r, x_t.shape

# ---------------------------------------------------------------------


def reshape_to_2d_to_4d(
        x: np.ndarray,
        x_t_shape: Tuple) -> np.ndarray:
    """
    Reshaped tensor from 2d to 4d

    :param x:
    :param x_t_shape:
    :return:
    """
    # --- argument checking
    if x is None:
        raise ValueError("input cannot be empty")
    if len(x.shape) != 2:
        raise ValueError("only 2-dimensional tensors allowed")

    # reshape and transpose back to original shape
    x_r = np.reshape(x, newshape=x_t_shape)
    x_t = np.transpose(x_r, axes=(1, 2, 3, 0))
    return x_t


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

    """
    project weights and keep percentage of the the variance
    """
    PCA_PROJECTION = 4

    """
    sorts the weights by absolute value and zeros out the bottom X percent
    """
    DROP_BOTTOM = 5

    @staticmethod
    def from_string(type_str: str) -> "PruneStrategy":
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


def prune_strategy_helper(
        strategy: PruneStrategy,
        **kwargs) -> Callable:
    """
    builds the pruning function to be called

    :param strategy:
    :param kwargs:
    :return:
    """
    if strategy == PruneStrategy.MINIMUM_THRESHOLD:
        minimum_threshold = kwargs["minimum_threshold"]

        def fn(x: np.ndarray) -> np.ndarray:
            x[np.abs(x) < minimum_threshold] = 0.0
            return x
    elif strategy == PruneStrategy.MINIMUM_THRESHOLD_BIFURCATE:
        minimum_threshold = kwargs["minimum_threshold"]

        def fn(x: np.ndarray) -> np.ndarray:
            mask = np.abs(x) < minimum_threshold
            rand = \
                np.random.uniform(
                    -minimum_threshold * 2.0,
                    +minimum_threshold * 2.0,
                    size=mask.shape)
            x[mask] = rand[mask]
            x[np.abs(x) < minimum_threshold] = 0.0
            return x
    elif strategy == PruneStrategy.MINIMUM_THRESHOLD_SHRINKAGE:
        shrinkage = kwargs["shrinkage"]
        minimum_threshold = kwargs["minimum_threshold"]
        shrinkage_threshold = kwargs["shrinkage_threshold"]

        def fn(x: np.ndarray) -> np.ndarray:
            mask = np.abs(x) < shrinkage_threshold
            x[mask] = x[mask] * shrinkage
            x[np.abs(x) < minimum_threshold] = 0.0
            return x
    elif strategy == PruneStrategy.PCA_PROJECTION:
        # required variance
        variance = kwargs["variance"]
        # optional minimum threshold
        minimum_threshold = kwargs.get("minimum_threshold", -1)

        def fn(x: np.ndarray) -> np.ndarray:
            # reshape x which is 4d to 2d
            x_r, x_t_shape = reshape_to_4d_to_2d(x)
            # threshold again to zero very small values
            if minimum_threshold > 0.0:
                x_r[np.abs(x_r) < minimum_threshold] = 0.0
            pca = PCA(n_components=variance)
            pca.fit(x_r)
            # forward pass
            x_r = pca.transform(x_r)
            # inverse pass
            x_r = pca.inverse_transform(x_r)
            # reshape and transpose back to original shape
            x = reshape_to_2d_to_4d(x_r, x_t_shape=x_t_shape)
            return x
    elif strategy == PruneStrategy.DROP_BOTTOM:
        percentage = kwargs["percentage"]

        def fn(x: np.ndarray) -> np.ndarray:
            x_sorted = np.sort(np.abs(x), axis=None)
            x_threshold_index = int(np.round(len(x_sorted) * percentage))
            x_threshold = x_sorted[x_threshold_index]
            x[np.abs(x) < x_threshold] = 0.0
            return x
    elif strategy == PruneStrategy.NONE:
        def fn(x: np.ndarray) -> np.ndarray:
            return x
    else:
        raise ValueError("invalid strategy")
    return fn


# ---------------------------------------------------------------------

def prune_conv2d_weights(
        model: keras.Model,
        prune_fn: Callable) -> keras.Model:
    """
    go through the model and prune its weights given the config and strategy

    :param model: model to be pruned
    :param prune_fn: pruning strategy function
    :return: pruned model
    """

    for layer in model.layers:
        layer_config = layer.get_config()
        if "layers" not in layer_config:
            continue
        if not layer.trainable:
            continue
        for layer_internal in layer.layers:
            # --- make sure to prune only convolutions
            if not isinstance(layer_internal, keras.layers.Conv2D) and \
                    not isinstance(layer_internal, keras.layers.DepthwiseConv2D):
                # skipping because not convolution
                continue
            layer_internal_config = layer_internal.get_config()
            # --- skipping because not trainable
            if not layer_internal_config["trainable"]:
                continue
            # --- get layer weights, prune layer weights with the pruning strategy
            pruned_weights = [
                prune_fn(x)
                for x in layer_internal.get_weights()
            ]
            # --- set weights back to update the model
            layer_internal.set_weights(pruned_weights)
    return model


# ---------------------------------------------------------------------


def prune_function_builder(
        config: Dict) -> Callable:
    """
    Constructs a pruning function
    :param config: pruning configuration
    :return: pruning function
    """
    strategy = PruneStrategy.from_string(config["strategy"])
    strategy_config = config["config"]
    prune_fn = prune_strategy_helper(strategy, **strategy_config)

    def prune(model: keras.Model) -> keras.Model:
        return \
            prune_conv2d_weights(
                model=model,
                prune_fn=prune_fn)

    return prune


# ---------------------------------------------------------------------


def get_conv2d_weights(
        model: keras.Model) -> np.ndarray:
    """
    Get the conv2d weights from the model concatenated

    :param model: model to get the weights
    :return: list of weights
    """
    weights = []
    for layer in model.layers:
        layer_config = layer.get_config()
        if "layers" not in layer_config:
            continue
        if not layer.trainable:
            continue
        for layer_internal in layer.layers:
            # --- make sure to prune only convolutions
            if not isinstance(layer_internal, keras.layers.Conv2D) and \
                    not isinstance(layer_internal, keras.layers.DepthwiseConv2D):
                # skipping because not convolution
                continue
            layer_internal_config = layer_internal.get_config()
            # --- skipping because not trainable
            if not layer_internal_config["trainable"]:
                continue
            # --- get layer weights
            layer_weights = [
                x.flatten()
                for x in layer_internal.get_weights()
            ]
            weights.append(np.concatenate(layer_weights))
    if len(weights) == 0:
        return np.ndarray([])
    return np.concatenate(weights)

# ---------------------------------------------------------------------
