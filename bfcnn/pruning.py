r"""build weight pruning strategies"""

# ---------------------------------------------------------------------

__author__ = "Nikolas Markou"
__version__ = "1.0.0"
__license__ = "MIT"

# ---------------------------------------------------------------------

import keras
import numpy as np
from enum import Enum
from typing import Dict, Callable, List
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

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

    """
    project weights and keep percentage of the the variance
    """
    PCA_PROJECTION = 4

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


def prune_conv2d_weights(
        model: keras.Model,
        strategy: PruneStrategy,
        **kwargs) -> keras.Model:
    """
    go through the model and prune its weights given the config and strategy

    :param model: model to be pruned
    :param strategy: pruning strategy
    :return: pruned model
    """
    if strategy == PruneStrategy.MINIMUM_THRESHOLD:
        minimum_weight_threshold = kwargs["minimum_threshold"]
    elif strategy == PruneStrategy.MINIMUM_THRESHOLD_BIFURCATE:
        minimum_weight_threshold = kwargs["minimum_threshold"]
    elif strategy == PruneStrategy.MINIMUM_THRESHOLD_SHRINKAGE:
        shrinkage = kwargs["shrinkage"]
        minimum_threshold = kwargs["minimum_threshold"]
        shrinkage_threshold = kwargs["shrinkage_threshold"]
    elif strategy == PruneStrategy.PCA_PROJECTION:
        # required variance
        variance = kwargs["variance"]
        # optional minimum threshold
        minimum_threshold = kwargs.get("minimum_threshold", -1)
    else:
        pass

    for layer in model.layers:
        layer_config = layer.get_config()
        if "layers" not in layer_config:
            continue
        if not layer.trainable:
            continue
        for layer_internal in layer.layers:
            # make sure to prune only convolutions
            if not isinstance(layer_internal, keras.layers.Conv2D) and \
                    not isinstance(layer_internal, keras.layers.DepthwiseConv2D):
                # skipping because not convolution
                continue
            layer_internal_config = layer_internal.get_config()
            # skipping because not trainable
            if not layer_internal_config["trainable"]:
                continue
            # ---
            pruned_weights = []
            # get layer weights
            layer_weights = layer_internal.get_weights()
            # ---
            if strategy == PruneStrategy.NONE:
                pruned_weights = layer_weights
            elif strategy == PruneStrategy.MINIMUM_THRESHOLD:
                for x in layer_weights:
                    x[np.abs(x) < minimum_weight_threshold] = 0.0
                    pruned_weights.append(x)
            elif strategy == PruneStrategy.MINIMUM_THRESHOLD_BIFURCATE:
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
                for x in layer_weights:
                    mask = np.abs(x) < shrinkage_threshold
                    x[mask] = x[mask] * shrinkage
                    x[np.abs(x) < minimum_threshold] = 0.0
                    pruned_weights.append(x)
            elif strategy == PruneStrategy.PCA_PROJECTION:
                for x in layer_weights:
                    if minimum_threshold != -1:
                        x[np.abs(x) < minimum_threshold] = 0.0
                    # reshape x which is 4d to 2d
                    x_transpose = np.transpose(x, axes=(3, 0, 1, 2))
                    x_transpose_shape = x_transpose.shape
                    x_reshaped = \
                        np.reshape(
                            x_transpose,
                            newshape=(
                                x_transpose_shape[0],
                                np.prod(x_transpose_shape[1:])))
                    mms = MinMaxScaler()
                    x_reshaped = mms.fit_transform(x_reshaped)
                    pca = PCA(n_components=variance)
                    pca.fit(x_reshaped)
                    # number of components missing
                    diff_components = x_reshaped.shape[1] - pca.n_components_
                    x_reshaped = pca.transform(x_reshaped)
                    # fill in zeros
                    if diff_components > 0:
                        x_reshaped = \
                            np.concatenate(
                                [x_reshaped, np.zeros(shape=(x_reshaped.shape[0], diff_components))],
                                axis=1)
                    x_reshaped = mms.inverse_transform(x_reshaped)
                    x_reshaped = \
                        np.reshape(
                            x_reshaped,
                            newshape=x_transpose_shape)
                    x_reshaped = \
                        np.transpose(
                            x_reshaped,
                            axes=(1, 2, 3, 0))
                    pruned_weights.append(x_reshaped)
            else:
                raise NotImplementedError("not implemented strategy")
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

    def prune(model: keras.Model) -> keras.Model:
        return \
            prune_conv2d_weights(
                model=model,
                strategy=strategy,
                **strategy_config)

    return prune

# ---------------------------------------------------------------------


def get_conv2d_weights(
        model: keras.Model,
        verbose: bool = False) -> np.ndarray:
    """
    Get the conv2d weights from the model concatenated

    :param model: model to get the weights
    :param verbose: if true show more messages
    :return: list of weights
    """
    weights = []
    for layer in model.layers:
        layer_config = layer.get_config()
        if "layers" not in layer_config:
            continue
        # get weights of the outer layer
        layer_weights = layer.get_weights()
        # --- iterate layer and get the weights internally
        for i, layer_internal in enumerate(layer_config["layers"]):
            layer_internal_name = layer_internal["name"]
            layer_internal_class = layer_internal["class_name"]
            # make sure to prune only convolutions
            if not layer_internal_class == "DepthwiseConv2D" and \
                    not layer_internal_class == "Conv2D":
                continue
            layer_internal_config = layer_internal["config"]
            # make sure to prune only trainable
            layer_trainable = layer_internal_config.get("trainable", False)
            if not layer_trainable:
                continue
            if verbose:
                logger.info("pruning layer: {0}".format(layer_internal_name))
            if i >= len(layer_weights):
                continue
            for w in layer_weights[i]:
                w_flat = w.flatten()
                weights.append(w_flat)
    if len(weights) == 0:
        return np.ndarray([])
    return np.concatenate(weights)

# ---------------------------------------------------------------------
