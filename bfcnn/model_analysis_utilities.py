from typing import Dict
import tensorflow as tf

# ---------------------------------------------------------------------


def compute_layer_distance_from_top(
        model: tf.keras.Model) -> Dict[str, int]:
    # --- argument checking
    if model is None:
        raise ValueError("model cannot be None")

    # --- set variables
    distances = {}

    # --- go through each layer
    for layer in model.layers:
        layer_config = layer.get_config()
        if "layers" not in layer_config:
            continue
        for layer_internal in layer.layers:
            pass
    return distances

# ---------------------------------------------------------------------


def compute_weight_distance_from_top(
        model: tf.keras.Model) -> Dict[str, int]:
    for w in model.trainable_weights:
        pass
    return {}

# ---------------------------------------------------------------------

