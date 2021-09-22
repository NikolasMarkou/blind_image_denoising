r"""
Constructs the optimizer builder
"""

# ---------------------------------------------------------------------

__author__ = "Nikolas Markou"
__version__ = "0.1.0"
__license__ = "None"

# ---------------------------------------------------------------------


from typing import Dict
import tensorflow.keras as keras

# ---------------------------------------------------------------------


def optimizer_builder(
        config: Dict):
    """
    Instantiate an optimizer.

    :param config:
    :return:
    """
    # --- argument checking
    if not isinstance(config, dict):
        raise ValueError("config must be a dictionary")

    # --- read configuration
    decay_rate = config["decay_rate"]
    decay_steps = config["decay_steps"]
    learning_rate = config["learning_rate"]
    gradient_clipping_by_norm = config["gradient_clipping_by_norm"]

    # --- set up schedule
    lr_schedule = \
        keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=learning_rate,
            decay_steps=decay_steps,
            decay_rate=decay_rate)
    return \
        keras.optimizers.RMSprop(
            learning_rate=lr_schedule,
            global_clipnorm=gradient_clipping_by_norm),\
        lr_schedule

# ---------------------------------------------------------------------
