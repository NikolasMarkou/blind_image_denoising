import tensorflow as tf
from tensorflow import keras
from typing import Dict, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .custom_logger import logger

# ---------------------------------------------------------------------


def schedule_builder(
        config: Dict) -> keras.optimizers.schedules.LearningRateSchedule:
    # --- argument checking
    if not isinstance(config, Dict):
        raise ValueError("config must be a dictionary")

    # --- select type
    schedule_type = config.get("type", None)

    # --- sanity checks
    if schedule_type is None:
        raise ValueError("schedule_type cannot be None")
    if not isinstance(schedule_type, str):
        raise ValueError("schedule_type must be a string")
    schedule_type = schedule_type.lower().strip()

    # --- select schedule
    schedule = None
    params = config.get("config", {})
    if schedule_type == "exponential_decay":
        decay_rate = params["decay_rate"]
        decay_steps = params["decay_steps"]
        learning_rate = params["learning_rate"]
        schedule = \
            keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=learning_rate,
                decay_steps=decay_steps,
                decay_rate=decay_rate)
    elif schedule_type == "cosine_decay_restarts":
        decay_steps = params["decay_steps"]
        learning_rate = params["learning_rate"]
        t_mul = params.get("t_mul", 2.0)
        m_mul = params.get("m_mul", 1.0)
        alpha = params.get("alpha", 0.0)
        schedule = \
            keras.optimizers.schedules.CosineDecayRestarts(
                initial_learning_rate=learning_rate,
                first_decay_steps=decay_steps,
                t_mul=t_mul,
                m_mul=m_mul,
                alpha=alpha)
    elif schedule_type == "cosine_decay":
        decay_steps = params["decay_steps"]
        learning_rate = params["learning_rate"]
        schedule = \
            keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=learning_rate,
                decay_steps=decay_steps,
                alpha=0.0,
                name=None)
    else:
        raise ValueError(f"don't know how to handle {schedule_type}")
    logger.info(f"created schedule: {schedule}")
    return schedule

# ---------------------------------------------------------------------


def optimizer_builder(
        config: Dict) -> Tuple[keras.optimizers.Optimizer, keras.optimizers.schedules.LearningRateSchedule]:
    """
    Instantiate an optimizer.

    :param config:
    :return:
    """
    # --- argument checking
    if not isinstance(config, Dict):
        raise ValueError("config must be a dictionary")

    # --- read configuration
    schedule_config = config["schedule"]
    gradient_clipping_by_norm = config["gradient_clipping_by_norm"]

    # --- set up schedule
    lr_schedule = \
        schedule_builder(config=schedule_config)

    return \
        keras.optimizers.RMSprop(
            learning_rate=lr_schedule,
            global_clipnorm=gradient_clipping_by_norm),\
        lr_schedule

# ---------------------------------------------------------------------
