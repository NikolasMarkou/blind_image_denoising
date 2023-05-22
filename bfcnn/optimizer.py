from tensorflow import keras
from typing import Dict, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .custom_logger import logger
from .constants import CONFIG_STR, TYPE_STR

# ---------------------------------------------------------------------


def schedule_builder(
        config: Dict) -> keras.optimizers.schedules.LearningRateSchedule:
    # --- argument checking
    if not isinstance(config, Dict):
        raise ValueError("config must be a dictionary")

    # --- select type
    schedule_type = config.get(TYPE_STR, None)

    # --- sanity checks
    if schedule_type is None:
        raise ValueError("schedule_type cannot be None")
    if not isinstance(schedule_type, str):
        raise ValueError("schedule_type must be a string")

    # --- select schedule
    schedule = None
    params = config.get(CONFIG_STR, {})
    schedule_type = schedule_type.lower().strip()
    logger.info(f"building schedule: {schedule_type}, with params: {params}")

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
        m_mul = params.get("m_mul", 0.9)
        alpha = params.get("alpha", 0.001)
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
    # ---
    return schedule

# ---------------------------------------------------------------------


def optimizer_builder(
        config: Dict) -> \
        Tuple[keras.optimizers.Optimizer,
              keras.optimizers.schedules.LearningRateSchedule]:
    """
    Instantiate an optimizer.

    :param config:
    :return: optimizer and learning schedule
    """
    # --- argument checking
    if not isinstance(config, Dict):
        raise ValueError("config must be a dictionary")

    # --- set up schedule
    schedule_config = config["schedule"]
    lr_schedule = \
        schedule_builder(config=schedule_config)

    # --- gradient clipping configuration
    # clip by value (every gradient independently)
    gradient_clipvalue = config.get("gradient_clipping_by_value", None)
    # clip by norm (every gradient independently)
    gradient_clipnorm = config.get("gradient_clipping_by_norm_local", None)
    # clip by norm all together
    gradient_global_clipnorm = config.get("gradient_clipping_by_norm", None)
    optimizer_type = config.get("type", "RMSprop").strip().upper()

    # --- build optimizer
    if optimizer_type == "RMSPROP":
        # RMSprop optimizer
        rho = config.get("rho", 0.9)
        momentum = config.get("momentum", 0.0)
        epsilon = config.get("epsilon", 1e-07)
        centered = config.get("centered", False)
        optimizer_parameters = dict(
            name="RMSprop",
            rho=rho,
            epsilon=epsilon,
            centered=centered,
            momentum=momentum,
            learning_rate=lr_schedule,
            clipvalue=gradient_clipvalue,
            clipnorm=gradient_clipnorm,
            global_clipnorm=gradient_global_clipnorm)
        optimizer = keras.optimizers.RMSprop(**optimizer_parameters)
    elif optimizer_type == "ADAM":
        # Adam optimizer
        beta_1 = config.get("beta_1", 0.9)
        beta_2 = config.get("beta_2", 0.999)
        epsilon = config.get("epsilon", 1e-07)
        amsgrad = config.get("amsgrad", False)
        optimizer_parameters = dict(
            name="Adam",
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=epsilon,
            amsgrad=amsgrad,
            learning_rate=lr_schedule,
            clipvalue=gradient_clipvalue,
            clipnorm=gradient_clipnorm,
            global_clipnorm=gradient_global_clipnorm)
        optimizer = keras.optimizers.Adam(**optimizer_parameters)
    elif optimizer_type == "ADADELTA":
        # Adadelta optimizer
        rho = config.get("rho", 0.9)
        epsilon = config.get("epsilon", 1e-07)
        optimizer_parameters = dict(
            name="Adadelta",
            rho=rho,
            epsilon=epsilon,
            learning_rate=lr_schedule,
            clipvalue=gradient_clipvalue,
            clipnorm=gradient_clipnorm,
            global_clipnorm=gradient_global_clipnorm)
        optimizer = keras.optimizers.Adadelta(**optimizer_parameters)
    else:
        raise ValueError(
            f"don't know how to handle [{optimizer_type}]")

    return \
        optimizer,\
        lr_schedule

# ---------------------------------------------------------------------
