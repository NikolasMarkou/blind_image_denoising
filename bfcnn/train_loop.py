r"""Constructs model, inputs, and training environment."""

# ---------------------------------------------------------------------

__author__ = "Nikolas Markou"
__version__ = "0.1.0"
__license__ = "None"

# ---------------------------------------------------------------------

import os
import time
import tensorflow as tf
from pathlib import Path
from typing import Union, Dict

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .utilities import *
from .model import model_builder
from .visualize import visualize
from .custom_logger import logger
from .dataset import dataset_builder
from .loss import loss_function_builder
from .optimizer import optimizer_builder

# ---------------------------------------------------------------------


def train_loop(
        pipeline_config_path: Union[str, Dict, Path],
        model_dir: Union[str, Path]):
    """
    Trains a blind image denoiser

    This method:
        1. Processes the pipeline configs
        2. (Optionally) saves the as-run config
        3. Builds the model & optimizer
        4. Gets the training input data
        5. Loads a fine-tuning detection
        6. Loops over the train data
        7. Checkpoints the model every `checkpoint_every_n` training steps.
        8. Logs the training metrics as TensorBoard summaries.

    :param pipeline_config_path: filepath to the configuration
    :param model_dir: directory to save checkpoints into
    :return:
    """
    # --- load configuration
    config = load_config(pipeline_config_path)

    # --- create model_dir if not exist
    if not os.path.isdir(str(model_dir)):
        # if path does not exist attempt to make it
        Path(str(model_dir)).mkdir(parents=True, exist_ok=True)
        # if it fails again throw exception
        if not os.path.isdir(str(model_dir)):
            raise ValueError("Model directory [{0}] is not valid".format(
                model_dir))

    # --- build the model
    model_denoise, model_normalize, model_denormalize = \
        model_builder(config=config["model"])

    # summary of model
    model_denoise.summary(print_fn=logger.info)
    # save model so we can visualize it easier
    model_denoise.save(os.path.join(model_dir, "model_denoise.h5"))

    # --- build dataset
    dataset_res = dataset_builder(config=config["dataset"])
    dataset = dataset_res["dataset"]
    augmentation_fn = dataset_res["augmentation"]

    # --- build loss function
    loss_fn = loss_function_builder(config=config["loss"])

    # --- build optimizer
    optimizer, lr_schedule = \
        optimizer_builder(config=config["train"]["optimizer"])

    # --- create the help variables
    global_step = tf.Variable(
        0, trainable=False, dtype=tf.dtypes.int64, name="global_step")
    global_epoch = tf.Variable(
        0, trainable=False, dtype=tf.dtypes.int64, name="global_epoch")
    summary_writer = tf.summary.create_file_writer(model_dir)

    # --- get the train configuration
    train_config = config["train"]
    epochs = train_config["epochs"]
    # how many times to run the model on a single batch
    iterations_choice = train_config.get("iterations_choice", [1])
    total_steps = train_config.get("total_steps", -1)
    # how many checkpoints to keep
    checkpoints_to_keep = train_config.get("checkpoints_to_keep", 3)
    # checkpoint every so many steps
    checkpoint_every = train_config.get("checkpoint_every", -1)
    # how many steps to make a visualization
    visualization_every = train_config["visualization_every"]
    # how many visualizations to show
    visualization_number = train_config.get("visualization_number", 5)

    # --- train the model
    with summary_writer.as_default():
        checkpoint = \
            tf.train.Checkpoint(
                step=global_step,
                epoch=global_epoch,
                optimizer=optimizer,
                model_denoise=model_denoise,
                model_normalize=model_normalize,
                model_denormalize=model_denormalize)
        manager = \
            tf.train.CheckpointManager(
                checkpoint=checkpoint,
                directory=model_dir,
                max_to_keep=checkpoints_to_keep)
        status =\
            checkpoint.restore(manager.latest_checkpoint).expect_partial()
        trainable_weights = \
            model_denoise.trainable_weights

        for epoch in range(int(global_epoch), int(epochs), 1):
            logger.info("epoch: {0}, step: {1}".format(epoch, int(global_step)))

            # --- iterate over the batches of the dataset
            for input_batch in dataset:
                start_time = time.time()

                # declare variables
                loss_map = None
                tmp_prediction_batch = None

                # augment data
                input_batch, noisy_batch, noise_std = \
                    augmentation_fn(input_batch)

                prediction_batch = \
                    model_normalize(noisy_batch, training=False)

                # run multiple iterations for stability
                iterations = np.random.choice(iterations_choice)

                for iteration in range(iterations):
                    # Open a GradientTape to record the operations run
                    # during the forward pass, which enables auto-differentiation.
                    with tf.GradientTape() as tape:
                        # Run the forward pass of the layer.
                        # The operations that the layer applies
                        # to its inputs are going to be recorded
                        # on the GradientTape.
                        prediction_batch = \
                            model_denoise(
                                prediction_batch,
                                training=True)
                        tmp_prediction_batch = \
                            model_denormalize(prediction_batch, training=False)

                        # compute the loss value for this mini-batch
                        loss_map = loss_fn(
                            difficulty=noise_std,
                            input_batch=input_batch,
                            noisy_batch=noisy_batch,
                            prediction_batch=tmp_prediction_batch,
                            model_losses=model_denoise.losses)

                        # Use the gradient tape to automatically retrieve
                        # the gradients of the trainable variables
                        # with respect to the loss.
                        grads = \
                            tape.gradient(
                                loss_map["mean_total_loss"],
                                trainable_weights)

                        # Run one step of gradient descent by updating
                        # the value of the variables to minimize the loss.
                        optimizer.apply_gradients(
                            zip(grads, trainable_weights))

                # --- add loss summaries for tensorboard
                if loss_map is not None:
                    for name, key in [
                        ("loss/mae", "mae_loss"),
                        ("loss/total", "mean_total_loss"),
                        ("quality/mae_noise", "mae_noise"),
                        ("loss/regularization", "regularization_loss"),
                        ("quality/mae_improvement", "mae_improvement")
                    ]:
                        tf.summary.scalar(
                            name,
                            loss_map[key], step=global_step)

                # --- add image prediction for tensorboard
                if global_step % visualization_every == 0:
                    random_batch = \
                        tf.random.truncated_normal(
                            mean=0.0,
                            stddev=0.25,
                            shape=tf.shape(input_batch))
                    for i in range(3):
                        random_batch = \
                            model_denoise(
                                random_batch,
                                training=False)
                    random_batch = \
                        model_denormalize(random_batch, training=False)
                    visualize(
                        global_step=global_step,
                        input_batch=input_batch,
                        noisy_batch=noisy_batch,
                        random_batch=random_batch,
                        prediction_batch=tmp_prediction_batch,
                        visualization_number=visualization_number)

                # --- check if it is time to save a checkpoint
                if checkpoint_every > 0:
                    if global_step % checkpoint_every == 0:
                        logger.info("checkpoint at step: {0}".format(
                            int(global_step)))
                        manager.save()

                # --- keep time of steps per second
                stop_time = time.time()
                step_time = stop_time - start_time

                tf.summary.scalar(
                    "training/steps_per_second",
                    1.0 / (step_time + 0.00001),
                    step=global_step)

                tf.summary.scalar(
                    "training/epoch",
                    int(global_epoch),
                    step=global_step)

                tf.summary.scalar(
                    "training/learning_rate",
                    lr_schedule(int(global_step)),
                    step=global_step)

                # ---
                global_step.assign_add(1)

                # --- check if total steps reached
                if total_steps > 0:
                    if total_steps <= global_step:
                        logger.info("total_steps reached [{0}]".format(
                            total_steps))
                        break
            logger.info("checkpoint at epoch: {0}".format(
                int(global_epoch)))
            manager.save()
            global_epoch.assign_add(1)

        # --- save checkpoint before exiting
        logger.info("checkpoint at step: {0}".format(
            int(global_step)))
        manager.save()
    logger.info("finished training")

# ---------------------------------------------------------------------
