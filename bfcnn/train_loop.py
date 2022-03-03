r"""Constructs model, inputs, and training environment."""

# ---------------------------------------------------------------------

__author__ = "Nikolas Markou"
__version__ = "1.0.0"
__license__ = "MIT"

# ---------------------------------------------------------------------

import os
import time

import keras.backend
import tensorflow as tf
from pathlib import Path
from typing import Union, Dict

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .pruning import *
from .constants import *
from .utilities import *
from .visualize import visualize
from .custom_logger import logger
from .dataset import dataset_builder
from .loss import loss_function_builder
from .optimizer import optimizer_builder
from .model_denoise import model_builder as model_denoise_builder
from .model_discriminate import model_builder as model_discriminate_builder


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
    trace_every = train_config.get("trace_every", 100)
    weight_buckets = train_config.get("weight_buckets", 100)
    total_steps = train_config.get("total_steps", -1)
    # how many checkpoints to keep
    checkpoints_to_keep = train_config.get("checkpoints_to_keep", 3)
    # checkpoint every so many steps
    checkpoint_every = train_config.get("checkpoint_every", -1)
    # how many steps to make a visualization
    visualization_every = train_config["visualization_every"]
    # how many visualizations to show
    visualization_number = train_config.get("visualization_number", 5)
    # how many times the random batch will be iterated
    random_batch_iterations = train_config.get("random_batch_iterations", 1)
    # min allowed difference
    random_batch_min_difference = \
        train_config.get(
            "random_batch_min_difference",
            0.1)
    # size of the random batch
    random_batch_size = \
        [visualization_number] + \
        train_config.get("random_batch_size", [256, 256, 3])
    # prune strategy
    prune_config = \
        train_config.get("prune", {"strategy": "none"})
    use_prune = \
        PruneStrategy.from_string(prune_config["strategy"]) != PruneStrategy.NONE
    prune_start_epoch = prune_config.get("start_epoch", 0)
    prune_function = prune_function_builder(prune_config)
    use_discriminator = MODEL_DISCRIMINATE_STR in config
    model_discriminate = None

    # --- build the denoise model
    tf.summary.trace_on(graph=True, profiler=False)
    with summary_writer.as_default():
        model_denoise, model_normalize, model_denormalize = \
            model_denoise_builder(config=config[MODEL_DENOISE_STR])

        # The function to be traced.
        @tf.function()
        def optimized_model(x_input):
            return model_denoise(x_input)

        x = \
            tf.random.truncated_normal(
                seed=0,
                mean=0.0,
                stddev=0.25,
                shape=random_batch_size)
        _ = optimized_model(x)
        tf.summary.trace_export(
            step=0,
            name="denoiser")
        tf.summary.flush()
        tf.summary.trace_off()
    # summary of model
    model_denoise.summary(print_fn=logger.info)
    # save model so we can visualize it easier
    model_denoise.save(
        os.path.join(model_dir, MODEL_DENOISE_DEFAULT_NAME_STR))

    # --- build the discriminate model
    if use_discriminator:
        tf.summary.trace_on(graph=True, profiler=False)
        with summary_writer.as_default():
            model_discriminate, _, _ = \
                model_discriminate_builder(config=config[MODEL_DISCRIMINATE_STR])

            # The function to be traced.
            @tf.function()
            def optimized_model(x_input):
                return model_discriminate(x_input)

            x = \
                tf.random.truncated_normal(
                    seed=0,
                    mean=0.0,
                    stddev=0.25,
                    shape=random_batch_size)
            _ = optimized_model(x)
            tf.summary.trace_export(
                step=0,
                name="discriminator")
            tf.summary.flush()
            tf.summary.trace_off()
        # summary of model
        model_discriminate.summary(print_fn=logger.info)
        # save model so we can visualize it easier
        model_discriminate.save(
            os.path.join(model_dir, MODEL_DISCRIMINATE_DEFAULT_NAME_STR))

    # --- create random image and iterate through the model
    def create_random_batch():
        x_iteration = 0
        x_diff = 1
        x = \
            tf.random.truncated_normal(
                seed=0,
                mean=0.0,
                stddev=0.25,
                shape=random_batch_size)
        while x_iteration < random_batch_iterations and \
                x_diff > random_batch_min_difference:
            x_tmp = \
                model_denoise(
                    x,
                    training=False)
            x_tmp = \
                tf.clip_by_value(
                    x_tmp,
                    clip_value_min=-0.5,
                    clip_value_max=+0.5)
            x_diff = tf.abs(x - x_tmp)
            x_diff = \
                tf.reduce_mean(
                    x_diff,
                    axis=[1, 2, 3])

            x_diff = \
                tf.reduce_sum(
                    x_diff,
                    axis=[0])
            x = x_tmp
            x_iteration += 1
        return \
            model_denormalize(
                x,
                training=False)

    # --- train the model
    with summary_writer.as_default():
        checkpoint = \
            tf.train.Checkpoint(
                step=global_step,
                epoch=global_epoch,
                optimizer=optimizer,
                model_denoise=model_denoise,
                model_normalize=model_normalize,
                model_denormalize=model_denormalize,
                model_discriminate=model_discriminate)
        manager = \
            tf.train.CheckpointManager(
                checkpoint=checkpoint,
                directory=model_dir,
                max_to_keep=checkpoints_to_keep)
        status = \
            checkpoint.restore(manager.latest_checkpoint).expect_partial()

        for epoch in range(int(global_epoch), int(epochs), 1):
            logger.info("epoch: {0}, step: {1}".format(epoch, int(global_step)))

            # --- pruning strategy
            if use_prune and epoch >= prune_start_epoch:
                logger.info("pruning weights")
                model_denoise = \
                    prune_function(model=model_denoise)

            # --- iterate over the batches of the dataset
            for input_batch in dataset:
                start_time = time.time()

                # augment data
                input_batch, noisy_batch, noise_std = \
                    augmentation_fn(input_batch)

                normalized_noisy_batch = \
                    model_normalize(noisy_batch, training=False)

                # --- add discriminator loss
                if use_discriminator:
                    with tf.GradientTape() as tape:
                        denoiser_step = global_step % 2 == 0
                        discriminator_step = not denoiser_step

                        # --- denoise and discriminate
                        normalized_input_batch = \
                            model_normalize(
                                input_batch,
                                training=False)

                        denoised_batch = \
                            model_denoise(
                                normalized_noisy_batch,
                                training=denoiser_step)

                        denormalized_denoised_batch = \
                            model_denormalize(
                                denoised_batch,
                                training=False)

                        discriminate_input_batch = \
                            keras.layers.Concatenate(axis=0)([
                                normalized_input_batch, denoised_batch])

                        discriminate_output_batch = \
                            model_discriminate(
                                discriminate_input_batch,
                                training=discriminator_step)

                        # --- create discriminate ground truth
                        input_shape = \
                            keras.backend.int_shape(input_batch)
                        ones_batch = \
                            tf.ones(
                                shape=(input_shape[0], input_shape[1], input_shape[2], 1),
                                dtype=tf.uint64)
                        zeros_batch = \
                            tf.zeros(
                                shape=(input_shape[0], input_shape[1], input_shape[2], 1),
                                dtype=tf.uint64)
                        discriminate_ground_truth = \
                            keras.layers.Concatenate(axis=0)([
                                ones_batch, zeros_batch])

                        # --- compute the loss value for this mini-batch
                        model_losses = None
                        if denoiser_step:
                            model_losses = model_denoise.losses

                        if discriminator_step:
                            model_losses = model_discriminate.losses

                        loss_map = loss_fn(
                            difficulty=noise_std,
                            input_batch=input_batch,
                            noisy_batch=noisy_batch,
                            model_losses=model_losses,
                            prediction_batch=denormalized_denoised_batch,
                            discriminate_batch=discriminate_output_batch,
                            discriminate_ground_truth=discriminate_ground_truth)

                        if denoiser_step:
                            grads = \
                                tape.gradient(
                                    target=loss_map[DISCRIMINATE_LOSS_STR] + loss_map[MEAN_TOTAL_LOSS_STR],
                                    sources=model_denoise.trainable_weights)

                            optimizer.apply_gradients(
                                grads_and_vars=zip(grads, model_denoise.trainable_weights))

                        if discriminator_step:
                            grads = \
                                tape.gradient(
                                    target=loss_map[DISCRIMINATE_LOSS_STR],
                                    sources=model_discriminate.trainable_weights)

                            optimizer.apply_gradients(
                                grads_and_vars=zip(grads, model_discriminate.trainable_weights))
                else:
                    # Open a GradientTape to record the operations run
                    # during the forward pass,
                    # which enables auto-differentiation.
                    with tf.GradientTape() as tape:
                        # Run the forward pass of the layer.
                        # The operations that the layer applies
                        # to its inputs are going to be recorded
                        # on the GradientTape.
                        denoised_batch = \
                            model_denoise(
                                normalized_noisy_batch,
                                training=True)
                        denormalized_denoised_batch = \
                            model_denormalize(
                                denoised_batch,
                                training=False)

                        # compute the loss value for this mini-batch
                        loss_map = loss_fn(
                            difficulty=noise_std,
                            input_batch=input_batch,
                            noisy_batch=noisy_batch,
                            model_losses=model_denoise.losses,
                            prediction_batch=denormalized_denoised_batch)

                        # Use the gradient tape to automatically retrieve
                        # the gradients of the trainable variables
                        # with respect to the loss.
                        grads = \
                            tape.gradient(
                                target=loss_map[MEAN_TOTAL_LOSS_STR],
                                sources=model_denoise.trainable_weights)

                        # Run one step of gradient descent by updating
                        # the value of the variables to minimize the loss.
                        optimizer.apply_gradients(
                            grads_and_vars=zip(grads, model_denoise.trainable_weights))

                # --- add loss summaries for tensorboard
                if loss_map is not None:
                    for name, key in [
                        ("loss/mae", "mae_loss"),
                        ("loss/nae", "nae_prediction"),
                        ("loss/total", MEAN_TOTAL_LOSS_STR),
                        ("quality/nae_noise", "nae_noise"),
                        ("quality/signal_to_noise_ratio", "snr"),
                        ("loss/regularization", REGULARIZATION_LOSS_STR),
                        ("quality/nae_improvement", "nae_improvement")
                    ]:
                        tf.summary.scalar(
                            name=name,
                            data=loss_map[key],
                            step=global_step)

                # --- add image prediction for tensorboard
                if global_step % visualization_every == 0:
                    random_batch = create_random_batch()
                    visualize(
                        global_step=global_step,
                        input_batch=input_batch,
                        noisy_batch=noisy_batch,
                        random_batch=random_batch,
                        prediction_batch=denormalized_denoised_batch,
                        visualization_number=visualization_number)
                    weights = \
                        get_conv2d_weights(
                            model=model_denoise,
                            verbose=False)
                    tf.summary.histogram(
                        data=weights,
                        step=global_step,
                        buckets=weight_buckets,
                        name="training/weights")

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
            logger.info("checkpoint at end of epoch: {0}".format(
                int(global_epoch)))
            global_epoch.assign_add(1)
            manager.save()

        # --- save checkpoint before exiting
        logger.info("checkpoint at step: {0}".format(
            int(global_step)))
        manager.save()
    logger.info("finished training")

# ---------------------------------------------------------------------
