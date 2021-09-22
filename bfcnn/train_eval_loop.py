r"""Constructs model, inputs, and training environment."""

# ---------------------------------------------------------------------

__author__ = "Nikolas Markou"
__version__ = "0.1.0"
__license__ = "None"

# ---------------------------------------------------------------------

import os
import time
import json
import itertools
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend as K

# ---------------------------------------------------------------------


from .custom_logger import logger
from .optimizer import optimizer_builder


# ---------------------------------------------------------------------


def train_loop(
        pipeline_config_path,
        model_dir):
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
    # --- argument checking
    if not os.path.isfile(pipeline_config_path):
        return ValueError(
            "pipeline configuration path [{0}] is not valid".format(
                pipeline_config_path
            ))

    # --- load configuration
    with open(pipeline_config_path, "r") as f:
        config = json.load(f)

    # --- build the model, optimizer
    model = model_builder(config=config["model"])

    # save model so we can visualize it easier
    model.save(os.path.join(model_dir, "model.h5"))

    # --- build dataset
    dataset = dataset_builder(config=config["dataset"])

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
    min_depth = train_config["min_depth"]
    max_depth = train_config["max_depth"]
    # how many epochs to run mix datasets
    mixed_epochs = train_config["mixed_epochs"]
    warmup_epochs = train_config["warmup_epochs"]
    total_steps = train_config["total_steps"]
    # how many checkpoints to keep
    checkpoints_to_keep = train_config["checkpoints_to_keep"]
    # checkpoint every so many steps
    checkpoint_every = train_config["checkpoint_every"]
    # how many steps to make a visualization
    visualization_every = train_config["visualization_every"]
    # how many visualizations to show
    visualization_number = train_config["visualization_number"]
    # noise std
    noise_std = train_config.get("noise_std", 0.0)
    random_noise = noise_std > 0.0
    # dataset augmentation
    random_rotate = train_config.get("random_rotate", 0.0)
    random_up_down = train_config.get("random_up_down", False)
    random_left_right = train_config.get("random_left_right", False)

    # --- train the model
    with summary_writer.as_default():
        checkpoint = \
            tf.train.Checkpoint(
                model=model,
                step=global_step,
                epoch=global_epoch,
                optimizer=optimizer)
        manager = \
            tf.train.CheckpointManager(
                checkpoint=checkpoint,
                directory=model_dir,
                max_to_keep=checkpoints_to_keep)
        status = checkpoint.restore(manager.latest_checkpoint).expect_partial()
        trainable_weights = model.trainable_weights

        for epoch in range(int(global_epoch), int(epochs), 1):
            logger.info("epoch: {0}, step: {1}".format(epoch, int(global_step)))

            # --- change dataset according to epoch
            if epoch < warmup_epochs:
                d = dataset["megadepth"]
                logger.info("using: [megadepth] dataset")
            elif epoch < warmup_epochs + mixed_epochs:
                d = merge_iterators(dataset["kitti"], dataset["megadepth"])
                logger.info("using: [kitti, megadepth] dataset")
            else:
                d = dataset["kitti"]
                logger.info("using: [kitti] dataset")

            # --- iterate over the batches of the dataset
            for (images_batch_train, depth_batch_train, fg_bg_batch_train) in d:
                start_time = time.time()

                # --- threshold depth
                depth_batch_train = \
                    K.clip(
                        depth_batch_train,
                        min_value=min_depth,
                        max_value=max_depth)

                # --- data augmentation (jitter noise)
                if random_noise:
                    if np.random.choice([True, False]):
                        images_batch_train = \
                            images_batch_train + \
                            tf.random.normal(
                                shape=images_batch_train.shape,
                                mean=0,
                                stddev=noise_std)
                    if np.random.choice([True, False]):
                        images_batch_train = \
                            images_batch_train * \
                            tf.random.normal(
                                shape=images_batch_train.shape,
                                mean=1,
                                stddev=0.05)

                # --- flip left right (dataset augmentation)
                if random_left_right:
                    if np.random.choice([True, False]):
                        images_batch_train = \
                            tf.image.flip_left_right(images_batch_train)
                        depth_batch_train = \
                            tf.image.flip_left_right(depth_batch_train)
                        fg_bg_batch_train = \
                            tf.image.flip_left_right(fg_bg_batch_train)

                # --- flip up down (dataset augmentation)
                if random_up_down:
                    if np.random.choice([True, False]):
                        images_batch_train = \
                            tf.image.flip_up_down(images_batch_train)
                        depth_batch_train = \
                            tf.image.flip_up_down(depth_batch_train)
                        fg_bg_batch_train = \
                            tf.image.flip_up_down(fg_bg_batch_train)

                # --- randomly rotate input (dataset augmentation)
                if random_rotate > 0.0:
                    if np.random.choice([True, False]):
                        batch_size = \
                            K.int_shape(images_batch_train)[0]
                        angles = \
                            tf.random.uniform(
                                dtype=tf.float32,
                                minval=-random_rotate,
                                maxval=random_rotate,
                                shape=(batch_size,))
                        images_batch_train = \
                            tfa.image.rotate(
                                images=images_batch_train,
                                interpolation="bilinear",
                                angles=angles,
                                fill_value=0,
                                fill_mode="constant")
                        depth_batch_train = \
                            tfa.image.rotate(
                                images=depth_batch_train,
                                interpolation="nearest",
                                angles=angles,
                                fill_value=max_depth,
                                fill_mode="constant")
                        fg_bg_batch_train = \
                            tfa.image.rotate(
                                images=fg_bg_batch_train,
                                interpolation="nearest",
                                angles=angles,
                                fill_value=0,
                                fill_mode="constant")

                images_batch_train_preprocessed = \
                    keras.applications.mobilenet_v2.preprocess_input(
                        images_batch_train)

                # Open a GradientTape to record the operations run
                # during the forward pass, which enables auto-differentiation.
                with tf.GradientTape() as tape:
                    # Run the forward pass of the layer.
                    # The operations that the layer applies
                    # to its inputs are going to be recorded
                    # on the GradientTape.
                    depth_predictions = \
                        model(images_batch_train_preprocessed, training=True)

                    # compute the loss value for this mini-batch
                    loss_map = loss_fn(
                        model_losses=model.losses,
                        depth_gt=depth_batch_train,
                        fg_bg_gt=fg_bg_batch_train,
                        depth_predictions=depth_predictions)

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
                    for name, key in [
                        ("loss/total", "mean_total_loss"),
                        ("loss/delta", "mean_delta_loss"),
                        ("loss/regularization", "regularization_loss"),
                        ("loss/relative_depth", "mean_depth_loss_top"),
                        ("loss/absolute_depth", "mean_absolute_depth_loss")]:
                        tf.summary.scalar(
                            name,
                            loss_map[key], step=global_step)

                    # --- add image prediction for tensorboard
                    if global_step % visualization_every == 0:
                        depth_visualize(global_step,
                                        images_batch_train,
                                        depth_batch_train,
                                        fg_bg_batch_train,
                                        depth_predictions[0],
                                        visualization_number,
                                        min_depth=min_depth,
                                        max_depth=max_depth)

                # --- check if it is time to save a checkpoint
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
            global_epoch.assign_add(1)

        # --- save checkpoint before exiting
        logger.info("checkpoint at step: {0}".format(
            int(global_step)))
        manager.save()
    logger.info("finished training")

# ---------------------------------------------------------------------
