import os
import time
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Union, Dict, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .constants import *
from .custom_logger import logger
from .loss import \
    loss_function_builder, \
    MODEL_LOSS_FN_STR, \
    DENOISER_LOSS_FN_STR, \
    SUPERRES_LOSS_FN_STR, \
    DENOISER_UQ_LOSS_FN_STR
from .optimizer import optimizer_builder
from .pruning import prune_function_builder, get_conv2d_weights
from .model_hydra import model_builder as model_hydra_builder
from .utilities import load_config
from .file_operations import load_image
from .dataset import *

# ---------------------------------------------------------------------

CURRENT_DIRECTORY = os.path.realpath(os.path.dirname(__file__))


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
    dataset_training = dataset_builder(config["dataset"])

    # --- build loss function
    loss_fn_map = loss_function_builder(config=config["loss"])
    model_loss_fn = tf.function(func=loss_fn_map[MODEL_LOSS_FN_STR], reduce_retracing=True)
    denoiser_loss_fn = tf.function(func=loss_fn_map[DENOISER_LOSS_FN_STR], reduce_retracing=True)
    superres_loss_fn = tf.function(func=loss_fn_map[SUPERRES_LOSS_FN_STR], reduce_retracing=True)
    denoiser_uq_loss_fn = tf.function(func=loss_fn_map[DENOISER_UQ_LOSS_FN_STR], reduce_retracing=True)

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
    no_iterations_per_batch = train_config.get("no_iterations_per_batch", 1)
    global_total_epochs = tf.Variable(
        epochs, trainable=False, dtype=tf.dtypes.int64, name="global_total_epochs")
    weight_buckets = train_config.get("weight_buckets", 100)
    total_steps = \
        tf.constant(
            train_config.get("total_steps", -1),
            dtype=tf.dtypes.int64,
            name="total_steps")
    # how many checkpoints to keep
    checkpoints_to_keep = train_config.get("checkpoints_to_keep", 3)
    # checkpoint every so many steps
    checkpoint_every = \
        tf.constant(
            train_config.get("checkpoint_every", -1),
            dtype=tf.dtypes.int64,
            name="checkpoint_every")
    # how many steps to make a visualization
    visualization_every = \
        tf.constant(
            train_config.get("visualization_every", 1000),
            dtype=tf.dtypes.int64,
            name="visualization_every")
    # how many visualizations to show
    visualization_number = train_config.get("visualization_number", 5)
    # size of the random batch
    random_batch_size = \
        [visualization_number] + \
        train_config.get("random_batch_size", [128, 128, 3])

    # test images
    use_test_images = train_config.get("use_test_images", False)
    test_images = []
    if use_test_images:
        images_directory = os.path.join(CURRENT_DIRECTORY, "images")
        for image_filename in os.listdir(images_directory):
            # create full image path
            image_path = os.path.join(images_directory, image_filename)
            # check if current path is a file
            if os.path.isfile(image_path):
                image = \
                    load_image(
                        path=image_path,
                        num_channels=3,
                        image_size=(256, 256),
                        expand_dims=True,
                        normalize=False,
                        interpolation=tf.image.ResizeMethod.BILINEAR)
                test_images.append(image)
        test_images = np.concatenate(test_images, axis=0)
        test_images = tf.constant(test_images)

    # --- prune strategy
    prune_config = \
        train_config.get("prune", {"strategies": []})
    prune_start_epoch = \
        tf.constant(
            prune_config.get("start_epoch", 0),
            dtype=tf.dtypes.int64,
            name="prune_start_epoch")
    prune_steps = \
        tf.constant(
            prune_config.get("steps", -1),
            dtype=tf.dtypes.int64,
            name="prune_steps")
    prune_strategies = prune_config.get("strategies", None)
    if prune_strategies is None:
        use_prune = False
    elif isinstance(prune_strategies, list):
        use_prune = len(prune_strategies) > 0
    elif isinstance(prune_strategies, dict):
        use_prune = True
    else:
        use_prune = False
    use_prune = tf.constant(use_prune)
    prune_fn = prune_function_builder(prune_strategies)

    # --- build the hydra model
    tf.summary.trace_on(graph=True, profiler=False)
    with summary_writer.as_default():
        models = model_hydra_builder(config=config[MODEL_STR])

        # The function to be traced.
        @tf.function
        def optimized_model(x_input):
            return models.hydra(x_input, training=False)

        x = \
            tf.random.uniform(
                seed=0,
                minval=0.0,
                maxval=255.0,
                dtype=tf.float32,
                shape=random_batch_size)
        _ = optimized_model(x)
        tf.summary.trace_export(
            step=0,
            name="hydra")
        tf.summary.flush()
        tf.summary.trace_off()

    # get each model
    hydra = models.hydra
    superres = models.superres
    backbone = models.backbone
    denoiser = models.denoiser
    normalizer = models.normalizer
    denormalizer = models.denormalizer

    # summary of model
    hydra.summary(print_fn=logger.info)
    # save model so we can visualize it easier
    hydra.save(
        filepath=os.path.join(model_dir, MODEL_HYDRA_DEFAULT_NAME_STR),
        include_optimizer=False)

    # --- train the model
    with summary_writer.as_default():
        # checkpoint managing
        checkpoint = \
            tf.train.Checkpoint(
                step=global_step,
                epoch=global_epoch,
                optimizer=optimizer,
                model_hydra=hydra,
                model_backbone=backbone,
                model_denoiser=denoiser,
                model_superres=superres,
                model_normalizer=normalizer,
                model_denormalizer=denormalizer)
        manager = \
            tf.train.CheckpointManager(
                checkpoint=checkpoint,
                directory=model_dir,
                max_to_keep=checkpoints_to_keep)
        latest_checkpoint = manager.restore_or_initialize()

        if latest_checkpoint:
            logger.info("!!! Found checkpoint to restore !!!")
            logger.info(f"latest checkpoint [{0}:{1}]".format(
                latest_checkpoint, manager.latest_checkpoint))
            checkpoint \
                .restore(manager.latest_checkpoint) \
                .expect_partial() \
                .assert_existing_objects_matched()
            logger.info(f"restored checkpoint "
                        f"at epoch [{int(global_epoch)}] "
                        f"and step [{int(global_step)}]")
        else:
            logger.info("!!! Did NOT find checkpoint to restore !!!")

        # ---
        while global_epoch < global_total_epochs:
            logger.info("epoch: {0}, step: {1}".format(
                int(global_epoch), int(global_step)))

            # --- pruning strategy
            if use_prune and (global_epoch >= prune_start_epoch):
                logger.info(f"pruning weights at step [{int(global_step)}]")
                hydra = prune_fn(model=hydra)

            @tf.function(reduce_retracing=True)
            def train_forward_step(
                    n: tf.Tensor,
                    d: tf.Tensor) -> \
                    Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
                de, de_uq, x0, x1 = hydra(n, training=True)
                x2, x3, s, s_uq = hydra(d, training=True)
                del x0, x1, x2, x3
                return de, de_uq, s, s_uq

            trainable_weights = hydra.trainable_weights
            start_time_epoch = time.time()

            # --- iterate over the batches of the dataset
            for (input_batch, noisy_batch, downsampled_batch) in dataset_training:
                start_time_forward_backward = time.time()

                # run the forward pass of the layer.
                # The operations that the layer applies
                # to its inputs are going to be recorded
                # on the GradientTape.
                with tf.GradientTape() as tape:
                    denoiser_output, denoiser_uq_output, superres_output, superres_uq_output = \
                        train_forward_step(noisy_batch, downsampled_batch)

                    # compute the loss value for this mini-batch
                    denoiser_loss = \
                        denoiser_loss_fn(input_batch=downsampled_batch, predicted_batch=denoiser_output)
                    denoiser_uq_loss = \
                        denoiser_uq_loss_fn(uncertainty_batch=denoiser_uq_output)
                    superres_loss = \
                        superres_loss_fn(input_batch=input_batch, predicted_batch=superres_output)
                    superres_uq_loss = \
                        denoiser_uq_loss_fn(uncertainty_batch=superres_uq_output)
                    model_loss = \
                        model_loss_fn(model=hydra)

                    # NOTE do not use uncertainty for loss,
                    # only for observation
                    total_loss = \
                        model_loss[TOTAL_LOSS_STR] + \
                        denoiser_loss[TOTAL_LOSS_STR] / 2.0 + \
                        superres_loss[TOTAL_LOSS_STR] / 2.0 + \
                        denoiser_uq_loss[TOTAL_LOSS_STR] + \
                        superres_uq_loss[TOTAL_LOSS_STR]

                    # --- apply weights
                    optimizer.apply_gradients(
                        grads_and_vars=zip(
                            tape.gradient(target=total_loss, sources=trainable_weights),
                            trainable_weights))

                # --- add loss summaries for tensorboard
                # denoiser
                tf.summary.scalar(name="quality/denoiser/psnr",
                                  data=denoiser_loss[PSNR_STR],
                                  step=global_step)
                tf.summary.scalar(name="loss/denoiser/mae",
                                  data=denoiser_loss[MAE_LOSS_STR],
                                  step=global_step)
                tf.summary.scalar(name="loss/denoiser/total",
                                  data=denoiser_loss[TOTAL_LOSS_STR],
                                  step=global_step)
                tf.summary.scalar(name="loss/denoiser/uncertainty",
                                  data=denoiser_uq_loss[TOTAL_LOSS_STR],
                                  step=global_step)
                tf.summary.scalar(name="quality/denoiser/uncertainty",
                                  data=denoiser_uq_loss[UNCERTAINTY_LOSS_STR],
                                  step=global_step)
                # superres
                tf.summary.scalar(name="loss/superres/uncertainty",
                                  data=superres_uq_loss[TOTAL_LOSS_STR],
                                  step=global_step)
                tf.summary.scalar(name="quality/superres/psnr",
                                  data=superres_loss[PSNR_STR],
                                  step=global_step)
                tf.summary.scalar(name="quality/superres/uncertainty",
                                  data=superres_uq_loss[UNCERTAINTY_LOSS_STR],
                                  step=global_step)
                tf.summary.scalar(name="loss/superres/mae",
                                  data=superres_loss[MAE_LOSS_STR],
                                  step=global_step)
                tf.summary.scalar(name="loss/superres/total",
                                  data=superres_loss[TOTAL_LOSS_STR],
                                  step=global_step)
                # model
                tf.summary.scalar(name="loss/regularization",
                                  data=model_loss[REGULARIZATION_LOSS_STR],
                                  step=global_step)
                tf.summary.scalar(name="loss/total",
                                  data=total_loss,
                                  step=global_step)

                # --- add image prediction for tensorboard
                if (global_step % visualization_every) == 0:
                    # original input
                    tf.summary.image(
                        name="input", data=input_batch / 255,
                        max_outputs=visualization_number, step=global_step)

                    # augmented
                    tf.summary.image(
                        name="augmented/denoiser", data=noisy_batch / 255,
                        max_outputs=visualization_number, step=global_step)
                    tf.summary.image(
                        name="augmented/superres", data=downsampled_batch / 255,
                        max_outputs=visualization_number, step=global_step)

                    # output
                    tf.summary.image(
                        name="output/denoiser", data=denoiser_output / 255,
                        max_outputs=visualization_number, step=global_step)
                    tf.summary.image(
                        name="output/superres", data=superres_output / 255,
                        max_outputs=visualization_number, step=global_step)

                    # uncertainty
                    tf.summary.image(
                        name="uncertainty/denoiser", data=denoiser_uq_output,
                        max_outputs=visualization_number, step=global_step)
                    tf.summary.image(
                        name="uncertainty/superres", data=superres_uq_output,
                        max_outputs=visualization_number, step=global_step)

                    if use_test_images:
                        test_denoiser_output, test_denoiser_uq_output, \
                        test_superres_output, test_superres_uq_output = \
                            hydra(test_images, training=False)
                        tf.summary.image(
                            name="test/denoiser", data=test_denoiser_output / 255,
                            max_outputs=visualization_number, step=global_step)
                        tf.summary.image(
                            name="test/superres", data=test_superres_output / 255,
                            max_outputs=visualization_number, step=global_step)
                        del test_denoiser_output, test_denoiser_uq_output, \
                            test_superres_output, test_superres_uq_output

                # --- free resources
                del input_batch, noisy_batch, downsampled_batch
                del denoiser_output, denoiser_uq_output, superres_output, superres_uq_output

                # --- prune conv2d
                if use_prune and (global_epoch >= prune_start_epoch) and \
                        (prune_steps > -1) and (global_step % prune_steps == 0):
                    logger.info(f"pruning weights at step [{int(global_step)}]")
                    hydra = prune_fn(model=hydra)
                    trainable_weights = hydra.trainable_weights

                # --- check if it is time to save a checkpoint
                if checkpoint_every > 0 and \
                        (global_step % checkpoint_every == 0):
                    logger.info("checkpoint at step: {0}".format(
                        int(global_step)))
                    manager.save()
                    # save model so we can visualize it easier
                    hydra.save(
                        os.path.join(model_dir, MODEL_HYDRA_DEFAULT_NAME_STR))

                # --- keep time of steps per second
                stop_time_forward_backward = time.time()
                step_time_forward_backward = \
                    stop_time_forward_backward - \
                    start_time_forward_backward

                tf.summary.scalar(name="training/epoch",
                                  data=int(global_epoch),
                                  step=global_step)
                tf.summary.scalar(name="training/learning_rate",
                                  data=lr_schedule(int(global_step)),
                                  step=global_step)
                tf.summary.scalar(name="training/steps_per_second",
                                  data=1.0 / (step_time_forward_backward + 0.00001),
                                  step=global_step)

                # ---
                hydra.reset_metrics()
                global_step.assign_add(1)

                # --- check if total steps reached
                if total_steps > 0:
                    if total_steps <= global_step:
                        logger.info("total_steps reached [{0}]".format(
                            int(total_steps)))
                        break

            end_time_epoch = time.time()
            epoch_time = end_time_epoch - start_time_epoch

            # --- end of the epoch
            logger.info("checkpoint at end of epoch [{0}], took [{1}] minutes".format(
                int(global_epoch), int(round(epoch_time / 60))))
            global_epoch.assign_add(1)
            manager.save()
            # save model so we can visualize it easier
            hydra.save(
                os.path.join(model_dir, MODEL_HYDRA_DEFAULT_NAME_STR))

    logger.info("finished training")

# ---------------------------------------------------------------------
