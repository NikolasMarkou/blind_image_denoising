import os
import time
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Union, Dict

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .constants import *
from .visualize import visualize
from .custom_logger import logger
from .loss import \
    loss_function_builder, \
    MODEL_LOSS_FN_STR, \
    INPAINT_LOSS_FN_STR, \
    DENOISER_LOSS_FN_STR, \
    SUPERRES_LOSS_FN_STR
from .optimizer import optimizer_builder
from .pruning import prune_function_builder, get_conv2d_weights
from .model_hydra import model_builder as model_hydra_builder
from .utilities import load_config
from .file_operations import load_image
from .dataset import \
    dataset_builder, \
    DATASET_TRAINING_FN_STR, \
    NOISE_AUGMENTATION_FN_STR, \
    INPAINT_AUGMENTATION_FN_STR, \
    SUPERRES_AUGMENTATION_FN_STR, \
    GEOMETRIC_AUGMENTATION_FN_STR

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
    dataset_res = dataset_builder(config["dataset"])
    dataset_training = dataset_res[DATASET_TRAINING_FN_STR]
    noise_augmentation_fn = dataset_res[NOISE_AUGMENTATION_FN_STR]
    inpaint_augmentation_fn = dataset_res[INPAINT_AUGMENTATION_FN_STR]
    superres_augmentation_fn = dataset_res[SUPERRES_AUGMENTATION_FN_STR]
    geometric_augmentation_fn = dataset_res[GEOMETRIC_AUGMENTATION_FN_STR]

    # --- build loss function
    loss_fn_map = loss_function_builder(config=config["loss"])
    inpaint_loss_fn = tf.function(func=loss_fn_map[INPAINT_LOSS_FN_STR], reduce_retracing=True)
    denoiser_loss_fn = tf.function(func=loss_fn_map[DENOISER_LOSS_FN_STR], reduce_retracing=True)
    superres_loss_fn = tf.function(func=loss_fn_map[SUPERRES_LOSS_FN_STR], reduce_retracing=True)
    model_loss_fn = loss_fn_map[MODEL_LOSS_FN_STR]

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
    # how many steps to make a decomposition
    decomposition_every = \
        tf.constant(
            train_config.get("decomposition_every", 1000),
            dtype=tf.dtypes.int64,
            name="decomposition_every")
    # how many times the random batch will be iterated
    random_batch_iterations = \
        train_config.get("random_batch_iterations", 1)
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
                        image_size=(512, 512),
                        expand_dims=True,
                        normalize=False)
                test_images.append(image)
        test_images = \
            np.concatenate(
                test_images,
                axis=0)

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
        def optimized_model(x_input, x_mask):
            return models.hydra([x_input, x_mask], training=False)

        x = \
            tf.random.uniform(
                seed=0,
                minval=0.0,
                maxval=255.0,
                shape=random_batch_size)
        m = \
            tf.random.uniform(
                seed=0,
                minval=0.0,
                maxval=1.0,
                shape=tuple(random_batch_size[0:-1]) + (1,))
        _ = optimized_model(x, m)
        tf.summary.trace_export(
            step=0,
            name="hydra")
        tf.summary.flush()
        tf.summary.trace_off()

    # get each model
    hydra = models.hydra
    inpaint = models.hydra
    superres = models.superres
    backbone = models.backbone
    denoiser = models.denoiser
    denoiser_uq = models.denoiser_uq
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
                model_denoiser_uq=denoiser_uq,
                model_inpaint=inpaint,
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
            checkpoint\
                .restore(manager.latest_checkpoint)\
                .expect_partial()\
                .assert_existing_objects_matched()
            logger.info(f"restored checkpoint "
                        f"at epoch [{int(global_epoch)}] "
                        f"and step [{int(global_step)}]")
        else:
            logger.info("!!! Did NOT find checkpoint to restore !!!")

        # augmentation function
        geometric_augmentation_fn = \
            tf.function(
                func=geometric_augmentation_fn,
                reduce_retracing=True,
                input_signature=[
                    tf.TensorSpec(shape=[None, None, None, None],
                                  dtype=tf.uint8)])

        noise_augmentation_fn = \
            tf.function(
                func=noise_augmentation_fn,
                reduce_retracing=True,
                input_signature=[
                    tf.TensorSpec(shape=[None, None, None, None],
                                  dtype=tf.float32)])

        inpaint_augmentation_fn = \
            tf.function(
                func=inpaint_augmentation_fn,
                reduce_retracing=True,
                input_signature=[
                    tf.TensorSpec(shape=[None, None, None, None],
                                  dtype=tf.float32)])

        superres_augmentation_fn = \
            tf.function(
                func=superres_augmentation_fn,
                reduce_retracing=True,
                input_signature=[
                    tf.TensorSpec(shape=[None, None, None, None],
                                  dtype=tf.float32)])

        # downsample test image because it produces OOM
        test_images = superres_augmentation_fn(test_images)
        mask_test_images = test_images[:, :, :, 0] * 0.0 + 1.0

        # ---
        while global_epoch < global_total_epochs:
            logger.info("epoch: {0}, step: {1}".format(
                int(global_epoch), int(global_step)))

            # --- pruning strategy
            if use_prune and (global_epoch >= prune_start_epoch):
                logger.info(f"pruning weights at step [{int(global_step)}]")
                hydra = prune_fn(model=hydra)

            model_hydra_weights = hydra.trainable_weights
            start_time_epoch = time.time()

            # --- iterate over the batches of the dataset
            for input_batch in dataset_training:
                start_time_forward_backward = time.time()
                # geometric augmentation and casting to float
                input_batch = geometric_augmentation_fn(input_batch)
                input_batch = tf.cast(input_batch, dtype=tf.float32)
                # create batches for all subnetworks
                noisy_batch = noise_augmentation_fn(input_batch)
                downsampled_batch = superres_augmentation_fn(input_batch)
                masked_batch, mask_batch = inpaint_augmentation_fn(input_batch)

                # Open a GradientTape to record the operations run
                # during the forward pass,
                # which enables auto-differentiation.
                with tf.GradientTape() as tape:
                    # run the forward pass of the layer.
                    # The operations that the layer applies
                    # to its inputs are going to be recorded
                    # on the GradientTape.
                    denoiser_output, _, _, denoiser_uq_output = \
                        hydra([noisy_batch,
                               (noisy_batch[:, :, :, 0] * 0.0 + 1.0)],
                              training=True)

                    _, inpaint_output, _, _ = \
                        hydra([masked_batch, mask_batch],
                              training=True)

                    _, _, superres_output, _ = \
                        hydra([downsampled_batch,
                               (downsampled_batch[:, :, :, 0] * 0.0 + 1.0)],
                              training=True)

                    # compute the loss value for this mini-batch
                    denoiser_loss_map = \
                        denoiser_loss_fn(
                            input_batch=input_batch,
                            predicted_batch=denoiser_output)
                    denoiser_uq_loss_map = \
                        denoiser_loss_fn(
                            input_batch=input_batch - denoiser_uq_output,
                            predicted_batch=denoiser_uq_output)
                    inpaint_loss_map = \
                        inpaint_loss_fn(
                            input_batch=input_batch,
                            predicted_batch=inpaint_output)
                    superres_loss_map = \
                        superres_loss_fn(
                            input_batch=input_batch,
                            predicted_batch=superres_output)
                    model_loss_map = \
                        model_loss_fn(model=hydra)

                    total_loss = \
                        denoiser_loss_map[TOTAL_LOSS_STR] + \
                        denoiser_uq_loss_map[TOTAL_LOSS_STR] + \
                        inpaint_loss_map[TOTAL_LOSS_STR] + \
                        superres_loss_map[TOTAL_LOSS_STR] + \
                        model_loss_map[TOTAL_LOSS_STR]

                    grads = \
                        tape.gradient(
                            target=total_loss,
                            sources=model_hydra_weights)

                # --- apply weights
                optimizer.apply_gradients(
                    grads_and_vars=zip(grads, model_hydra_weights))

                # --- add loss summaries for tensorboard
                tf.summary.scalar(name="quality/denoiser_psnr", data=denoiser_loss_map[PSNR_STR], step=global_step)
                tf.summary.scalar(name="loss/denoiser_mae", data=denoiser_loss_map[MAE_LOSS_STR], step=global_step)
                tf.summary.scalar(name="loss/denoiser_total", data=denoiser_loss_map[TOTAL_LOSS_STR], step=global_step)

                tf.summary.scalar(name="quality/denoiser_uq_psnr", data=denoiser_uq_loss_map[PSNR_STR], step=global_step)
                tf.summary.scalar(name="loss/denoiser_uq_mae", data=denoiser_uq_loss_map[MAE_LOSS_STR], step=global_step)
                tf.summary.scalar(name="loss/denoiser_uq_total", data=denoiser_uq_loss_map[TOTAL_LOSS_STR], step=global_step)

                tf.summary.scalar(name="quality/inpaint_psnr", data=inpaint_loss_map[PSNR_STR], step=global_step)
                tf.summary.scalar(name="loss/inpaint_mae", data=inpaint_loss_map[MAE_LOSS_STR], step=global_step)
                tf.summary.scalar(name="loss/inpaint_total", data=inpaint_loss_map[TOTAL_LOSS_STR], step=global_step)

                tf.summary.scalar(name="quality/superres_psnr", data=superres_loss_map[PSNR_STR], step=global_step)
                tf.summary.scalar(name="loss/superres_mae", data=superres_loss_map[MAE_LOSS_STR], step=global_step)
                tf.summary.scalar(name="loss/superres_total", data=superres_loss_map[TOTAL_LOSS_STR], step=global_step)

                tf.summary.scalar(name="loss/regularization", data=model_loss_map[REGULARIZATION_LOSS_STR], step=global_step)
                tf.summary.scalar(name="loss/total", data=total_loss, step=global_step)

                # --- add image prediction for tensorboard
                if (global_step % visualization_every) == 0:
                    test_denoiser_output, _, test_superres_output, _ = \
                        hydra([test_images, mask_test_images], training=False)
                    test_denoiser_output, _, test_superres_output, _ = \
                        hydra([test_images, mask_test_images], training=False)
                    visualize(
                        global_step=global_step,
                        input_batch=input_batch,
                        noisy_batch=noisy_batch,
                        inpaint_batch=inpaint_output,
                        denoiser_batch=denoiser_output,
                        denoiser_uq_batch=denoiser_uq_output,
                        superres_batch=superres_output,
                        test_denoiser_batch=test_denoiser_output,
                        test_superres_batch=test_superres_output,
                        visualization_number=visualization_number)

                    # add weight visualization
                    tf.summary.histogram(
                        data=get_conv2d_weights(model=hydra),
                        step=global_step,
                        buckets=weight_buckets,
                        name="training/weights")

                # --- prune conv2d
                if use_prune and (global_epoch >= prune_start_epoch) and \
                        (int(prune_steps) != -1) and (global_step % prune_steps == 0):
                    logger.info(f"pruning weights at step [{int(global_step)}]")
                    hydra = prune_fn(model=hydra)
                    model_hydra_weights = hydra.trainable_weights

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

                tf.summary.scalar(
                    "training/steps_per_second",
                    1.0 / (step_time_forward_backward + 0.00001),
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
                            int(total_steps)))
                        break

            end_time_epoch = time.time()
            epoch_time = end_time_epoch - start_time_epoch

            # --- end of the epoch
            logger.info("checkpoint at end of epoch [{0}], took [{1}] minutes".format(
                int(global_epoch), int(round(epoch_time/60))))
            global_epoch.assign_add(1)
            manager.save()
            # save model so we can visualize it easier
            hydra.save(
                os.path.join(model_dir, MODEL_HYDRA_DEFAULT_NAME_STR))

    logger.info("finished training")

# ---------------------------------------------------------------------
