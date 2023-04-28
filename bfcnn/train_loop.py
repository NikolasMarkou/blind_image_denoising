import os
import time
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Union, Dict, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .dataset import *
from .constants import *
from .custom_logger import logger
from .file_operations import load_image
from .optimizer import optimizer_builder
from .pruning import prune_function_builder
from .loss import loss_function_builder, improvement
from .utilities import load_config, create_checkpoint
from .model import model_builder as model_hydra_builder

# ---------------------------------------------------------------------

CURRENT_DIRECTORY = os.path.realpath(os.path.dirname(__file__))


# ---------------------------------------------------------------------


def train_loop(
        pipeline_config_path: Union[str, Dict, Path],
        model_dir: Union[str, Path],
        weights_dir: Union[str, Path] = None):
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
    :param weights_dir: directory to load weights from
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
    dataset_config = config["dataset"]
    dataset_training = dataset_builder(dataset_config)
    batch_size = dataset_config["batch_size"]
    input_shape = dataset_config["input_shape"]

    # --- build loss function
    loss_fn_map = loss_function_builder(config=config["loss"])
    model_loss_fn = tf.function(func=loss_fn_map[MODEL_LOSS_FN_STR], reduce_retracing=True)
    denoiser_loss_fn = tf.function(func=loss_fn_map[DENOISER_LOSS_FN_STR], reduce_retracing=True)
    superres_loss_fn = tf.function(func=loss_fn_map[SUPERRES_LOSS_FN_STR], reduce_retracing=True)

    # --- build optimizer
    optimizer, _ = \
        optimizer_builder(config=config["train"]["optimizer"])

    # --- get the train configuration
    train_config = config["train"]
    epochs = train_config["epochs"]
    gpu_batches_per_step = train_config.get("gpu_batches_per_step", 1)
    if gpu_batches_per_step is None or gpu_batches_per_step <= 0:
        raise ValueError("gpu_batches_per_step must be > 0")

    global_total_epochs = tf.Variable(
        epochs, trainable=False, dtype=tf.dtypes.int64, name="global_total_epochs")
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
        train_config.get("random_batch_size", input_shape)

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
                        num_channels=input_shape[-1],
                        image_size=input_shape[:-1],
                        expand_dims=True,
                        normalize=False,
                        interpolation=tf.image.ResizeMethod.BILINEAR)
                test_images.append(image)
        test_images = np.concatenate(test_images, axis=0)
        test_images = tf.constant(test_images)
        test_images = tf.cast(test_images, dtype=tf.float32)

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

    # --- train the model
    with tf.summary.create_file_writer(model_dir).as_default():
        # --- checkpoint managing
        ckpt = \
            create_checkpoint(
                model=model_hydra_builder(
                    config=config[MODEL_STR]).hydra)

        # summary of model
        ckpt.model.reset_states()
        ckpt.model.reset_metrics()
        ckpt.model.summary(print_fn=logger.info)

        # save model so we can visualize it easier
        ckpt.model.save(
            filepath=os.path.join(model_dir, MODEL_HYDRA_DEFAULT_NAME_STR),
            include_optimizer=False)

        manager = \
            tf.train.CheckpointManager(
                checkpoint=ckpt,
                directory=model_dir,
                max_to_keep=checkpoints_to_keep)

        def save_checkpoint_model_fn():
            # save model and weights
            logger.info("saving checkpoint at step: [{0}]".format(
                int(ckpt.step)))
            save_path = manager.save()
            logger.info(f"saved checkpoint to [{save_path}]")

        if manager.restore_or_initialize():
            logger.info("!!! Found checkpoint to restore !!!")
            ckpt \
                .restore(manager.latest_checkpoint) \
                .expect_partial()
            logger.info(f"restored checkpoint "
                        f"at epoch [{int(ckpt.epoch)}] "
                        f"and step [{int(ckpt.step)}]")
            # restore learning rate
            optimizer.iterations.assign(ckpt.step)
        else:
            logger.info("!!! Did NOT find checkpoint to restore !!!")
            if weights_dir is not None and \
                    len(weights_dir) > 0 and \
                    os.path.isdir(weights_dir):
                # restore weights from a directory
                loaded_weights = False

                if not loaded_weights:
                    try:
                        logger.info(f"loading weights from [{weights_dir}]")
                        tmp_model = tf.keras.models.clone_model(ckpt.model)
                        # restore checkpoint
                        tmp_checkpoint = \
                            create_checkpoint(
                                model=tf.keras.models.clone_model(ckpt.model),
                                path=weights_dir)
                        tmp_model.set_weights(tmp_checkpoint.model.get_weights())
                        ckpt.model = tmp_model
                        ckpt.step.assign(0)
                        ckpt.epoch.assign(0)
                        del tmp_model
                        del tmp_checkpoint
                        loaded_weights = True
                        logger.info("successfully loaded weights")
                    except Exception as e:
                        logger.info(
                            f"!!! failed to load weights from [{weights_dir}]] !!!")
                        logger.error(f"!!! {e}")

                if not loaded_weights:
                    logger.info("!!! failed to load weights")

            save_checkpoint_model_fn()

        @tf.function(
            reduce_retracing=True)
        def train_forward_step(
                n: tf.Tensor,
                d: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
            de, _ = ckpt.model(n, training=True)
            _, sr = ckpt.model(d, training=True)
            return de, sr

        @tf.function(
            reduce_retracing=True)
        def test_step(
                n: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
            return ckpt.model(n, training=False)

        if ckpt.step == 0:
            tf.summary.trace_on(graph=True, profiler=False)

            # run a single step
            _ = test_step(
                n=iter(dataset_training).get_next()[0]
            )

            tf.summary.trace_export(
                step=ckpt.step,
                name="model_hydra")
            tf.summary.flush()
            tf.summary.trace_off()

        # ---
        finished_training = False

        while not finished_training and \
                (global_total_epochs == -1 or ckpt.epoch < global_total_epochs):
            logger.info("epoch [{0}], step [{1}]".format(
                int(ckpt.epoch), int(ckpt.step)))

            # --- pruning strategy
            if use_prune and (ckpt.epoch >= prune_start_epoch):
                logger.info(f"pruning weights at step [{int(ckpt.step)}]")
                hydra = prune_fn(model=hydra)

            start_time_epoch = time.time()
            trainable_variables = hydra.trainable_variables

            # --- iterate over the batches of the dataset
            dataset_iterator = iter(dataset_training)
            gpu_batches = 0
            step_time_dataset = 0.0
            total_loss = tf.constant(0.0, dtype=tf.float32)

            # --- check if total steps reached
            if total_steps != -1:
                if total_steps <= ckpt.step:
                    logger.info("total_steps reached [{0}]".format(
                        int(total_steps)))
                    finished_training = True

            # --- iterate over the batches of the dataset
            while not finished_training:
                try:
                    start_time_dataset = time.time()
                    (input_batch, noisy_batch, downsampled_batch) = \
                        dataset_iterator.get_next()
                    stop_time_dataset = time.time()
                    step_time_dataset = stop_time_dataset - start_time_dataset
                except tf.errors.OutOfRangeError:
                    break

                start_time_forward_backward = time.time()

                with tf.GradientTape(watch_accessed_variables=False) as tape:
                    tape.watch(trainable_variables)
                    de, sr = \
                        train_forward_step(
                            n=noisy_batch,
                            d=downsampled_batch)

                    # compute the loss value for this mini-batch
                    de_loss = denoiser_loss_fn(gt_batch=input_batch, predicted_batch=de)
                    sr_loss = superres_loss_fn(gt_batch=input_batch, predicted_batch=sr)

                    # combine losses
                    total_loss += \
                        (de_loss[TOTAL_LOSS_STR] +
                         sr_loss[TOTAL_LOSS_STR]) / 2

                    gpu_batches += 1

                    if gpu_batches < gpu_batches_per_step:
                        continue
                    else:
                        gpu_batches = 0
                        model_loss = model_loss_fn(model=hydra)
                        total_loss += \
                            total_loss / gpu_batches_per_step + \
                            model_loss[TOTAL_LOSS_STR]
                        # apply weights
                        optimizer.apply_gradients(
                            grads_and_vars=zip(
                                tape.gradient(target=total_loss, sources=trainable_variables),
                                trainable_variables))

                # --- add loss summaries for tensorboard
                for summary in [(DENOISER_STR, de_loss),
                                (SUPERRES_STR, sr_loss)]:
                    title = summary[0]
                    loss_map = summary[1]
                    tf.summary.scalar(name=f"quality/{title}/psnr",
                                      data=loss_map[PSNR_STR],
                                      step=ckpt.step)
                    tf.summary.scalar(name=f"loss/{title}/mae",
                                      data=loss_map[MAE_LOSS_STR],
                                      step=ckpt.step)
                    tf.summary.scalar(name=f"loss/{title}/ssim",
                                      data=loss_map[SSIM_LOSS_STR],
                                      step=ckpt.step)
                    tf.summary.scalar(name=f"loss/{title}/total",
                                      data=loss_map[TOTAL_LOSS_STR],
                                      step=ckpt.step)

                # denoiser specific
                tf.summary.scalar(name=f"quality/denoiser/improvement",
                                  data=improvement(original=input_batch,
                                                   noisy=noisy_batch,
                                                   denoised=de),
                                  step=ckpt.step)

                # model
                tf.summary.scalar(name="loss/regularization",
                                  data=model_loss[REGULARIZATION_LOSS_STR],
                                  step=ckpt.step)
                tf.summary.scalar(name="loss/total",
                                  data=total_loss,
                                  step=ckpt.step)

                # --- add image prediction for tensorboard
                if (ckpt.step % visualization_every) == 0:
                    # original input
                    tf.summary.image(
                        name="input", data=input_batch / 255,
                        max_outputs=visualization_number, step=ckpt.step)

                    # augmented
                    tf.summary.image(
                        name="input_augmented/denoiser", data=noisy_batch / 255,
                        max_outputs=visualization_number, step=ckpt.step)
                    tf.summary.image(
                        name="input_augmented/superres", data=downsampled_batch / 255,
                        max_outputs=visualization_number, step=ckpt.step)

                    # output
                    tf.summary.image(
                        name="output/denoiser", data=de / 255,
                        max_outputs=visualization_number, step=ckpt.step)
                    tf.summary.image(
                        name="output/superres", data=sr / 255,
                        max_outputs=visualization_number, step=ckpt.step)

                    if use_test_images:
                        test_de, test_sr = test_step(test_images)
                        tf.summary.image(
                            name="test/denoiser", data=test_de / 255,
                            max_outputs=visualization_number, step=ckpt.step)
                        tf.summary.image(
                            name="test/superres", data=test_sr / 255,
                            max_outputs=visualization_number, step=ckpt.step)

                # --- prune conv2d
                if use_prune and (ckpt.epoch >= prune_start_epoch) and \
                        (prune_steps > -1) and (ckpt.step % prune_steps == 0):
                    logger.info(f"pruning weights at step [{int(ckpt.step)}]")
                    hydra = prune_fn(model=hydra)
                    trainable_variables = hydra.trainable_variables

                # --- check if it is time to save a checkpoint
                if checkpoint_every > 0 and \
                        (ckpt.step % checkpoint_every == 0):
                    save_checkpoint_model_fn()

                # --- keep time of steps per second
                stop_time_forward_backward = time.time()
                step_time_forward_backward = \
                    stop_time_forward_backward - \
                    start_time_forward_backward
                step_time_training = \
                    stop_time_forward_backward - \
                    start_time_dataset

                tf.summary.scalar(name="training/epoch",
                                  data=int(ckpt.epoch),
                                  step=ckpt.step)
                tf.summary.scalar(name="training/learning_rate",
                                  data=optimizer.learning_rate,
                                  step=ckpt.step)
                tf.summary.scalar(name="training/training_step_time",
                                  data=step_time_training,
                                  step=ckpt.step)
                tf.summary.scalar(name="training/inference_step_time",
                                  data=step_time_forward_backward,
                                  step=ckpt.step)
                tf.summary.scalar(name="training/dataset_step_time",
                                  data=step_time_dataset,
                                  step=ckpt.step)

                # ---
                ckpt.step.assign_add(1)
                total_loss = tf.constant(0.0, dtype=tf.float32)

                # --- check if total steps reached
                if 0 < total_steps <= ckpt.step:
                    logger.info("total_steps reached [{0}]".format(
                        int(total_steps)))
                    finished_training = True
                    break

            end_time_epoch = time.time()
            epoch_time = end_time_epoch - start_time_epoch

            # --- end of the epoch
            logger.info("end of epoch [{0}], took [{1}] seconds".format(
                int(ckpt.epoch), int(round(epoch_time))))
            ckpt.epoch.assign_add(1)
            save_checkpoint_model_fn()

    logger.info("finished training")

# ---------------------------------------------------------------------
