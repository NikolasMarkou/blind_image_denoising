import copy
import os
import time
import json
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Union, Dict, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .dataset import *
from .constants import *
from .model import model_builder
from .custom_logger import logger
from .file_operations import load_image
from .optimizer import (
    optimizer_builder,
    deep_supervision_schedule_builder)
from .loss import loss_function_builder
from .utilities import \
    load_config, \
    create_checkpoint, \
    save_config, \
    multiscales_generator_fn, \
    find_layer_by_name
from .visualize import \
    visualize_weights_boxplot, \
    visualize_weights_heatmap, \
    visualize_gradient_boxplot
from .images import images as evaluation_image_paths

# ---------------------------------------------------------------------

CURRENT_DIRECTORY = os.path.realpath(os.path.dirname(__file__))

# ---------------------------------------------------------------------


def train_loop(
        pipeline_config_path: Union[str, Dict, Path],
        checkpoint_directory: Union[str, Path],
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
    :param checkpoint_directory: directory to save checkpoints into
    :param weights_dir: directory to load weights from
    :return:
    """
    # --- load configuration
    config = load_config(pipeline_config_path)
    tf.random.set_seed(0)

    # --- create model_dir if not exist
    if not os.path.isdir(str(checkpoint_directory)):
        # if path does not exist attempt to make it
        Path(str(checkpoint_directory)).mkdir(parents=True, exist_ok=True)
        # if it fails again throw exception
        if not os.path.isdir(str(checkpoint_directory)):
            raise ValueError("Model directory [{0}] is not valid".format(
                checkpoint_directory))

    # --- save configuration into path, makes it easier to compare afterwards
    save_config(
        config=config,
        filename=os.path.join(str(checkpoint_directory), CONFIG_PATH_STR))

    # --- build dataset
    dataset = dataset_builder(config[DATASET_STR])

    batch_size = dataset.batch_size
    input_shape = dataset.input_shape
    dataset_training = dataset.training
    no_color_channels = config[DATASET_STR][INPUT_SHAPE_STR][-1]
    evaluation_batch = (
        tf.concat([
            load_image(path=img,
                       image_size=(512,512),
                       num_channels=3,
                       expand_dims=True,
                       normalize=False)
            for img in evaluation_image_paths
        ], axis=0))
    evaluation_batch = tf.cast(evaluation_batch, dtype=tf.float32)
    # --- build loss function
    loss_fn_map = loss_function_builder(config=config["loss"])
    model_loss_fn = \
        tf.function(
            func=loss_fn_map[MODEL_LOSS_FN_STR],
            reduce_retracing=True)

    # --- build optimizer
    optimizer, lr_schedule = \
        optimizer_builder(config=config["train"]["optimizer"])

    # --- get the train configuration
    train_config = config["train"]

    #
    epochs = train_config["epochs"]
    gpu_batches_per_step = int(train_config.get("gpu_batches_per_step", 1))
    if gpu_batches_per_step <= 0:
        raise ValueError("gpu_batches_per_step must be > 0")

    # how many checkpoints to keep
    checkpoints_to_keep = \
        train_config.get("checkpoints_to_keep", 3)
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
    # layers for visualization
    activity_layers = \
        train_config.get("activity_layers", [])

    # --- train the model
    with tf.summary.create_file_writer(checkpoint_directory).as_default():
        # --- write configuration in tensorboard
        tf.summary.text("config", json.dumps(config, indent=4), step=0)

        # --- create the help variables
        total_epochs = tf.constant(
            epochs, dtype=tf.dtypes.int64, name="total_epochs")
        total_steps = tf.constant(
            train_config.get("total_steps", -1), dtype=tf.dtypes.int64, name="total_steps")

        # --- build the hydra model
        config[MODEL_STR][BATCH_SIZE_STR] = batch_size
        models = model_builder(config=config[MODEL_STR])
        ckpt = \
            create_checkpoint(
                model=models.hydra,
                path=None)
        # summary of model and save model, so we can inspect with netron
        ckpt.model.summary(print_fn=logger.info)
        ckpt.model.save(
            os.path.join(checkpoint_directory, MODEL_HYDRA_DEFAULT_NAME_STR))

        manager = \
            tf.train.CheckpointManager(
                checkpoint_name="ckpt",
                checkpoint=ckpt,
                directory=checkpoint_directory,
                max_to_keep=checkpoints_to_keep)

        def save_checkpoint_model_fn():
            # save model and weights
            logger.info("saving checkpoint at step: [{0}]".format(
                int(ckpt.step)))
            save_path = manager.save()
            logger.info(f"saved checkpoint to [{save_path}]")

        if manager.latest_checkpoint:
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
                logger.info(f"!!! attempting to load weights [{weights_dir}]")
                loaded_weights = False

                for d in [weights_dir]:
                    if not loaded_weights:
                        try:
                            logger.info(f"loading weights from [{d}]")
                            tmp_model = tf.keras.models.clone_model(ckpt.model)
                            # restore checkpoint
                            tmp_checkpoint = create_checkpoint(model=tf.keras.models.clone_model(ckpt.model), path=d)
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
                                f"!!! failed to load weights from [{d}]] !!!")
                            logger.error(f"!!! {e}")

                if not loaded_weights:
                    logger.info("!!! failed to load weights")
            else:
                logger.info(f"!!! no weights to load [{weights_dir}]")

            save_checkpoint_model_fn()

        # find indices of denoiser, materials segmentation, bg lumen wall segmentation
        # first third is denoiser, second third is materials, last third is bg_lumen_wall
        model_no_outputs = len(ckpt.model.outputs)
        logger.info(f"model number of outputs: [{model_no_outputs}]")

        denoiser_index = [
            i for i in range(0, int(model_no_outputs))
        ]
        sizes = [
            (int(input_shape[0] / (2 ** i)), int(input_shape[1] / (2 ** i)))
            for i in range(len(denoiser_index))
        ]
        logger.info(f"model denoiser_index: {denoiser_index}")
        denoiser_loss_fn_list = [
            tf.function(
                func=loss_fn_map[DENOISER_LOSS_FN_STR],
                input_signature=[
                    tf.TensorSpec(shape=[batch_size, sizes[i][0], sizes[i][1], no_color_channels], dtype=tf.float32),
                    tf.TensorSpec(shape=[batch_size, sizes[i][0], sizes[i][1], no_color_channels], dtype=tf.float32),
                ],
                reduce_retracing=True)
            for i in range(len(denoiser_index))
        ]

        multiscales_fn = \
            multiscales_generator_fn(
                shape=[batch_size, None, None, no_color_channels],
                no_scales=len(denoiser_index),
                clip_values=True,
                round_values=True,
                jit_compile=False,
                normalize_values=False,
            )

        @tf.function(reduce_retracing=True, jit_compile=False)
        def train_step(n: List[tf.Tensor]):
            return ckpt.model(n, training=True)

        @tf.function(reduce_retracing=True, jit_compile=False)
        def test_step(n: List[tf.Tensor]):
            if model_no_outputs == 1:
                return ckpt.model(n, training=False)
            return ckpt.model(n, training=False)[denoiser_index[0]]

        @tf.function(
            autograph=True,
            jit_compile=False,
            reduce_retracing=True)
        def train_step_single_gpu(
                p_input_image_batch: tf.Tensor,
                p_noisy_image_batch: tf.Tensor,
                p_depth_weight: tf.Tensor,
                p_percentage_done: tf.Tensor,
                p_trainable_variables: List):

            p_all_denoiser_loss = [None] * len(denoiser_index)
            p_total_denoiser_loss = tf.constant(0.0, dtype=tf.float32)

            p_scale_gt_image_batch = \
                multiscales_fn(p_input_image_batch)

            with tf.GradientTape() as tape:
                p_predictions = train_step(n=[p_noisy_image_batch])

                # get denoise loss for each depth,
                if model_no_outputs == 1:
                    p_loss = denoiser_loss_fn_list[0](
                        gt_batch=p_input_image_batch,
                        predicted_batch=p_predictions)
                    p_total_denoiser_loss += \
                        p_loss[TOTAL_LOSS_STR] * p_depth_weight[0]
                    p_all_denoiser_loss[0] = p_loss
                else:
                    for i in range(len(denoiser_index)):
                        p_loss = denoiser_loss_fn_list[i](
                            gt_batch=p_scale_gt_image_batch[i],
                            predicted_batch=p_predictions[i])
                        p_total_denoiser_loss += \
                            p_loss[TOTAL_LOSS_STR] * p_depth_weight[i]
                        p_all_denoiser_loss[i] = p_loss

                # combine losses
                p_model_loss = \
                    model_loss_fn(model=ckpt.model)
                p_total_loss = \
                    p_total_denoiser_loss + \
                    p_model_loss[TOTAL_LOSS_STR]
                p_grads = \
                    tape.gradient(target=p_total_loss,
                                  sources=p_trainable_variables)

            return (
                p_total_loss,
                p_model_loss,
                p_all_denoiser_loss,
                p_predictions,
                p_grads
            )

        @tf.function(
            reduce_retracing=True)
        def apply_grads(
                internal_optimizer,
                internal_gradients,
                internal_trainable_variables):
            internal_optimizer.apply_gradients(
                zip(internal_gradients, internal_trainable_variables))

        if ckpt.step == 0:
            tf.summary.trace_on(graph=True, profiler=False)

            # run a single step
            _, tmp_noisy = iter(dataset_training).get_next()
            results = train_step([tmp_noisy])

            if isinstance(results, list):
                for i, tensor in enumerate(results):
                    logger.info(f"train_step: output[{i}], => {tf.keras.backend.int_shape(tensor)}")

            tf.summary.trace_export(
                step=ckpt.step,
                name="model_hydra")
            tf.summary.flush()
            tf.summary.trace_off()

        # ---
        finished_training = False
        trainable_variables = ckpt.model.trainable_variables
        counter = tf.Variable(0, dtype=tf.uint32, trainable=False)
        gradients_accumulation = [
            tf.Variable(tf.zeros_like(v))
            for v in trainable_variables
        ]
        gradients_constant = tf.constant(1.0 / float(gpu_batches_per_step), dtype=tf.float32)

        deep_supervision_schedule = \
            deep_supervision_schedule_builder(
                config=train_config.get("deep_supervision", {
                    TYPE_STR: "linear_low_to_high"
                }),
                no_outputs=len(ckpt.model.outputs))

        while not finished_training and \
                (total_epochs == -1 or ckpt.epoch < total_epochs):

            logger.info("epoch [{0}], step [{1}]".format(
                int(ckpt.epoch), int(ckpt.step)))

            start_time_epoch = time.time()

            # --- training percentage
            if total_epochs > 0:
                percentage_done = float(ckpt.epoch) / float(total_epochs)
            elif total_steps > 0:
                percentage_done = float(ckpt.step) / float(total_steps)
            else:
                percentage_done = 0.0

            logger.info("percentage done [{:.2f}]".format(float(percentage_done)))
            depth_weight = \
                deep_supervision_schedule(percentage_done=percentage_done)
            depth_weight_str = [
                "{0:.2f}".format(d)
                for d in depth_weight
            ]
            depth_weight = tf.constant(depth_weight, dtype=tf.float32)
            logger.info(f"weight per output index: {depth_weight_str}")

            # --- initialize iterators
            dataset_train = iter(dataset_training)

            # --- check if total steps reached
            if total_steps > 0:
                if total_steps <= ckpt.step:
                    logger.info("total_steps reached [{0}]".format(
                        int(total_steps)))
                    finished_training = True

            # --- training percentage
            if total_epochs > 0:
                percentage_done = float(ckpt.epoch) / float(total_epochs)
            elif total_steps > 0:
                percentage_done = float(ckpt.step) / float(total_steps)
            else:
                percentage_done = 0.0
            percentage_done = tf.constant(percentage_done, dtype=tf.float32)

            # --- iterate over the batches of the dataset
            for input_image_batch, noisy_image_batch in dataset_train:
                if counter == 0:
                    start_time_forward_backward = time.time()
                    # zero out gradients
                    for v in gradients_accumulation:
                        v.assign(v * 0.0)

                total_loss, model_loss, all_denoiser_loss, predictions, grads = \
                    train_step_single_gpu(
                        p_input_image_batch=input_image_batch,
                        p_noisy_image_batch=noisy_image_batch,
                        p_depth_weight=depth_weight,
                        p_percentage_done=percentage_done,
                        p_trainable_variables=trainable_variables)

                for i, grad in enumerate(grads):
                    gradients_accumulation[i].assign_add(grad)

                if counter >= gpu_batches_per_step:
                    counter.assign(value=0)
                    # average gradients
                    for v in gradients_accumulation:
                        v.assign(v * gradients_constant)

                    # !!! IMPORTANT !!!!
                    # apply gradient to change weights
                    # this is a hack to stop retracing the update function
                    # https://stackoverflow.com/questions/77028664/tf-keras-optimizers-adam-apply-gradients-triggers-tf-function-retracing
                    apply_grads(
                        internal_optimizer=optimizer,
                        internal_gradients=gradients_accumulation,
                        internal_trainable_variables=trainable_variables)
                else:
                    counter.assign_add(delta=1)
                    continue

                # --- add loss summaries for tensorboard
                for i, d in enumerate(all_denoiser_loss):
                    tf.summary.scalar(name=f"loss_denoiser/scale_{i}/mae",
                                      data=d[MAE_LOSS_STR],
                                      step=ckpt.step)
                    tf.summary.scalar(name=f"loss_denoiser/scale_{i}/mse",
                                      data=d[MSE_LOSS_STR],
                                      step=ckpt.step)
                    tf.summary.scalar(name=f"loss_denoiser/scale_{i}/ssim",
                                      data=d[SSIM_LOSS_STR],
                                      step=ckpt.step)
                    tf.summary.scalar(name=f"loss_denoiser/scale_{i}/total",
                                      data=d[TOTAL_LOSS_STR],
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
                    # --- denoiser
                    tf.summary.image(name="denoiser/input", data=input_image_batch / 255,
                                     max_outputs=visualization_number, step=ckpt.step)
                    # noisy batch
                    tf.summary.image(name="denoiser/noisy", data=noisy_image_batch / 255,
                                     max_outputs=visualization_number, step=ckpt.step)
                    # denoised batch
                    prediction = predictions
                    if isinstance(predictions, list):
                        for i, d in enumerate(predictions):
                            tf.summary.image(name=f"denoiser/scale_{i}/output", data=d / 255,
                                             max_outputs=visualization_number, step=ckpt.step)
                        prediction = predictions[0]
                    else:
                        tf.summary.image(name=f"denoiser/scale_0/output", data=predictions / 255,
                                         max_outputs=visualization_number, step=ckpt.step)

                    # error per pixel
                    tf.summary.image(name=f"error/mae",
                                     data=tf.clip_by_value(
                                         tf.abs(prediction - input_image_batch),
                                         clip_value_min=0.0,
                                         clip_value_max=255.0) / 255,
                                     max_outputs=visualization_number,
                                     step=ckpt.step)

                    tf.summary.histogram(name="error/mae_distribution",
                                         data=tf.clip_by_value(
                                                tf.abs(prediction - input_image_batch),
                                                clip_value_min=0.0,
                                                clip_value_max=255.0),
                                         step=ckpt.step,
                                         buckets=64)

                    tf.summary.histogram(name="training/noise_distribution",
                                         data=tf.clip_by_value(
                                                tf.abs(noisy_image_batch - input_image_batch),
                                                clip_value_min=0.0,
                                                clip_value_max=255.0),
                                         step=ckpt.step,
                                         buckets=64)

                    # --- evaluation
                    for i in range(5):
                        std_noise = float(i) * 20.0
                        evaluation_batch_noise =\
                            evaluation_batch + \
                            tf.random.normal(shape=tf.shape(evaluation_batch),
                                             mean=0.0,
                                             stddev=std_noise,
                                             dtype=tf.float32)
                        evaluation_batch_noise = \
                            tf.clip_by_value(evaluation_batch_noise,
                                             clip_value_min=0.0,
                                             clip_value_max=255.0)
                        evaluation_result = test_step(evaluation_batch_noise)
                        tf.summary.image(name=f"evaluation_{std_noise}/input",
                                         data=evaluation_batch_noise / 255,
                                         max_outputs=visualization_number,
                                         step=ckpt.step,
                                         description="evaluation noisy")
                        tf.summary.image(name=f"evaluation_{std_noise}/output",
                                         data=evaluation_result / 255,
                                         max_outputs=visualization_number,
                                         step=ckpt.step,
                                         description="evaluation denoised")


                    # --- add gradient activity
                    gradient_activity = \
                        visualize_gradient_boxplot(
                            gradients=gradients_accumulation,
                            trainable_variables=trainable_variables) / 255
                    tf.summary.image(name="weights/gradients",
                                     data=gradient_activity,
                                     max_outputs=visualization_number,
                                     step=ckpt.step,
                                     description="gradient activity")

                    # --- add weights distribution
                    weights_boxplot = \
                        visualize_weights_boxplot(
                            trainable_variables=trainable_variables) / 255
                    tf.summary.image(name="weights/boxplot",
                                     data=weights_boxplot,
                                     max_outputs=visualization_number,
                                     step=ckpt.step,
                                     description="weights boxplot")
                    weights_heatmap = \
                        visualize_weights_heatmap(
                            trainable_variables=trainable_variables) / 255
                    tf.summary.image(name="weights/heatmap",
                                     data=weights_heatmap,
                                     max_outputs=visualization_number,
                                     step=ckpt.step,
                                     description="weights heatmap")

                    # --- add activity distribution
                    for activity_layer in activity_layers:
                        logger.info(f"{activity_layer}")
                        layer = find_layer_by_name(model=ckpt.model.get_layer, layer_name=activity_layer)
                        if layer is None:
                            logger.info(f"Failed to find {activity_layer}")
                        else:
                            keras_fn = keras.backend.function([ckpt.model.input], [layer.output])
                            tf.summary.histogram(name=f"activity/{layer.name}",
                                                 data=keras_fn(evaluation_batch),
                                                 step=ckpt.step,
                                                 buckets=64)
                            del keras_fn

                # --- check if it is time to save a checkpoint
                if (checkpoint_every > 0) and (ckpt.step > 0) and \
                        (ckpt.step % checkpoint_every == 0):
                    save_checkpoint_model_fn()

                # --- keep time of steps per second
                stop_time_forward_backward = time.time()
                step_time_forward_backward = \
                    stop_time_forward_backward - \
                    start_time_forward_backward

                tf.summary.scalar(name="training/epoch",
                                  data=int(ckpt.epoch),
                                  step=ckpt.step)
                tf.summary.scalar(name="training/learning_rate",
                                  data=optimizer.learning_rate,
                                  step=ckpt.step)
                tf.summary.scalar(name="training/steps_per_second",
                                  data=1.0 / (step_time_forward_backward + 0.00001),
                                  step=ckpt.step)
                # ---
                ckpt.step.assign_add(1)

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
    return

# ---------------------------------------------------------------------
