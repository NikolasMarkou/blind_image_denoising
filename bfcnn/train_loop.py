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
from .optimizer import optimizer_builder
from .loss import loss_function_builder
from .utilities import \
    load_config, \
    create_checkpoint, \
    save_config, \
    multiscales_generator_fn
from .visualize import \
    visualize_weights_boxplot, \
    visualize_weights_heatmap, \
    visualize_gradient_boxplot

# ---------------------------------------------------------------------

CURRENT_DIRECTORY = os.path.realpath(os.path.dirname(__file__))

tf.random.set_seed(0)

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

    # --- save configuration into path, makes it easier to compare afterwards
    save_config(
        config=config,
        filename=os.path.join(str(model_dir), CONFIG_PATH_STR))

    # --- build dataset
    dataset = dataset_builder(config[DATASET_STR])

    batch_size = dataset.batch_size
    input_shape = dataset.input_shape
    dataset_training = dataset.training
    no_color_channels = config[DATASET_STR][INPUT_SHAPE_STR][-1]

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
    # controls how different outputs of the model get discounted in the loss
    # 1.0 all equal
    # discount lower depth losses as epochs carry forward
    # assuming output_discount_factor = 0.25
    # for percentage_done in [0.0, 0.25, 0.5, 0.75, 1.0]:
    #     x = [0.25 ** (float(i) * percentage_done) for i in range(5)]
    #     print(x)
    # [1.0, 1.0, 1.0, 1.0, 1.0]
    # [1.0, 0.707, 0.5, 0.353, 0.25]
    # [1.0, 0.5, 0.25, 0.125, 0.0625]
    # [1.0, 0.353, 0.125, 0.0441, 0.015625]
    # [1.0, 0.25, 0.0625, 0.015625, 0.00390625]
    output_discount_factor = train_config.get("output_discount_factor", 1.0)
    if output_discount_factor > 1.0 or output_discount_factor < 0.0:
        raise ValueError(f"output_discount_factor [{output_discount_factor}] "
                         f"must be between 0.0 and 1.0")
    #
    ssl_epochs = train_config.get("ssl_epochs", -1)
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

    # --- train the model
    with (tf.summary.create_file_writer(model_dir).as_default()):
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
            os.path.join(model_dir, MODEL_HYDRA_DEFAULT_NAME_STR))

        manager = \
            tf.train.CheckpointManager(
                checkpoint_name="ckpt",
                checkpoint=ckpt,
                directory=model_dir,
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

            save_checkpoint_model_fn()

        # find indices of denoiser, materials segmentation, bg lumen wall segmentation
        # first third is denoiser, second third is materials, last third is bg_lumen_wall
        model_no_outputs = len(ckpt.model.outputs)
        logger.info(f"model number of outputs: [{model_no_outputs}]")

        denoiser_index = [
            i for i in range(0, int(model_no_outputs))
        ]
        logger.info(f"model denoiser_index: {denoiser_index}")
        denoiser_loss_fn_list = [
            tf.function(
                func=loss_fn_map[DENOISER_LOSS_FN_STR],
                input_signature=[
                    tf.TensorSpec(shape=[batch_size, None, None, no_color_channels], dtype=tf.float32),
                    tf.TensorSpec(shape=[batch_size, None, None, no_color_channels], dtype=tf.float32),
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
                concrete_functions=True
            )

        @tf.function(reduce_retracing=True, jit_compile=False)
        def train_step(n: tf.Tensor) -> List[tf.Tensor]:
            return ckpt.model(n, training=True)

        @tf.function(reduce_retracing=True, jit_compile=False)
        def test_step(n: tf.Tensor) -> tf.Tensor:
            results = ckpt.model(n, training=False)
            return results[denoiser_index[0]]

        @tf.function(
            autograph=True,
            jit_compile=False,
            reduce_retracing=True)
        def train_step_single_gpu(
                _input_image_batch: tf.Tensor,
                _noisy_image_batch: tf.Tensor,
                _depth_weight: tf.Tensor,
                _percentage_done: tf.Tensor,
                _trainable_variables: List):

            _all_denoiser_loss = [None] * len(denoiser_index)
            _total_denoiser_loss = tf.constant(0.0, dtype=tf.float32)

            _scale_gt_image_batch = \
                multiscales_fn(_input_image_batch)

            with tf.GradientTape() as tape:
                _predictions = train_step(_noisy_image_batch)

                # get denoise loss for each depth,
                for i in range(len(denoiser_index)):
                    _loss = denoiser_loss_fn_list[i](
                        input_batch=_scale_gt_image_batch[i],
                        predicted_batch=_predictions[i])
                    _total_denoiser_loss += \
                        _loss[TOTAL_LOSS_STR] * _depth_weight[i]
                    _all_denoiser_loss[i] = _loss

                # combine losses
                _model_loss = \
                    model_loss_fn(model=ckpt.model)
                _total_loss = \
                    _total_denoiser_loss * (1.0 - _percentage_done) + \
                    _model_loss[TOTAL_LOSS_STR]
                _grads = \
                    tape.gradient(target=_total_loss,
                                  sources=_trainable_variables)

            return (
                _total_loss,
                _model_loss,
                _all_denoiser_loss,
                _predictions,
                _grads
            )

        if ckpt.step == 0:
            tf.summary.trace_on(graph=True, profiler=False)

            # run a single step
            _ = train_step(iter(dataset_training).get_next()[0])

            tf.summary.trace_export(
                step=ckpt.step,
                name="model_hydra")
            tf.summary.flush()
            tf.summary.trace_off()

        sizes = [
            (int(input_shape[0] / (2 ** i)), int(input_shape[1] / (2 ** i)))
            for i in range(len(denoiser_index))
        ]

        # ---
        finished_training = False
        counter = tf.Variable(0, dtype=tf.uint32, trainable=False)
        gradients = [
            tf.constant(tf.zeros_like(v))
            for v in ckpt.model.trainable_variables
        ]

        @tf.function(
            reduce_retracing=True)
        def apply_grads(internal_gradients):
            optimizer.apply_gradients(
                zip(internal_gradients, ckpt.model.trainable_variables))

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
            depth_weight = [output_discount_factor ** (float(i) * percentage_done) for i in range(len(denoiser_index))]
            depth_weight_str = [
                "{0:.2f}".format(d)
                for d in depth_weight
            ]
            depth_weight = tf.constant(depth_weight, dtype=tf.float32)
            logger.info(f"weight per output index: {depth_weight_str}")

            # --- initialize iterators
            epoch_finished_training = False
            dataset_train = iter(dataset_training)
            total_loss = tf.constant(0.0, dtype=tf.float32)

            # --- check if total steps reached
            if total_steps != -1:
                if total_steps <= ckpt.step:
                    logger.info("total_steps reached [{0}]".format(
                        int(total_steps)))
                    finished_training = True

            # --- iterate over the batches of the dataset
            while not finished_training and \
                    not epoch_finished_training:

                start_time_forward_backward = time.time()

                logger.info("epoch [{0}], step [{1}]".format(
                    int(ckpt.epoch), int(ckpt.step)))

                # --- training percentage
                if total_epochs > 0:
                    percentage_done = float(ckpt.epoch) / float(total_epochs)
                elif total_steps > 0:
                    percentage_done = float(ckpt.step) / float(total_steps)
                else:
                    percentage_done = 0.0
                percentage_done = tf.constant(percentage_done, dtype=tf.float32)

                # --- check if total steps reached
                if total_steps != -1:
                    if total_steps <= ckpt.step:
                        logger.info("total_steps reached [{0}]".format(
                            int(total_steps)))
                        finished_training = True

                for input_image_batch, noisy_image_batch in dataset_train:
                    if counter == 0:
                        start_time_forward_backward = time.time()
                        # zero out gradients
                        for i in range(len(gradients)):
                            gradients[i] *= 0.0

                    total_loss = tf.constant(0.0, dtype=tf.float32)
                    model_loss = {
                        REGULARIZATION_LOSS_STR: tf.constant(0.0, dtype=tf.float32),
                        TOTAL_LOSS_STR: tf.constant(0.0, dtype=tf.float32)
                    }

                    all_denoiser_loss = [
                        {
                            MAE_LOSS_STR: tf.constant(0.0, dtype=tf.float32),
                            SSIM_LOSS_STR: tf.constant(0.0, dtype=tf.float32),
                            TOTAL_LOSS_STR: tf.constant(0.0, dtype=tf.float32)
                        }
                        for i in range(len(denoiser_index))
                    ]
                    predictions = input_image_batch
                    grads = gradients

                    # total_loss, model_loss, all_denoiser_loss, predictions, grads = \
                    #     train_step_single_gpu(
                    #         _input_image_batch=input_image_batch,
                    #         _noisy_image_batch=noisy_image_batch,
                    #         _depth_weight=depth_weight,
                    #         _percentage_done=percentage_done,
                    #         _trainable_variables=trainable_variables)


                    for i, grad in enumerate(grads):
                        gradients[i] += grad

                    if counter >= gpu_batches_per_step:
                        counter.assign(value=0)
                        # sanitize and average gradients
                        for i in range(len(gradients)):
                            gradients[i] /= float(gpu_batches_per_step)
                        # !!! IMPORTANT !!!!
                        # apply gradient to change weights
                        # this is a hack to stop retracing the update function
                        # https://stackoverflow.com/questions/77028664/tf-keras-optimizers-adam-apply-gradients-triggers-tf-function-retracing
                        apply_grads(
                            internal_gradients=gradients)
                        # optimizer.apply_gradients(
                        #     zip(gradients, ckpt.model.trainable_variables))
                    else:
                        counter.assign_add(delta=1)
                        continue


                    # --- add loss summaries for tensorboard
                    for i, d in enumerate(all_denoiser_loss):
                        tf.summary.scalar(name=f"loss_denoiser/scale_{i}/mae",
                                          data=d[MAE_LOSS_STR],
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
                        if isinstance(predictions, list):
                            for i, d in enumerate(predictions):
                                tf.summary.image(name=f"denoiser/scale_{i}/output", data=d / 255,
                                                 max_outputs=visualization_number, step=ckpt.step)
                        else:
                            tf.summary.image(name=f"denoiser/scale_0/output", data=predictions / 255,
                                             max_outputs=visualization_number, step=ckpt.step)

                        # --- add gradient activity
                        # gradient_activity = \
                        #     visualize_gradient_boxplot(
                        #         gradients=gradients_moving_average,
                        #         trainable_variables=trainable_variables) / 255
                        # tf.summary.image(name="weights/gradients",
                        #                  data=gradient_activity,
                        #                  max_outputs=visualization_number,
                        #                  step=ckpt.step,
                        #                  description="gradient activity")

                        # --- add weights distribution
                        weights_boxplot = \
                            visualize_weights_boxplot(
                                trainable_variables=ckpt.model.trainable_variables) / 255
                        tf.summary.image(name="weights/boxplot",
                                         data=weights_boxplot,
                                         max_outputs=visualization_number,
                                         step=ckpt.step,
                                         description="weights boxplot")
                        weights_heatmap = \
                            visualize_weights_heatmap(
                                trainable_variables=ckpt.model.trainable_variables) / 255
                        tf.summary.image(name="weights/heatmap",
                                         data=weights_heatmap,
                                         max_outputs=visualization_number,
                                         step=ckpt.step,
                                         description="weights heatmap")

                    # --- check if it is time to save a checkpoint
                    if checkpoint_every > 0 and ckpt.step > 0 and \
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
                    if total_steps > 0:
                        if total_steps <= ckpt.step:
                            logger.info("total_steps reached [{0}]".format(
                                int(total_steps)))
                            finished_training = True

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
