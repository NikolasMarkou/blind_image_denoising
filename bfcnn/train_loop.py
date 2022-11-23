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
from .loss import loss_function_builder
from .optimizer import optimizer_builder
from .pruning import prune_function_builder, get_conv2d_weights
from .model_denoiser import model_builder as model_denoise_builder
from .utilities import load_config, load_image, probabilistic_drop_off, clip_tensor
from .dataset import \
    dataset_builder, \
    AUGMENTATION_FN_STR, \
    DATASET_TRAINING_FN_STR, \
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
    noise_augmentation_fn = dataset_res[AUGMENTATION_FN_STR]
    geometric_augmentation_fn = dataset_res[GEOMETRIC_AUGMENTATION_FN_STR]

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
                        color_mode="rgb",
                        target_size=(512, 512),
                        normalize=True)
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

    # --- build the denoise model
    tf.summary.trace_on(graph=True, profiler=False)
    with summary_writer.as_default():
        models = \
            model_denoise_builder(config=config[MODEL_DENOISE_STR])

        # The function to be traced.
        @tf.function
        def optimized_model(x_input):
            return models.denoiser(x_input, training=False)

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

    # get each model
    denoiser = models.denoiser
    normalizer = models.normalizer
    denormalizer = models.denormalizer
    denoiser_decomposition = models.backbone

    # summary of model
    denoiser.summary(print_fn=logger.info)
    # save model so we can visualize it easier
    denoiser.save(
        filepath=os.path.join(model_dir, MODEL_DENOISE_DEFAULT_NAME_STR),
        include_optimizer=False)

    # --- test image
    def denoise_test_batch():
        x_noisy = \
            tf.random.truncated_normal(
                seed=0,
                mean=0.0,
                stddev=0.1,
                shape=test_images.shape) + \
            test_images
        x_noisy = clip_tensor(x_noisy)
        x_noisy_denormalized = \
            denormalizer(
                x_noisy,
                training=False)
        x_denoised = \
            denoiser(x_noisy, training=False)
        x_denoised_denormalized = \
            denormalizer(x_denoised, training=False)

        return x_noisy_denormalized, x_denoised_denormalized

    # --- create random image and iterate through the model
    def create_random_batch():
        x_iteration = 0
        x_random = \
            tf.random.truncated_normal(
                seed=0,
                mean=0.0,
                stddev=0.1,
                shape=random_batch_size)
        while x_iteration < random_batch_iterations:
            x_random = clip_tensor(x_random)
            x_random = denoiser(x_random, training=False)
            x_iteration += 1
        return denormalizer(x_random, training=False)

    # --- train the model
    with summary_writer.as_default():
        checkpoint = \
            tf.train.Checkpoint(
                step=global_step,
                epoch=global_epoch,
                optimizer=optimizer,
                model_denoise=denoiser,
                model_normalize=normalizer,
                model_denormalize=denormalizer,
                model_denoise_decomposition=denoiser_decomposition)
        manager = \
            tf.train.CheckpointManager(
                checkpoint=checkpoint,
                directory=model_dir,
                max_to_keep=checkpoints_to_keep)
        status = \
            checkpoint.restore(manager.latest_checkpoint).expect_partial()

        # --- define denoise fn
        @tf.function
        def denoise_fn(x_input: tf.Tensor) -> tf.Tensor:
            # normalize
            x_tmp = normalizer(x_input, training=False)
            # denoise
            x_tmp = denoiser(x_tmp, training=True)
            # denormalize
            return denormalizer(x_tmp, training=False)

        # --- define decompose fn
        @tf.function
        def decompose_fn(x_input):
            return denoiser_decomposition(x_input, training=False)

        # --- dataset options
        options = tf.data.Options()
        options.deterministic = False
        options.autotune.enabled = True
        options.autotune.cpu_budget = 6
        options.threading.max_intra_op_parallelism = 4
        options.threading.private_threadpool_size = 16

        # ---
        while global_epoch < global_total_epochs:
            logger.info("epoch: {0}, step: {1}".format(
                int(global_epoch), int(global_step)))

            # --- pruning strategy
            if use_prune and (global_epoch >= prune_start_epoch):
                logger.info(f"pruning weights at step [{int(global_step)}]")
                denoiser = prune_fn(model=denoiser)

            model_denoise_weights = \
                denoiser.trainable_weights
            dataset_training_iterator = \
                iter(dataset_training.with_options(options))

            # --- iterate over the batches of the dataset
            while True:
                try:
                    start_time = time.time()

                    # --- do this for small batches
                    grads_batch = None
                    for i in range(no_iterations_per_batch):
                        input_batch = dataset_training_iterator.get_next()
                        input_batch = geometric_augmentation_fn(input_batch)
                        noisy_batch = noise_augmentation_fn(input_batch)

                        # Open a GradientTape to record the operations run
                        # during the forward pass,
                        # which enables auto-differentiation.
                        with tf.GradientTape() as tape:
                            # run the forward pass of the layer.
                            # The operations that the layer applies
                            # to its inputs are going to be recorded
                            # on the GradientTape.
                            denoised_batch = denoise_fn(noisy_batch)

                            # compute the loss value for this mini-batch
                            loss_map = \
                                loss_fn(
                                    input_batch=input_batch,
                                    noisy_batch=noisy_batch,
                                    model_losses=denoiser.losses,
                                    prediction_batch=denoised_batch)
                            grads = \
                                tape.gradient(
                                    target=loss_map[MEAN_TOTAL_LOSS_STR],
                                    sources=model_denoise_weights)

                        if grads_batch is None:
                            grads_batch = grads
                        else:
                            for j in range(len(grads)):
                                grads_batch[j] += grads[j]

                    # --- apply weights
                    optimizer.apply_gradients(
                        grads_and_vars=zip(grads_batch, model_denoise_weights))

                    # --- add loss summaries for tensorboard
                    for name, key in [
                        ("quality/psnr", PSNR_STR),
                        ("loss/mae", MAE_LOSS_STR),
                        ("loss/kl_loss", KL_LOSS_STR),
                        ("loss/total", MEAN_TOTAL_LOSS_STR),
                        ("quality/nae_noise", NAE_NOISE_STR),
                        ("loss/nae", NAE_PREDICTION_LOSS_STR),
                        ("loss/regularization", REGULARIZATION_LOSS_STR),
                        ("quality/nae_improvement", NAE_IMPROVEMENT_QUALITY_STR)
                    ]:
                        if key in loss_map:
                            tf.summary.scalar(
                                name=name,
                                data=loss_map[key],
                                step=global_step)

                    # --- add image prediction for tensorboard
                    if (global_step % visualization_every) == 0:
                        test_input_batch = None
                        test_output_batch = None
                        if use_test_images:
                            test_input_batch, test_output_batch = denoise_test_batch()
                        visualize(
                            global_step=global_step,
                            input_batch=input_batch,
                            noisy_batch=noisy_batch,
                            random_batch=create_random_batch(),
                            test_input_batch=test_input_batch,
                            test_output_batch=test_output_batch,
                            prediction_batch=denoised_batch,
                            visualization_number=visualization_number)
                        # add weight visualization
                        tf.summary.histogram(
                            data=get_conv2d_weights(model=denoiser),
                            step=global_step,
                            buckets=weight_buckets,
                            name="training/weights")

                    # --- add image decomposition
                    if decomposition_every > 0 and \
                            (global_step % decomposition_every) == 0:
                        test_image = test_images[0, :, :, :]
                        test_image = tf.expand_dims(test_image, axis=0)
                        test_image = tf.image.resize(test_image, size=(128, 128))
                        decomposed_image = decompose_fn(test_image)
                        # squash decomposition between [-1, +1] interval
                        decomposed_image = tf.tanh(decomposed_image)
                        # move to [0, +1] interval
                        decomposed_image = (decomposed_image + 1) / 2
                        decomposed_image = tf.transpose(decomposed_image, perm=(3, 1, 2, 0))
                        tf.summary.image(
                            name=f"test_output_decomposition_0",
                            step=global_step,
                            data=decomposed_image,
                            max_outputs=12)

                    # --- prune conv2d
                    if use_prune and (global_epoch >= prune_start_epoch) and \
                            (int(prune_steps) != -1) and (global_step % prune_steps == 0):
                        logger.info(f"pruning weights at step [{int(global_step)}]")
                        denoiser = prune_fn(model=denoiser)
                        model_denoise_weights = denoiser.trainable_weights

                    # --- check if it is time to save a checkpoint
                    if checkpoint_every > 0 and \
                            (global_step % checkpoint_every == 0):
                        logger.info("checkpoint at step: {0}".format(
                            int(global_step)))
                        manager.save()
                        # save model so we can visualize it easier
                        denoiser.save(
                            os.path.join(model_dir, MODEL_DENOISE_DEFAULT_NAME_STR))

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
                                int(total_steps)))
                            break
                except tf.errors.OutOfRangeError:
                    break

            # --- end of the epoch
            logger.info("checkpoint at end of epoch: {0}".format(
                int(global_epoch)))
            global_epoch.assign_add(1)
            manager.save()
            # save model so we can visualize it easier
            denoiser.save(
                os.path.join(model_dir, MODEL_DENOISE_DEFAULT_NAME_STR))

    logger.info("finished training")

# ---------------------------------------------------------------------
