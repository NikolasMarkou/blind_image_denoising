import os
import math
import keras
import pathlib
import numpy as np
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator

# ==============================================================================

from .utilities import *
from .custom_logger import logger
from .custom_schedule import step_decay_schedule
from .custom_callbacks import SaveIntermediateResultsCallback

# ==============================================================================


class BFCNN:
    """
    Bias Free De-noising Convolutional Neural Network
    """

    def __init__(
            self,
            input_dims,
            min_value=0.0,
            max_value=255.0,
            channels_index=2):
        """

        :param input_dims:
        :param min_value:
        :param max_value:
        :param channels_index:
        """
        self._input_dims = input_dims
        self._min_value = min_value
        self._max_value = max_value
        self._channels_index = channels_index
        self._trainable_model = self.build_model(input_dims)

    # --------------------------------------------------

    @property
    def trainable_model(self):
        return self._trainable_model

    # --------------------------------------------------

    @staticmethod
    def build_model(
            input_dims,
            no_layers: int = 5,
            kernel_size: int = 3,
            filters: int = 32,
            min_value: float = 0.0,
            max_value: float = 255.0,
            channel_index: int = 2,
            kernel_regularizer=None,
            kernel_initializer=None) -> keras.Model:
        """
        Build Bias Free CNN model

        :param input_dims:
        :param no_layers: number of convolutional layers
        :param kernel_size: convolution kernel size
        :param filters: number of filters per convolutional layer
        :param max_value:
        :param min_value:
        :param channel_index:
        :param kernel_initializer:
        :param kernel_regularizer:
        :return: keras model
        """
        # --- argument checking
        # TODO
        # --- variables
        negative_slope = 0.1

        # --- build bfcnn
        model_input = keras.Input(shape=input_dims)

        # --- normalize input from [min_value, max_value] to [-1.0, +1.0]
        x = keras.layers.Lambda(layer_normalize, name="normalize")([
            model_input, float(min_value), float(max_value)])

        # --- add layers
        for i in range(no_layers):
            previous_layer = x
            x = keras.layers.Conv2D(
                filters=filters,
                kernel_size=kernel_size,
                use_bias=False,
                strides=(1, 1),
                padding="same",
                activation="linear",
                kernel_regularizer=kernel_regularizer,
                kernel_initializer=kernel_initializer)(x)
            x = keras.layers.ReLU(negative_slope=negative_slope)(x)
            x = keras.layers.Conv2D(
                filters=filters,
                kernel_size=kernel_size,
                use_bias=False,
                strides=(1, 1),
                padding="same",
                activation="linear",
                kernel_regularizer=kernel_regularizer,
                kernel_initializer=kernel_initializer)(x)
            x = keras.layers.BatchNormalization(center=False)(x)
            # add skip on all but the first
            if i > 0:
                x = previous_layer - x
                x = keras.layers.ReLU(negative_slope=negative_slope)(x)

        # --- output to original channels
        x = keras.layers.Conv2D(
            filters=input_dims[channel_index],
            kernel_size=kernel_size,
            strides=(1, 1),
            padding="same",
            activation="linear",
            use_bias=False,
            kernel_regularizer=kernel_regularizer,
            kernel_initializer=kernel_initializer)(x)
        # --- denormalize output from [-1.0, +1.0] [min_value, max_value]
        model_output = \
            keras.layers.Lambda(layer_denormalize, name="denormalize")([
                x, float(min_value), float(max_value)])
        # --- wrap model
        return keras.Model(
            inputs=model_input,
            outputs=model_output)

    # --------------------------------------------------

    @staticmethod
    def train(model,
              input_dims,
              dataset,
              batch_size: int = 32,
              epochs: int = 1,
              lr_initial: float = 0.01,
              lr_decay: float = 1.0,
              step_size: int = 1000,
              clip_norm: float = 1.0,
              min_noise_std: float = 0.1,
              max_noise_std: float = 10,
              print_every_n_batches: int = 100,
              run_folder: str = "./training/",
              save_checkpoint_weights: bool = False):
        """
        Train the model using the dataset and return a trained model

        :param model:
        :param input_dims:
        :param dataset: dataset to train on
        :param batch_size:
        :param epochs:
        :param run_folder:
        :param print_every_n_batches:
        :param lr_initial: initial learning rate
        :param lr_decay: decay of learning rate per step size
        :param step_size:
        :param clip_norm:
        :param max_noise_std:
        :param min_noise_std:
        :param save_checkpoint_weights:
        :return:
        """
        # --- argument checking

        # --- set variables
        initial_epoch = 0
        batches_per_epoch = int(math.ceil(len(dataset) / batch_size))

        # --- define mae reconstruction loss
        def mae_loss(y_true, y_pred):
            tmp_pixels = K.abs(y_true[0] - y_pred[0])
            return K.mean(tmp_pixels)

        optimizer = keras.optimizers.Adagrad(
            lr=lr_initial,
            clipnorm=clip_norm)

        model.compile(
            optimizer=optimizer,
            loss=mae_loss,
            metrics=[mae_loss])

        # --- fix dataset dimensions if they dont match
        if isinstance(dataset, np.ndarray):
            if len(input_dims) == 3:
                if len(dataset.shape[1:]) != len(input_dims):
                    dataset = np.expand_dims(dataset, axis=3)
            dataset = dataset.astype("float32")

        # --- create data generator
        data_generator = \
            ImageDataGenerator(
                featurewise_center=False,
                featurewise_std_normalization=False,
                zoom_range=0.1,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                horizontal_flip=True)

        data_generator.fit(dataset, augment=False, seed=0)

        # --- define callbacks
        subset_size = 25
        subset_std = 20
        subset = dataset[0:subset_size, :, :, :] + \
                 np.random.normal(0.0, subset_std, (subset_size,) + input_dims)
        callback_intermediate_results = \
            SaveIntermediateResultsCallback(
                model=model,
                run_folder=run_folder,
                initial_epoch=initial_epoch,
                images=subset)
        lr_schedule = \
            step_decay_schedule(
                initial_lr=lr_initial,
                decay_factor=lr_decay,
                step_size=step_size)
        weights_path = os.path.join(run_folder, "weights")
        pathlib.Path(weights_path).mkdir(parents=True, exist_ok=True)

        checkpoint_filepath = os.path.join(
            run_folder,
            "weights/weights-{epoch:03d}-{loss:.2f}.h5")
        checkpoint1 = keras.callbacks.ModelCheckpoint(
            checkpoint_filepath,
            save_weights_only=True,
            verbose=1)
        checkpoint2 = keras.callbacks.ModelCheckpoint(
            os.path.join(
                run_folder,
                "weights/weights.h5"),
            save_weights_only=True,
            verbose=1)

        callbacks_fns = [
            lr_schedule,
            callback_intermediate_results
        ]

        if save_checkpoint_weights:
            callbacks_fns += [checkpoint1, checkpoint2]

        # --- manually flow to augment the noise
        logger.info("begin training | batches per epoch {0}".format(batches_per_epoch))
        report_batch = 0

        for e in range(epochs):
            logger.info("epoch {0}".format(e))
            epoch_batch = 0

            # iterate over random batches
            for x_batch in data_generator.flow(x=dataset,
                                               shuffle=True,
                                               batch_size=batch_size):
                # adjust the std of the noise
                std = np.random.uniform(low=min_noise_std, high=max_noise_std)

                # add noise to create the noisy input and clip to constraint
                y_batch = x_batch + np.random.normal(0.0, std, (batch_size,) + input_dims)

                model.fit(y_batch, x_batch, verbose=False)

                # show progress on denoising
                if report_batch % print_every_n_batches == 0:
                    # print loss
                    evaluation_results = model.evaluate(y_batch, x_batch)
                    logger.info("[batch:{0}] loss: {1:.2f}".format(
                        report_batch, evaluation_results[0]))
                    # show collage of images
                    callback_intermediate_results.on_batch_end(batch=report_batch)

                # we need to break the loop by hand
                # because the generator loops indefinitely
                if epoch_batch >= batches_per_epoch:
                    break

                epoch_batch += 1
                report_batch += 1
        return model

    # --------------------------------------------------

# ==============================================================================
