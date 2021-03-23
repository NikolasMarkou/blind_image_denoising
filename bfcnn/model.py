import os
import keras
import numpy as np
from keras import backend as K

# ==============================================================================

from .utilities import *
from .custom_logger import logger
from .custom_schedule import step_decay_schedule
from .custom_callbacks import SaveIntermediateResultsCallback

# ==============================================================================


class BFCNN:
    """
    Bias Free Convolutional Neural Network
    """

    # --------------------------------------------------

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
        self._model = self.build_model(input_dims)
        self._trainable_model = self.trainable_wrap_model(input_dims, self._model)



    # --------------------------------------------------

    def compile(self,
                learning_rate: float = 1.0,
                clip_norm: float = 1.0):
        """

        :param clip_norm:
        :param learning_rate:
        :return:
        """
        self._learning_rate = learning_rate

        optimizer = keras.optimizers.Adagrad(
            lr=self._learning_rate,
            clipnorm=clip_norm)

        # --------- Define mae reconstruction loss
        def mae_loss(y_true, y_pred):
            tmp_pixels = K.abs(y_true[1] - y_pred[0])
            return K.mean(tmp_pixels)

        self._trainable_model.compile(
            optimizer=optimizer,
            loss=mae_loss,
            metrics=[mae_loss])

    # --------------------------------------------------

    def train(self,
              x,
              batch_size,
              epochs,
              run_folder,
              print_every_n_batches=100,
              initial_epoch=0,
              step_size=1,
              lr_decay=1,
              save_checkpoint_weights=False):
        """

        :param x:
        :param batch_size:
        :param epochs:
        :param run_folder:
        :param print_every_n_batches:
        :param initial_epoch:
        :param step_size:
        :param lr_decay:
        :param save_checkpoint_weights:
        :return:
        """
        custom_callback = SaveIntermediateResultsCallback(
            run_folder,
            print_every_n_batches,
            initial_epoch,
            x_train[0:16, :, :, :],
            self)
        lr_schedule = step_decay_schedule(
            initial_lr=self._learning_rate,
            decay_factor=lr_decay,
            step_size=step_size)
        weights_path = os.path.join(run_folder, "weights")
        if not os.path.exists(weights_path):
            os.mkdir(weights_path)
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
            custom_callback
        ]

        if save_checkpoint_weights:
            callbacks_fns += [checkpoint1, checkpoint2]

        self._model_trainable.fit(
            x_train,
            x_train,
            batch_size=batch_size,
            shuffle=True,
            epochs=epochs,
            initial_epoch=initial_epoch,
            callbacks=callbacks_fns)

    # --------------------------------------------------

    @staticmethod
    def trainable_wrap_model(
            input_dims,
            model: keras.Model) -> keras.Model:
        """
        Wrap model so we have 2 inputs one the noisy input and one the normal input

        :param input_dims:
        :param model:
        :return:
        """
        input_noisy = keras.Input(shape=input_dims, name="input_noisy")
        input_normal = keras.Input(shape=input_dims, name="input_normal")
        output_denoiser = model(input_noisy)
        return keras.Model(
            inputs=[input_noisy, input_normal],
            outputs=[output_denoiser])

    # --------------------------------------------------

    @staticmethod
    def build_model(
            input_dims,
            no_layers: int = 10,
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
        # --- build bfcnn
        model_input = keras.Input(shape=input_dims)
        # --- normalize input
        x = keras.layers.Lambda(layer_normalize, name="normalize")([
            model_input, float(min_value), float(max_value)])
        # --- add layers
        for i in range(no_layers):
            x = keras.layers.Conv2D(
                filters=filters,
                kernel_size=kernel_size,
                strides=(1, 1),
                padding="same",
                activation="linear",
                kernel_regularizer=kernel_regularizer,
                kernel_initializer=kernel_initializer)(x)
            x = keras.layers.ReLU(negative_slope=0.1)(x)
        # --- output to original channels
        x = keras.layers.Conv2D(
            filters=input_dims[channel_index],
            kernel_size=kernel_size,
            strides=(1, 1),
            padding="same",
            activation="linear",
            kernel_regularizer=kernel_regularizer,
            kernel_initializer=kernel_initializer)(x)
        # --- denormalize output
        model_output = \
            keras.layers.Lambda(layer_denormalize, name="denormalize")([
                x, float(min_value), float(max_value)])
        # --- wrap model
        return keras.Model(
            inputs=model_input,
            outputs=model_output)

# ==============================================================================
