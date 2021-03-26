"""
Provide some ready made training configurations for common datasets
"""

import os
import tensorflow as tf
from keras import datasets

# ==============================================================================

from .model import BFCNN
from .custom_logger import logger

# ==============================================================================


def train_mnist():
    # --- setup environment
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    tf.compat.v1.disable_eager_execution()
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    # --- set variables
    NO_LAYERS = 10
    MIN_STD = 0.1
    MAX_STD = 50.0
    EPOCHS = 20
    LR_DECAY = 0.8
    LR_INITIAL = 0.1
    BATCH_SIZE = 32
    CLIP_NORMAL = 1.0
    INPUT_SHAPE = (28, 28, 1)
    PRINT_EVERY_N_BATCHES = 1000
    # --- build model
    logger.info("building mnist model")
    model = \
        BFCNN(
            input_dims=INPUT_SHAPE,
            no_layers=NO_LAYERS)
    # --- loading dataset
    logger.info("loading mnist dataset")
    (x_train, y_train), _ = datasets.mnist.load_data()
    # --- train model
    logger.info("training mnist model")
    trained_model, history = \
        BFCNN.train(
            model=model,
            input_dims=INPUT_SHAPE,
            dataset=x_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            min_noise_std=MIN_STD,
            max_noise_std=MAX_STD,
            lr_initial=LR_INITIAL,
            lr_decay=LR_DECAY,
            clip_norm=CLIP_NORMAL,
            print_every_n_batches=PRINT_EVERY_N_BATCHES)
    return trained_model, history

# ==============================================================================


def train_cifar10():
    # --- setup environment
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    tf.compat.v1.disable_eager_execution()
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    # --- set variables
    NO_LAYERS = 10
    MIN_STD = 0.1
    MAX_STD = 50.0
    EPOCHS = 20
    LR_DECAY = 0.8
    LR_INITIAL = 0.1
    BATCH_SIZE = 32
    CLIP_NORMAL = 1.0
    INPUT_SHAPE = (32, 32, 3)
    PRINT_EVERY_N_BATCHES = 1000
    # --- build model
    logger.info("building cifar10 model")
    model = \
        BFCNN(
            input_dims=INPUT_SHAPE,
            no_layers=NO_LAYERS)
    # --- loading dataset
    logger.info("loading cifar10 dataset")
    (x_train, y_train), _ = datasets.cifar10.load_data()
    # --- train model
    logger.info("training cifar10 model")
    trained_model, history = \
        BFCNN.train(
            model=model,
            input_dims=INPUT_SHAPE,
            dataset=x_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            min_noise_std=MIN_STD,
            max_noise_std=MAX_STD,
            lr_initial=LR_INITIAL,
            lr_decay=LR_DECAY,
            clip_norm=CLIP_NORMAL,
            print_every_n_batches=PRINT_EVERY_N_BATCHES)
    return trained_model, history

# ==============================================================================
