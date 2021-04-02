"""
Provide some ready made training configurations for common datasets
"""

import os
import keras
import argparse
import tensorflow as tf
from keras import datasets
import matplotlib.pyplot as plt

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
    EPOCHS = 100
    FILTERS = 32
    NO_LAYERS = 5
    MIN_STD = 1.0
    MAX_STD = 100.0
    LR_DECAY = 0.9
    LR_INITIAL = 0.1
    BATCH_SIZE = 64
    CLIP_NORMAL = 1.0
    INPUT_SHAPE = (28, 28, 1)
    PRINT_EVERY_N_BATCHES = 1000

    # --- build model
    logger.info("building mnist model")
    model = \
        BFCNN(
            input_dims=INPUT_SHAPE,
            no_layers=NO_LAYERS,
            filters=FILTERS,
            kernel_regularizer=keras.regularizers.l2(0.001))

    # --- loading dataset
    logger.info("loading mnist dataset")
    (x_train, y_train), _ = datasets.mnist.load_data()

    # --- train model
    logger.info("training mnist model")
    trained_model, history = \
        BFCNN.train(
            model=model,
            dataset=x_train,
            input_dims=INPUT_SHAPE,
            epochs=EPOCHS,
            clip_norm=CLIP_NORMAL,
            batch_size=BATCH_SIZE,
            min_noise_std=MIN_STD,
            max_noise_std=MAX_STD,
            lr_initial=LR_INITIAL,
            lr_decay=LR_DECAY,
            print_every_n_batches=PRINT_EVERY_N_BATCHES)

    return trained_model, history

# ==============================================================================


def train_cifar10():

    # --- setup environment
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    tf.compat.v1.disable_eager_execution()
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    # --- set variables
    EPOCHS = 100
    FILTERS = 32
    NO_LAYERS = 5
    MIN_STD = 1.0
    MAX_STD = 100.0
    LR_DECAY = 0.9
    LR_INITIAL = 0.1
    BATCH_SIZE = 64
    CLIP_NORMAL = 1.0
    INPUT_SHAPE = (32, 32, 3)
    PRINT_EVERY_N_BATCHES = 1000

    # --- build model
    logger.info("building cifar10 model")
    model = \
        BFCNN(
            input_dims=INPUT_SHAPE,
            no_layers=NO_LAYERS,
            filters=FILTERS,
            kernel_regularizer=keras.regularizers.l2(0.01))

    # --- loading dataset
    logger.info("loading cifar10 dataset")
    (x_train, y_train), _ = datasets.cifar10.load_data()

    # --- train model
    logger.info("training cifar10 model")
    trained_model, history = \
        BFCNN.train(
            model=model,
            dataset=x_train,
            input_dims=INPUT_SHAPE,
            epochs=EPOCHS,
            clip_norm=CLIP_NORMAL,
            batch_size=BATCH_SIZE,
            min_noise_std=MIN_STD,
            max_noise_std=MAX_STD,
            lr_initial=LR_INITIAL,
            lr_decay=LR_DECAY,
            print_every_n_batches=PRINT_EVERY_N_BATCHES)

    return trained_model, history

# ==============================================================================


if __name__ == "__main__":

    # --- parse arguments
    parser = \
        argparse.ArgumentParser(
            description="Train denoiser on common dataset")
    parser.add_argument("--dataset",
                        type=str,
                        default="mnist",
                        const="mnist",
                        nargs="?",
                        choices=["mnist", "cifar10"],
                        help="datasets you can train with")
    args = parser.parse_args()

    # --- train model
    if args.dataset == "mnist":
        trained_model, history = train_mnist()
    elif args.dataset == "cifar10":
        trained_model, history = train_cifar10()
    else:
        raise NotImplementedError(
            "don't know how to train [{0}]".format(
                args.dataset))

    # --- save model
    model_filepath = os.path.join("./", "model.h5")
    logger.info("saving model to {0}".format(model_filepath))
    trained_model.save(model_filepath)

    # --- summarize history for loss
    plt.figure(figsize=(15, 5))
    plt.plot(history.history["loss"],
             marker="o",
             color="red",
             linewidth=3,
             markersize=6)
    plt.grid(True)
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train"], loc="upper right")
    history_filepath = os.path.join("./", "history.png")
    logger.info("saving history to {0}".format(history_filepath))
    plt.savefig(history_filepath)
    plt.close()
    #
    exit(0)

# ==============================================================================
