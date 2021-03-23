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
    MIN_STD = 0.1
    MAX_STD = 10
    EPOCHS = 100
    STEP_SIZE = 30
    LR_DECAY = 0.5
    BATCH_SIZE = 32
    INITIAL_EPOCH = 0
    CLIP_NORMAL = 1.0
    LEARNING_RATE = 0.01
    INPUT_SHAPE = (28, 28, 1)
    PRINT_EVERY_N_BATCHES = 1000
    # --- build model
    logger.info("building model")
    model = BFCNN(input_dims=INPUT_SHAPE)
    # --- loading dataset
    logger.info("loading dataset")
    (x_train, y_train), _ = datasets.mnist.load_data()
    # --- train model
    logger.info("training model")
    trained_model = \
        BFCNN.train(
            model=model.trainable_model,
            input_dims=INPUT_SHAPE,
            dataset=x_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            print_every_n_batches=PRINT_EVERY_N_BATCHES)
    # --- save model
    logger.info("todo")

# ==============================================================================
