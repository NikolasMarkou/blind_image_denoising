import os
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
from keras.callbacks import Callback

# ==============================================================================

from .utilities import *
from .custom_logger import logger

# ==============================================================================


def collage(images_batch):
    shape = images_batch.shape
    no_images = shape[0]
    images = []
    result = None
    width = np.ceil(np.sqrt(no_images))
    height = no_images / width

    for i in range(no_images):
        images.append(images_batch[i, :, :, :])

        if len(images) % width == 0:
            if result is None:
                result = np.hstack(images)
            else:
                tmp = np.hstack(images)
                result = np.vstack([result, tmp])
            images.clear()
    return result

# ==============================================================================


class SaveIntermediateResultsCallback(Callback):

    def __init__(self,
                 images,
                 model,
                 run_folder: str = "./",
                 initial_epoch: int = 0,
                 resize_shape=(256, 256)):
        """
        Callback for saving the intermediate result image

        :param run_folder:
        :param initial_epoch:
        :param images:
        :param model:
        :param resize_shape:
        """
        self._model = model
        self._images = images
        self._epoch = initial_epoch
        self._run_folder = run_folder
        self._resize_shape = resize_shape
        images_path = os.path.join(self._run_folder, "images")
        pathlib.Path(images_path).mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------

    def on_batch_end(self, batch, logs={}):
        predictions = self._model.predict(self._images)
        predictions = predictions / 255.0
        # --- create collage of the predictions
        x = collage(predictions)
        # --- resize to output size
        x = resize(x, self._resize_shape, order=0)
        filepath_x = os.path.join(
            self._run_folder,
            "images",
            "img_" + str(self._epoch).zfill(3) +
            "_" + str(batch) + ".png")
        # ---
        if len(x.shape) == 2 or x.shape[-1] == 1:
            if len(x.shape) == 3 and x.shape[-1] == 1:
                x = np.squeeze(x, axis=2)
            plt.imsave(filepath_x, x, cmap="gray_r")
        else:
            plt.imsave(filepath_x, x)

    # --------------------------------------------------

    def on_epoch_begin(self, epoch, logs={}):
        self._epoch += 1

# ==============================================================================
