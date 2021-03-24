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
                 model,
                 original_images,
                 noisy_images,
                 every_n_batches: int = 100,
                 run_folder: str = "./",
                 initial_epoch: int = 0,
                 resize_shape=(256, 256)):
        """
        Callback for saving the intermediate result image

        :param images:
        :param model:
        :param every_n_batches:
        :param run_folder:
        :param initial_epoch:
        :param resize_shape:
        """
        self._model = model
        self._epoch = initial_epoch
        self._run_folder = run_folder
        self._resize_shape = resize_shape
        self._noisy_images = noisy_images
        self._original_images = original_images
        self._every_n_batches = every_n_batches
        images_path = os.path.join(self._run_folder, "images")
        pathlib.Path(images_path).mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------

    def on_batch_end(self, batch, logs={}):
        if batch % self._every_n_batches != 0:
            return
        predictions = self._model.predict(self._noisy_images)
        predictions = predictions / 255.0
        # --- create histograms
        x_hist = np.histogram(predictions, bins=255, range=(0.0, 1.0), density=True)
        y_hist = np.histogram(self._noisy_images / 255.0, bins=255, range=(0.0, 1.0), density=True)
        z_hist = np.histogram(self._original_images / 255.0, bins=255, range=(0.0, 1.0), density=True)
        # --- create collage of the predictions
        x = collage(predictions)
        y = collage(self._noisy_images / 255.0)
        z = collage(self._original_images / 255.0)
        # --- resize to output size
        x = resize(x, self._resize_shape, order=0)
        y = resize(y, self._resize_shape, order=0)
        z = resize(z, self._resize_shape, order=0)
        # --- concat
        result = np.concatenate((z, y, x), axis=1)
        filepath_result = os.path.join(
            self._run_folder,
            "images",
            "img_" + str(self._epoch).zfill(3) +
            "_" + str(batch) + ".png")
        # ---
        if len(result.shape) == 2 or result.shape[-1] == 1:
            if len(result.shape) == 3 and result.shape[-1] == 1:
                result = np.squeeze(result, axis=2)
            plt.imsave(filepath_result, result, cmap="gray_r")
        else:
            plt.imsave(filepath_result, result)

    # --------------------------------------------------

    def on_epoch_begin(self, epoch, logs={}):
        self._epoch += 1

# ==============================================================================
