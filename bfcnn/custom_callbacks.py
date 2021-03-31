import os
import glob
import pathlib
import matplotlib
import numpy as np
from typing import Tuple, List
import matplotlib.pyplot as plt
from skimage.transform import resize
from keras.callbacks import Callback

# ==============================================================================

from .utilities import *
from .custom_logger import logger

# ==============================================================================

matplotlib.use("Agg")

# ==============================================================================


class SaveIntermediateResultsCallback(Callback):

    RESULTS_EXTENSIONS = ".png"

    def __init__(self,
                 model,
                 original_images,
                 noisy_images,
                 every_n_batches: int = 100,
                 run_folder: str = "./",
                 initial_epoch: int = 0,
                 histogram_bins: int = 500,
                 histogram_range: Tuple = (-0.5, +0.5),
                 resize_shape: Tuple = (128, 128)):
        """
        Callback for saving the intermediate result image

        :param model: Model to run denoising with
        :param original_images: Clean test images
        :param noisy_images Noisy test images
        :param every_n_batches: Save intermediate results every so many batches
        :param run_folder: Save results in this folder
        :param initial_epoch: Start counting from this time
        :param resize_shape: Resize final collage to this resolution
        """

        self._model = model
        self._epoch = initial_epoch
        self._run_folder = run_folder
        self._bins = histogram_bins
        self._range = histogram_range
        self._resize_shape = resize_shape
        self._noisy_images = noisy_images
        self._original_images = original_images
        self._every_n_batches = every_n_batches
        # create training image path
        images_path = os.path.join(self._run_folder, "images")
        pathlib.Path(images_path).mkdir(parents=True, exist_ok=True)
        # delete images already in path
        logger.info("deleting existing training image in {0}".format(images_path))
        #
        for filename in glob.glob(
                images_path + "/*" + self.RESULTS_EXTENSIONS, recursive=True):
            try:
                os.remove(filename)
            except Exception as e:
                logger.error("Error while deleting file [{0}] : {1}".format(filename, e))

    # --------------------------------------------------

    def on_batch_end(self, batch, logs={}):
        if batch % self._every_n_batches != 0:
            return
        predictions = self._model.predict(self._noisy_images)
        # --- create collage of the predictions
        x = collage(predictions / 255.0)
        y = collage(self._noisy_images / 255.0)
        z = collage(self._original_images / 255.0)
        # --- resize to output size
        x = resize(x, self._resize_shape, order=0)
        y = resize(y, self._resize_shape, order=0)
        z = resize(z, self._resize_shape, order=0)
        # --- concat image and save result
        result = np.concatenate((z, y, x), axis=1)
        filepath_result = os.path.join(
            self._run_folder,
            "images",
            "img_" + str(self._epoch).zfill(3) +
            "_" + str(batch) + self.RESULTS_EXTENSIONS)
        if len(result.shape) == 2 or result.shape[-1] == 1:
            if len(result.shape) == 3 and result.shape[-1] == 1:
                result = np.squeeze(result, axis=2)
            plt.imsave(filepath_result, result, cmap="gray_r")
        else:
            plt.imsave(filepath_result, result)
        # --- save weights snapshot
        weights = get_conv2d_weights(self._model)
        filepath_result = os.path.join(
            self._run_folder,
            "images",
            "weights_" + str(self._epoch).zfill(3) +
            "_" + str(batch) + self.RESULTS_EXTENSIONS)
        plt.figure(figsize=(10, 3))
        plt.grid(True)
        plt.hist(x=weights,
                 bins=self._bins,
                 range=self._range,
                 histtype="bar",
                 log=True)
        plt.savefig(filepath_result)
        plt.close()

    # --------------------------------------------------

    def on_epoch_begin(self, epoch, logs={}):
        self._epoch += 1

# ==============================================================================
