import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
from keras.callbacks import Callback


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
                 run_folder,
                 print_every_n_batches,
                 initial_epoch,
                 images,
                 model,
                 resize_shape=(256, 256)):
        """
        Callback for saving the intermediate result image
        :param run_folder:
        :param print_every_n_batches:
        :param initial_epoch:
        :param images:
        :param model:
        :param resize_shape:
        """
        self.model = model
        self.images = images
        self.epoch = initial_epoch
        self.run_folder = run_folder
        self._resize_shape = resize_shape
        self.print_every_n_batches = print_every_n_batches
        images_path = os.path.join(self.run_folder, "images")
        if not os.path.exists(images_path):
            os.mkdir(images_path)

    def on_batch_end(self, batch, logs={}):
        if batch % self.print_every_n_batches == 0:
            predictions = self.vae.model_trainable.predict(self.images)
            predictions = self.vae.normalize(predictions)
            predictions = np.clip(predictions, a_min=0.0, a_max=1.0)
            # --- create collage of the predictions
            x = collage(predictions)
            # --- resize to output size
            x = resize(x, self._resize_shape, order=0)
            filepath_x = os.path.join(
                self.run_folder,
                "images",
                "img_" + str(self.epoch).zfill(3) +
                "_" + str(batch) + ".png")
            if len(x.shape) == 2:
                plt.imsave(filepath_x, x, cmap="gray_r")
            else:
                plt.imsave(filepath_x, x)

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch += 1

# ==============================================================================
