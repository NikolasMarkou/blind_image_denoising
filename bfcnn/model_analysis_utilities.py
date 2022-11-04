import io
import matplotlib
import numpy as np
from typing import Dict
import tensorflow as tf
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------

DEFAULT_DPI = 100

# ---------------------------------------------------------------------


def draw_figure_to_buffer(
        fig: matplotlib.figure.Figure,
        dpi: int = DEFAULT_DPI) -> np.ndarray:
    """
    draw figure into numpy buffer

    :param fig: figure to draw
    :param dpi: dots per inch to draw
    :return: buffer
    """

    # open binary buffer
    with io.BytesIO() as io_buf:
        # save figure and reset to start
        fig.savefig(io_buf, format="raw", dpi=dpi)
        io_buf.seek(0)
        # load buffer into numpy and reshape
        img_arr = \
            np.reshape(
                a=np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
    return img_arr

# ---------------------------------------------------------------------


def visualize_channels(model, img_tensor):
    layer_outputs = [layer.output for layer in model.layers[:12]]  # Extracts the outputs of the top 12 layers
    # Creates a model that will return these outputs, given the model input
    activation_model = tf.keras.Model(inputs=model.input, outputs=layer_outputs)
    # Returns a list of five Numpy arrays: one array per layer activation
    activations = activation_model.predict(img_tensor)
    layer_names = []
    for layer in model.layers[:12]:
        layer_names.append(layer.name)  # Names of the layers, so you can have them as part of your plot

    images_per_row = 16
    for layer_name, layer_activation in zip(layer_names, activations):  # Displays the feature maps
        n_features = layer_activation.shape[-1]  # Number of features in the feature map
        size = layer_activation.shape[1]  # The feature map has shape (1, size, size, n_features).
        n_cols = n_features // images_per_row  # Tiles the activation channels in this matrix
        display_grid = np.zeros((size * n_cols, images_per_row * size))
        for col in range(n_cols):  # Tiles each filter into a big horizontal grid
            for row in range(images_per_row):
                channel_image = layer_activation[0, :, :, col * images_per_row + row]
                channel_image -= channel_image.mean()  # Post-processes the feature to make it visually palatable
                channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                display_grid[col * size: (col + 1) * size,  # Displays the grid
                row * size: (row + 1) * size] = channel_image
        scale = 1. / size
        plt.figure(figsize=(scale * display_grid.shape[1],
                            scale * display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')

# ---------------------------------------------------------------------


def compute_layer_distance_from_top(
        model: tf.keras.Model) -> Dict[str, int]:
    # --- argument checking
    if model is None:
        raise ValueError("model cannot be None")

    # --- set variables
    distances = {}

    # --- go through each layer
    for layer in model.layers:
        layer_config = layer.get_config()
        if "layers" not in layer_config:
            continue
        for layer_internal in layer.layers:
            pass
    return distances

# ---------------------------------------------------------------------


def compute_weight_distance_from_top(
        model: tf.keras.Model) -> Dict[str, int]:
    for w in model.trainable_weights:
        pass
    return {}

# ---------------------------------------------------------------------

