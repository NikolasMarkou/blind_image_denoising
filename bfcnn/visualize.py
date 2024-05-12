r"""Tensorboard samples visualization"""

import io
import matplotlib
import numpy as np
import tensorflow as tf
from sklearn import metrics
from typing import Union, List, Tuple

# ---------------------------------------------------------------------

"""
You may be able to work on separate figures from separate threads. 
However, you must in that case use a non-interactive backend (typically Agg), 
because most GUI backends require being run from the main thread as well.

from https://matplotlib.org/stable/users/faq/howto_faq.html

Do not remove this otherwise you will get XIO:  fatal IO error 25 (Inappropriate ioctl for device) on X server 
"localhost:10.0"                                                                                                      
│························ after 1942 requests (1942 known processed) with 7 events remaining.     
························ File "tools/train.py", line 142, in <module> 

when you detach the terminal
"""

matplotlib.use('Agg')

import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .constants import *

# ---------------------------------------------------------------------


def collage(
        images_batch):
    """
    Create a collage of image from a batch

    :param images_batch:
    :return:
    """
    shape = images_batch.shape
    no_images = shape[0]
    images = []
    result = None
    width = np.ceil(np.sqrt(no_images))

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


# ---------------------------------------------------------------------


def draw_figure_to_buffer(
        fig: matplotlib.figure.Figure,
        dpi: int) -> np.ndarray:
    """
    draw figure into numpy buffer

    :param fig: figure to draw
    :param dpi: dots per inch to draw
    :return: np.ndarray buffer
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


def visualize_gradient_boxplot(
        gradients: List,
        trainable_variables: List,
        y_limits: Tuple[float, float] = (-0.1, +0.1),
        figsize: Tuple[int, int] = DEFAULT_NON_SYMMETRIC_FIGSIZE,
        dpi: int = DEFAULT_DPI,
        expand_dims: bool = True) -> np.ndarray:
    """
    visualize the gradients as a series of boxplots

    :param gradients:
    :param trainable_variables:
    :param y_limits: limits to the gradient values
    :param figsize: figure size
    :param dpi: dots per inch to draw
    :param expand_dims: if true add extra dim to make it tensorboard compatible

    :return: np.ndarray with image
    """
    # --- argument check
    if len(gradients) <= 0:
        raise ValueError("gradients should be a list with >= 1 elements")

    if len(gradients) != len(trainable_variables):
        raise ValueError("gradients_list length and trainable_weights length must be equal")

    # --- extract variables of interest
    y_limits = [y_limits[0], y_limits[1]]
    variable_names = [
        str(w.name).split("/")[0] for w in trainable_variables
    ]
    gradients_flat = [
        np.clip(g.numpy().flatten(), a_min=y_limits[0], a_max=y_limits[1])
        for g in gradients
    ]

    # --- plot bars
    plt.ioff()
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111)

    ax.boxplot(
        x=gradients_flat,
        labels=variable_names,
        widths=0.5,
        showfliers=False,
        vert=True)

    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=30,
        fontsize=6,
        ha="right")
    ax.grid(True)
    ax.set_ylim(y_limits)
    ax.set_ylabel("gradient distribution")
    fig.tight_layout()

    fig_buffer = draw_figure_to_buffer(fig=fig, dpi=dpi)
    plt.close(fig)
    plt.ion()

    if expand_dims:
        # add a batch dim to make it tensorboard compatible
        fig_buffer = np.expand_dims(fig_buffer, axis=0)

    return fig_buffer

# ---------------------------------------------------------------------


def visualize_weights_boxplot(
        trainable_variables: List,
        y_limits: Tuple[float, float] = (-0.2, +0.2),
        figsize: Tuple[int, int] = DEFAULT_NON_SYMMETRIC_FIGSIZE,
        dpi: int = DEFAULT_DPI,
        expand_dims: bool = True) -> np.ndarray:
    """
    visualize the weights as a series of boxplots

    :param trainable_variables:
    :param y_limits: limits to weight values
    :param figsize: figure size
    :param dpi: dots per inch to draw
    :param expand_dims: if true add extra dim to make it tensorboard compatible

    :return: np.ndarray with image
    """
    # --- argument check
    if len(trainable_variables) <= 0:
        raise ValueError("trainable_variables should be a list with >= 1 elements")

    # --- extract variables of interest
    y_limits = [y_limits[0], y_limits[1]]
    trainable_variables = [
        w
        for w in trainable_variables
        if "batch_normalization" not in w.name.lower()
    ]
    variable_names = [
        str(w.name).split("/")[0] for w in trainable_variables
    ]
    weights_flat = [
        np.clip(w.numpy().flatten(), a_min=y_limits[0], a_max=y_limits[1])
        for w in trainable_variables
    ]

    # --- plot bars
    plt.ioff()
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111)

    ax.boxplot(
        x=weights_flat,
        labels=variable_names,
        widths=0.5,
        showfliers=False,
        vert=True)

    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=30,
        fontsize=6,
        ha="right")
    ax.grid(True)
    ax.set_ylim(y_limits)
    ax.set_ylabel("weight distribution")
    fig.tight_layout()
    fig_buffer = draw_figure_to_buffer(fig=fig, dpi=dpi)
    plt.close(fig)
    plt.ion()

    if expand_dims:
        # add a batch dim to make it tensorboard compatible
        fig_buffer = np.expand_dims(fig_buffer, axis=0)

    return fig_buffer

# ---------------------------------------------------------------------


def visualize_weights_heatmap(
        trainable_variables: List,
        no_bins: int = 21,
        y_limits: Tuple[float, float] = (-0.2, +0.2),
        figsize: Tuple[int, int] = DEFAULT_NON_SYMMETRIC_FIGSIZE,
        dpi: int = DEFAULT_DPI,
        expand_dims: bool = True) -> np.ndarray:
    """
    visualize the weights as a series of heatmaps

    :param trainable_variables:
    :param no_bins: number of bins
    :param y_limits: limits to weight values
    :param figsize: figure size
    :param dpi: dots per inch to draw
    :param expand_dims: if true add extra dim to make it tensorboard compatible

    :return: np.ndarray with image
    """
    # --- argument check
    if len(trainable_variables) <= 0:
        raise ValueError("trainable_variables should be a list with >= 1 elements")
    bins = no_bins
    fontsize = 6

    # --- extract variables of interest (keep out batch norms)
    y_limits = [y_limits[0], y_limits[1]]
    trainable_variables = [
        w
        for w in trainable_variables
        if "batch_normalization" not in w.name.lower() and
           "layer_normalization" not in w.name.lower()
    ]
    variable_names = [
        str(w.name).split("/")[0] for w in trainable_variables
    ]
    weights_flat = np.zeros((len(trainable_variables), bins))
    for i, w in enumerate(trainable_variables):
        weights_flat[i, :] =\
            np.histogram(np.clip(w.numpy().flatten(), a_min=y_limits[0], a_max=y_limits[1]),
                         bins=bins,
                         density=True,
                         range=y_limits)[0]
    weights_flat = np.transpose(weights_flat)
    bins_space = [
        "{:.2f}".format(b)
        for b in
        np.linspace(
            start=y_limits[0],
            stop=y_limits[1],
            num=no_bins,
            endpoint=True)
    ]

    # --- plot bars
    plt.ioff()
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111)
    _ = \
        ax.imshow(
            weights_flat,
            cmap="viridis",
            interpolation="nearest")
    ax.set_xticks(
        np.arange(len(variable_names)),
        labels=variable_names)
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=30,
        fontsize=fontsize,
        ha="right")

    ax.set_yticks(
        np.arange(no_bins),
        labels=bins_space
    )
    ax.set_yticklabels(
        ax.get_yticklabels(),
        fontsize=fontsize)
    ax.set_title("weights distribution per layer")
    fig.tight_layout()
    fig_buffer = draw_figure_to_buffer(fig=fig, dpi=dpi)
    plt.close(fig)
    plt.ion()

    if expand_dims:
        # add a batch dim to make it tensorboard compatible
        fig_buffer = np.expand_dims(fig_buffer, axis=0)

    return fig_buffer

# ---------------------------------------------------------------------

