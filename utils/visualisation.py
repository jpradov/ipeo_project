"""This module contains all function related to visualisation of images and results. """

import matplotlib.pyplot as plt
import seaborn as sns

from utils.data import normalized_image


def visualise_image(image, label):
    """Function to show true color, false color and segmentation mask of an image."""

    normalized = normalized_image(image)

    fig, axes = plt.subplots(
        nrows=1, ncols=3, sharex=True, sharey=True, figsize=(18, 8)
    )
    axes[0].imshow(normalized[:, :, [0, 1, 2]])
    axes[0].set_title("True Color Image")
    axes[0].axis("off")

    axes[1].imshow(normalized[:, :, [3, 0, 1]])
    axes[1].set_title("False Color Image (NINFR, RED, GREEN)")
    axes[1].axis("off")

    axes[2].imshow(label)
    axes[2].set_title("Segmentation Mask")
    axes[2].axis("off")

    return


def visualise_image_3_channels(image, label):
    """Function to show true color, false color and segmentation mask of an image."""

    normalized = normalized_image(image)

    fig, axes = plt.subplots(
        nrows=1, ncols=2, sharex=True, sharey=True, figsize=(18, 8)
    )
    axes[0].imshow(normalized[:, :, [0, 1, 2]])
    axes[0].set_title("True Color Image")
    axes[0].axis("off")

    axes[1].imshow(label)
    axes[1].set_title("Segmentation Mask")
    axes[1].axis("off")

    return


def plot_band_distribution(image):
    fig, axes = plt.subplots(1, 4, sharex=False, sharey=True, figsize=(14, 4))
    for k in range(image.shape[2]):
        band = image[:, :, k].flatten()
        sns.histplot(band, ax=axes[k], kde=True)
        axes[k].set_title("Band {}".format(k + 1))
