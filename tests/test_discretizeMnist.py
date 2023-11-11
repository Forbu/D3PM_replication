"""
Tests for discretizeMnist.py
"""

import pytest

import numpy as np
import matplotlib.pyplot as plt

from d3pm.dataset_discretize_mnist import DiscretizeMnist, DiscretizeD3PMMNIST


def test_class():
    dataset = DiscretizeMnist(num_bins=4)

    print(dataset[0])


def test_discretized_mnist():
    dataset = DiscretizeD3PMMNIST(num_bins=4, nb_steps=254)

    # we want to plot the first 0, 50, 100, 150, 200, 250 images
    idxs = [0, 50, 100, 150, 200, 250]

    for idx in idxs:
        data_t, data_next, label, data, idx_step = dataset[idx]

        # plot the data
        plt.imshow(data, cmap="gray")

        # title
        plt.title(f"data = {idx}")

        # save the figure
        plt.savefig(f"tests/data.png")

        # close the figure
        plt.close()

        # plot the data_t
        plt.imshow(data_t, cmap="gray")

        # title
        plt.title(f"data_t = {idx}")

        # save the figure
        plt.savefig(f"tests/figures/data_t_{idx}.png")

        # close the figure

        # plot the data_next
        plt.imshow(data_next, cmap="gray")

        # title
        plt.title(f"data_next = {idx}")

        # save the figure
        plt.savefig(f"tests/figures/data_next_{idx}.png")

        # close the figure
        plt.close()

        ## now we plot the difference between data_next and data_t
        # plot the data_next
        plt.imshow(data_next - data_t, cmap="gray")

        # title
        plt.title(f"data_next - data_t = {idx}")
        
        # save the figure
        plt.savefig(f"tests/figures/data_next_minus_data_t_{idx}.png")

        # close the figure
        plt.close()

    exit()
