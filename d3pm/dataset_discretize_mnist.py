"""
Module for discretizing the MNIST dataset.
"""


import torch
import torchvision
import torchvision.transforms as transforms

# import dataset class from torch
from torch.utils.data import Dataset

import numpy as np

from d3pm.diffusion_generation import (
    compute_transition_matrices,
    compute_accumulated_transition_matrices,
    generate_beta_t,
)


class DiscretizeMnist(Dataset):
    """
    Simple dataset that discretizes the MNIST dataset.
    """

    def __init__(self, num_bins=10):
        """
        Args:
            data (torch.Tensor): data to be discretized
            labels (torch.Tensor): labels for the data
            num_bins (int): number of bins to discretize the data into
        """
        # get the data and labels
        self.num_bins = num_bins

        # import classic mnist dataset
        self.mnist = torchvision.datasets.MNIST(
            root="./data", train=True, download=True
        )

        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Resize(224)]
        )

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        """
        Returns a tuple of (discretized data, label)
        """
        # get the data and label
        data, label = self.mnist[idx]

        # transform the data
        data = self.transform(data)

        # discretize the data
        data = self.discretize(data)

        return data, label

    def discretize(self, data):
        """
        Discretizes the data into self.num_bins bins.
        """
        # get the min and max of the data
        min_val = torch.min(data)
        max_val = torch.max(data)

        # get the bin size
        bin_size = (max_val - min_val) / self.num_bins

        # discretize the data
        data = torch.floor((data - min_val) / bin_size)

        # max being num_bins - 1
        data = torch.clamp(data, 0, self.num_bins - 1)

        return data


# now we create the diffusion dataset
class DiscretizeD3PMMNIST(Dataset):
    def __init__(self, num_bins=4, nb_steps=254):
        self.num_bins = num_bins
        self.dataset = DiscretizeMnist(num_bins=num_bins)
        self.nb_steps = nb_steps

        # get the beta_t values
        array_t = np.arange(nb_steps)
        beta_t = generate_beta_t(array_t)

        # compute the transition matrices
        self.transition_matrices = compute_transition_matrices(
            beta_t, array_t, num_bins
        )

        # compute the accumulated transition matrices
        self.cumulated_transition = compute_accumulated_transition_matrices(
            self.transition_matrices
        )

    def __len__(self):
        return len(self.dataset) * self.nb_steps

    def __getitem__(self, idx):
        idx_data = idx // self.nb_steps
        idx_step = idx % self.nb_steps

        tmp_transition = torch.tensor(self.cumulated_transition[idx_step, :, :])
        tmp_transition_next = torch.tensor(
            self.transition_matrices[min(idx_step + 1, self.nb_steps - 1), :, :]
        )

        data, label = self.dataset[idx_data]

        size = data.squeeze().shape

        # in the data tensor, we have only one channel with num_bins values
        # we need to apply the transition matrix to each pixel
        data = data.squeeze()  # 100x100
        data_flatten = data.view(-1).long()  # 10000

        # apply the transition matrix
        data_bernouilli_proba = tmp_transition[data_flatten]

        data_sample_t = torch.distributions.categorical.Categorical(
            data_bernouilli_proba
        ).sample()

        # now we want to create the image at time t+1
        data_bernouilli_proba_next = tmp_transition_next[data_sample_t.long()]

        data_sample_t_next = torch.distributions.categorical.Categorical(
            data_bernouilli_proba_next
        ).sample()

        data_sample_t = data_sample_t.view(size[0], size[0])
        data_sample_t_next = data_sample_t_next.view(size[0], size[0])

        return data_sample_t, data_sample_t_next, label, data, (idx_step+1) / (self.nb_steps-1)
