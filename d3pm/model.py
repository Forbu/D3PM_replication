"""
Module with the core training loop for the diffusion models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MnistModel(nn.Module):
    """
    This is a simple model for discretized MNIST.
    It basicly takes a 28x28 discretize image and outputs another 28x28xnum_bins image.
    It also takes a time step in input
    """

    def __init__(self, num_bins=4, hidden_size=10, nb_block=3):
        """
        Args:
            num_bins (int): number of bins to discretize the data into
        """
        super().__init__()

        self.num_bins = num_bins
        self.hidden_size = hidden_size
        self.nb_block = nb_block

        # embedding layer
        self.embedding = nn.Embedding(num_bins, hidden_size)

        # time layer
        self.time_layer = nn.Linear(1, hidden_size)

        def create_block_conv():
            return nn.Sequential(
                nn.Conv2d(
                    in_channels=hidden_size,
                    out_channels=hidden_size,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(hidden_size),
                nn.ReLU(),
            )

        # last layer (linear one to get the logits)
        self.last_layer = nn.Linear(hidden_size, num_bins)

        # create the blocks
        self.blocks = nn.ModuleList([create_block_conv() for _ in range(nb_block)])

    def forward(self, data, t):
        """
        Args:
            data (torch.Tensor): data to be discretized (dim = (batch_size, 28, 28)
            t (int): time step (dim = (batch_size, 1))
        """
        # get the embedding
        x = self.embedding(data) # (batch_size, 28, 28, hidden_size)

        if len(t.shape) == 1:
            t = t.unsqueeze(1)

        # get the time embedding
        t = self.time_layer(t) # (batch_size, hidden_size)

        # add the time embedding to the data
        x = x + t.unsqueeze(1).unsqueeze(1) # (batch_size, 28, 28, hidden_size)

        # put in the right shape
        x = x.permute(0, 3, 1, 2)

        # loop through the blocks
        for block in self.blocks:
            # apply the block
            x = block(x)

        # put in the right shape
        x = x.permute(0, 2, 3, 1)

        # apply the last layer
        x = self.last_layer(x)

        return x
