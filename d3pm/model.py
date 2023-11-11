"""
Module with the core training loop for the diffusion models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import segmentation_models_pytorch as smp


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

        # last layer (linear one to get the logits)
        self.last_layer = nn.Linear(hidden_size, num_bins)

        # create the blocks
        self.blocks = smp.Unet(
            encoder_name="resnet34",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights=None,  # use `imagenet` pre-trained weights for encoder initialization
            in_channels=hidden_size,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=num_bins,  # model output channels (number of classes in your dataset)
        )

    def forward(self, data, t):
        """
        Args:
            data (torch.Tensor): data to be discretized (dim = (batch_size, 28, 28)
            t (int): time step (dim = (batch_size, 1))
        """
        # get the embedding
        x = self.embedding(data)  # (batch_size, 28, 28, hidden_size)

        if len(t.shape) == 1:
            t = t.unsqueeze(1)

        # get the time embedding
        t = self.time_layer(t)  # (batch_size, hidden_size)

        # add the time embedding to the data
        x = x + t.unsqueeze(1).unsqueeze(1)  # (batch_size, 28, 28, hidden_size)

        # put in the right shape
        x = x.permute(0, 3, 1, 2)

        # loop through the blocks
        x = self.blocks(x)
        
        x = x.permute(0, 2, 3, 1)

        return x
