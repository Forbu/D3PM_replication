"""
test module for the model deeplearning.ai
"""
import os
import pytest
import torch
from d3pm.model_deeplearningai import ContextUnet

def test_model_deeplearninai():

    nb_bins = 4
    nb_steps = 254

    # two input in themodel : x and t
    # x : (batch, n_feat, h, w) : input image
    # t : (batch, 1)      : time step
    x = torch.randint(0, nb_bins, (10, 1, 28, 28))
    t = torch.randn(10, 1)

    # create the model
    model = ContextUnet(in_channels=1, n_feat=256, n_cfeat=10, height=28, nb_class=nb_bins)

    # forward pass
    y = model(x, t)

    # check the shape
    assert y.shape == (10, nb_bins, 28, 28)
