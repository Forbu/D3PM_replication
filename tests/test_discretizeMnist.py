"""
Tests for discretizeMnist.py
"""

import pytest

from d3pm.dataset_discretize_mnist import DiscretizeMnist

def test_class():

    dataset = DiscretizeMnist(num_bins=4)

    print(dataset[0])
    exit()
