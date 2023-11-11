"""
Test functions for diffusion_generation.py

"""

import pytest

import numpy as np

from d3pm.diffusion_generation import (
    generate_beta_t,
    compute_transition_matrices,
    compute_accumulated_transition_matrices,
)


def test_generate_beta_t():
    # get the beta_t values
    beta_t = generate_beta_t(np.arange(10))

    # check the values
    assert beta_t.shape == (10,)


def test_compute_transition_matrices():
    t_array = np.arange(254)
    beta_t = generate_beta_t(t_array)

    num_bins = 4

    transition_matrices = compute_transition_matrices(beta_t, t_array, num_bins)

    # check the shape
    assert transition_matrices.shape == (254, num_bins, num_bins)


def test_compute_accumulated_transition_matrices():
    t_array = np.arange(254)
    beta_t = generate_beta_t(t_array)

    transition_matrices = compute_transition_matrices(beta_t, t_array)

    cumulated_transition = compute_accumulated_transition_matrices(transition_matrices)

    print(cumulated_transition[100, :, :])

    exit()
