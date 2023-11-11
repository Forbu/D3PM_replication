"""
helper function for generating diffusion data
"""

import torch
import numpy as np

def generate_beta_t(array_t, beta_0=0.001, beta_T=0.01):
    """
    Returns the beta_t values for each time step.
    It's a linear interpolation from beta_0 to beta_T.
    """
    # get the number of time steps
    nb_steps = array_t.shape[0]

    # create the beta_t array
    beta_t = np.zeros(nb_steps)

    # compute the beta_t values
    for t in range(nb_steps):
        beta_t[t] = beta_0 + (beta_T - beta_0) * (t / (nb_steps - 1))

    return beta_t
    


def compute_transition_matrices(beta_t, array_t, num_bins=4):
    """
    Computes the transition matrices for each time step.
    """
    # get the number of time steps
    nb_steps = array_t.shape[0]

    # create the transition matrices
    transition_matrices = np.zeros((nb_steps, num_bins, num_bins))

    # loop through each time step
    for t in range(nb_steps):
        # get the beta for this time step
        beta = beta_t[t]

        # compute the transition matrix
        transition_matrix = np.zeros((num_bins, num_bins))
        for i in range(num_bins):
            for j in range(num_bins):
                if i == j:
                    transition_matrix[i, j] = 1 - beta
                else:
                    transition_matrix[i, j] = beta / (num_bins - 1)

        # add the transition matrix to the list
        transition_matrices[t] = transition_matrix

    return transition_matrices


def compute_accumulated_transition_matrices(transition_matrices):
    """
    Computes the accumulated transition matrices.
    """
    # get the number of time steps
    nb_steps = transition_matrices.shape[0]

    # create the accumulated transition matrices
    accumulated_transition_matrices = np.zeros(
        (nb_steps, transition_matrices.shape[1], transition_matrices.shape[2])
    )

    accumulated_transition_matrices[0] = transition_matrices[0]

    # compute the accumulated transition matrices
    for t in range(1, nb_steps):
        accumulated_transition_matrices[t] = np.matmul(
            accumulated_transition_matrices[t - 1], transition_matrices[t]
        )

    return accumulated_transition_matrices
