"""
Script to train a model on MNIST on discretized data with a diffusion model (D3PM).
"""


import os
import sys

import argparse

import lightning.pytorch as pl
import torch

# import the deepgraphgen modules that are just above
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from d3pm.dataset_discretize_mnist import DiscretizeD3PMMNIST
from d3pm.pl_trainer import MnistTrainer

# first we define the dataset
dataset = DiscretizeD3PMMNIST(num_bins=4, nb_steps=254)

# now we create the loader
loader = torch.utils.data.DataLoader(
    dataset, batch_size=32, num_workers=8, shuffle=True
)

# now we create the model
model = MnistTrainer(hidden_dim=16, num_bins=4, nb_block=3, nb_time_steps=254)

# now we create the trainer
trainer = pl.Trainer(
    max_epochs=10,
    limit_train_batches=0.01,
    gradient_clip_val=1.0,
)

# now we train the model
trainer.fit(model, loader)
