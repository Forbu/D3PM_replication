"""
Module to handle the training of the model.
"""
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

import einops

from tqdm import tqdm

import lightning.pytorch as pl

from d3pm.model import MnistModel


class MnistTrainer(pl.LightningModule):
    """
    Trainer module for the MNIST model.
    """

    def __init__(self, hidden_dim, num_bins, nb_time_steps=254):
        """
        Args:
            hidden_dim (int): hidden dimension of the model
            num_bins (int): number of bins to discretize the data into
            nb_block (int): number of blocks in the model
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_bins = num_bins

        self.nb_time_steps = nb_time_steps

        # create the model
        self.model = MnistModel(
            hidden_size=hidden_dim, num_bins=num_bins
        )

        # create the loss function
        self.loss = nn.CrossEntropyLoss()

    def forward(self, data, t):
        """
        Forward pass of the model.
        """
        # get the logits
        logits = self.model(data, t)

        # change the shape
        logits = logits.permute(0, 3, 1, 2)

        return logits

    def compute_loss(self, logits, data, init_data):
        """
        Computes the loss.
        """
        # compute the loss
        loss_vb = self.loss(logits, data)
        loss_init = 0.0 * self.loss(logits, init_data)

        loss = loss_vb + loss_init

        return loss, (loss_vb, loss_init)

    def training_step(self, batch, _):
        """
        Training step.
        """
        # get the data and label
        data, data_next, _, init_data, time_step = batch

        # get the logits
        logits = self.forward(data_next.long(), time_step.float())

        # compute the loss
        loss, (loss_vb, loss_init) = self.compute_loss(
            logits, data.long(), init_data.long()
        )

        # log the loss
        self.log("train_loss", loss)
        self.log("train_loss_vb", loss_vb)
        self.log("train_loss_init", loss_init)

        return loss

    # on training end
    def on_train_epoch_end(self):
        # we should generate some images
        self.eval()
        with torch.no_grad():
            self.generate()

    def generate(self):
        """
        Method to generate some images.
        """
        # generate some images
        device = self.device

        # initialize the data with random values between 0 and num_bins
        data = torch.randint(0, self.num_bins, (1, 28, 28)).long().to(device)

        # initialize the time step
        time_step = torch.tensor([[1.0]]).to(device)

        # plot time step
        plot_index = [0, 1, 50, 100, 150, 200, 240]

        for i in range(self.nb_time_steps):
            if i in plot_index:
                # save the image
                self.save_image(data, i)

            # get the logits

            logits = self.forward(data, time_step)

            # get the probabilities
            logits_flatten = einops.rearrange(logits, "a b c d -> (a c d) b")

            proba = F.softmax(logits_flatten, dim=1)

            # sample from the probabilities
            data = torch.distributions.Categorical(probs=proba).sample()

            data = einops.rearrange(
                data,
                "(a c d) -> a c d",
                a=logits.shape[0],
                c=logits.shape[2],
                d=logits.shape[3],
            )

            # update the time step
            time_step = time_step - 1.0 / self.nb_time_steps

            self.save_image(data, 254)

    def save_image(self, data, i):
        """
        Saves the image.
        """
        # plot the data
        plt.imshow(data.squeeze().cpu().numpy(), cmap="gray")

        # title
        plt.title(f"data = {i}")

        # save the figure
        plt.savefig(f"/content/images/data_{i}.png")

        # close the figure
        plt.close()

    def configure_optimizers(self):
        """
        Configure the optimizer.
        """
        # create the optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

        return optimizer
