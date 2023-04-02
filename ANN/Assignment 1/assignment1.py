import numpy as np
import matplotlib.pyplot as plt
import os
import torch.optim
from torch import nn
from torch.utils.data import Dataset, DataLoader
from typing import Callable



# Get appropriate directories
master_dir = os.getcwd()


class Trainer:

    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, train_data: tuple[np.array, np.array],
                 test_data: tuple[np.array, np.array], loss_fn: Callable = nn.MSELoss()):

        # Save parameters as fields
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn

        # Save data, convert from Numpy array to Torch tensor
        self.train_x = torch.from_numpy(train_data[0]).float()
        self.train_y = torch.from_numpy(train_data[1]).float()
        self.test_x  = torch.from_numpy(test_data[0]).float()
        self.test_y  = torch.from_numpy(test_data[1]).float()

        # Initialize empty lists for training losses
        self.training_losses = []
        self.test_losses = []

    def train(self, nb_of_epochs: int = 100, verbose=True):

        for i in range(nb_of_epochs):
            # Zero the gradients
            self.optimizer.zero_grad()

            # Compute the output of the network
            train_outputs = self.model(self.train_x)
            test_outputs = self.model(self.test_x)

            # Compute the loss
            train_loss = self.loss_fn(train_outputs, self.train_y)
            test_loss = self.loss_fn(test_outputs, self.train_y)

            # Compute the gradients
            train_loss.backward()

            # Update the weights
            self.optimizer.step()

            if verbose:
                if (i + 1) % 5 == 0:
                    print('Epoch [{}/{}], Loss: {:.4f}'.format(i + 1, nb_of_epochs, train_loss.item()))

