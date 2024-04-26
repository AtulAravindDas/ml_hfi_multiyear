"""Metrics for training and evaluation.

Classes:
    SMSELoss: Square root of the mean squared error loss.
    MSELoss: Mean squared error loss.

"""

import torch
import numpy as np
from torch.nn import functional as F


class SMSELoss(torch.nn.Module):
    """
    Square root of the mean squared error loss with optional
    zero-weighting.
    """

    def __init__(self, zero_weighting=None):
        super().__init__()
        self.zero_weighting = zero_weighting
        self.nonzero_weighting = 1.0

    def forward(self, output, target):
        if self.zero_weighting is not None:
            weights = torch.where(
                target > 0, self.nonzero_weighting, self.zero_weighting
            )
            loss = torch.mean(weights * torch.square(output - target))
        else:
            loss = torch.mean(torch.square(output - target))

        return torch.sqrt(loss)


class MSELoss(torch.nn.Module):
    """
    Square root of the mean squared error loss.
    """

    def __init__(self):
        super().__init__()

    def forward(self, output, target):

        return F.mse_loss(output, target)
