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
    zero-weighting and one-weighting.
    """

    def __init__(self, zero_weighting=None, one_weighting=None, one_threshold=0.90):
        super().__init__()
        self.zero_weighting = zero_weighting
        self.one_weighting = one_weighting
        self.one_threshold = one_threshold
        self.non_weighting = 1.0

    def forward(self, output, target):
        if self.zero_weighting is not None and self.one_weighting is None:
            weights = torch.where(target == 0, self.zero_weighting, self.non_weighting)
            loss = torch.mean(weights * torch.square(output - target))

        elif self.zero_weighting is None and self.one_weighting is not None:
            weights = torch.where(
                target > self.one_threshold, self.one_weighting, self.non_weighting
            )
            loss = torch.mean(weights * torch.square(output - target))

        elif self.zero_weighting is not None and self.one_weighting is not None:
            weights = torch.where(target == 0, self.zero_weighting, self.non_weighting)
            weights = torch.where(
                target > self.one_threshold, self.one_weighting, weights
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
