"""Metrics for training and evaluation.

Functions
---------
custom_mae(output, target)
iqr_capture(output, target)
sign_test(output, target)
pit(output, target)

"""

import torch
import numpy as np
from torch.nn import functional as F


class SMSELoss(torch.nn.Module):
    """
    Square root of the mean squared error loss.
    """

    def __init__(self):
        super().__init__()

    def forward(self, output, target):

        loss = F.mse_loss(output, target)
        return torch.sqrt(loss).mean()


class MSELoss(torch.nn.Module):
    """
    Square root of the mean squared error loss.
    """

    def __init__(self):
        super().__init__()

    def forward(self, output, target):

        loss = F.mse_loss(output, target)
        return loss.mean()