"""Metrics for training and evaluation.

Classes
---------
None

Functions
---------
custom_mae(output, target):
    Compute the prediction mean absolute error.

"""

import torch
import numpy as np


def custom_mae(output, target):
    """Compute the prediction mean absolute error.
    The "predicted value" is the median of the conditional distribution.

    """
    with torch.no_grad():

        assert len(output[:, 0]) == len(target)

        return torch.mean(torch.abs(output - target)).item()
