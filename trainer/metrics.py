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
        return torch.mean(torch.abs(output - target)).item()


def count_zeros(output, target):
    """Count the number of zeros in the output.

    """
    return torch.count_nonzero(output < 0.01).item()


def missed_zeros(output, target):
    """Count the number of zeros in the target that were missed.

    """
    return (
        torch.sum((output >= 0.01) & (target < 0.01))
        / (torch.count_nonzero(target < 0.01))
    ).item()
