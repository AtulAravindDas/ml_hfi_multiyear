"""Metrics for training and evaluation.

Classes:
    SMSELoss: Square root of the mean squared error loss.
    MSELoss: Mean squared error loss.

"""

import torch
import numpy as np
from torch.nn import functional as F


class WeightedSMSELoss(torch.nn.Module):
    """
    Square root of the mean squared error loss with optional
    zero-weighting and one-weighting.
    """

    def __init__(
        self,
        zero_weighting=None,
        zero_threshold=0.0,
        one_weighting=None,
        one_threshold=0.90,
        kluge_value_for_zero=None,
    ):
        super().__init__()
        self.zero_weighting = zero_weighting
        self.one_weighting = one_weighting
        self.zero_threshold = zero_threshold
        self.one_threshold = one_threshold
        self.kluge_value_for_zero = kluge_value_for_zero
        self.non_weighting = 1.0

    def forward(self, output, target):

        if self.kluge_value_for_zero is not None:
            kluge_addon = torch.where(target == 0, self.kluge_value_for_zero, 0.0)
            error = output - target + kluge_addon
        else:
            error = output - target

        if self.zero_weighting is not None and self.one_weighting is None:
            weights = torch.where(
                torch.logical_or(
                    target <= self.zero_threshold, output <= self.zero_threshold
                ),
                self.zero_weighting,
                self.non_weighting,
            )
            loss = torch.mean(weights * torch.square(error))

        elif self.zero_weighting is None and self.one_weighting is not None:
            weights = torch.where(
                torch.logical_or(
                    target >= self.one_threshold, output >= self.one_threshold
                ),
                self.one_weighting,
                self.non_weighting
            )
            loss = torch.mean(weights * torch.square(error))

        elif self.zero_weighting is not None and self.one_weighting is not None:
            weights = torch.where(
                torch.logical_or(
                    target <= self.zero_threshold, output <= self.zero_threshold
                ),
                self.zero_weighting,
                self.non_weighting
            )
            weights = torch.where(
                torch.logical_or(
                    target >= self.one_threshold, output >= self.one_threshold
                ),
                self.one_weighting,
                weights
            )
            loss = torch.mean(weights * torch.square(error))

        else:
            loss = torch.mean(torch.square(error))

        # if self.zero_weighting is not None and self.one_weighting is None:
        #     weights = torch.where(
        #         target <= self.zero_threshold, self.zero_weighting, self.non_weighting
        #     )
        #     loss = torch.mean(weights * torch.square(error))

        # elif self.zero_weighting is None and self.one_weighting is not None:
        #     weights = torch.where(
        #         target > self.one_threshold, self.one_weighting, self.non_weighting
        #     )
        #     loss = torch.mean(weights * torch.square(error))

        # elif self.zero_weighting is not None and self.one_weighting is not None:
        #     weights = torch.where(
        #         target <= self.zero_threshold, self.zero_weighting, self.non_weighting
        #     )
        #     weights = torch.where(
        #         target > self.one_threshold, self.one_weighting, weights
        #     )
        #     loss = torch.mean(weights * torch.square(error))

        # else:
        #     loss = torch.mean(torch.square(error))

        return torch.sqrt(loss)
