"""
Merged model for boosting base model predictions.

Functions
---------

Classes
---------
BoostModel(base.base_model.BaseModel)
    A class representing a torch-based model that combines a baseline
    model and an updater/booster model.

"""

import torch
from base.base_model import BaseModel
from base.base_model import dense_block


class BoostModel(BaseModel):
    """
    TorchModel class represents a torch-based model for a specific task.

    Args:
        config (dict): Configuration parameters for the model.

    Attributes:
        config (dict): Configuration parameters for the model.

    Methods:
        forward(input): Performs forward pass of the model.

    """

    def __init__(self, config, baseline_model, update_model):
        """
        Initializes the TorchModel.

        Args:
            config (dict): Configuration parameters for the model.

        """
        super().__init__()

        self.config = config

        self.baseline_model = baseline_model
        for name, param in self.baseline_model.named_parameters():
            param.requires_grad = False
        self.update_model = update_model

        self.denseblock = dense_block(
            [2, 2],
            ["ReLU", "ReLU"],
            [
                2,
            ],
        )

    def forward(self, input_map):
        """
        Performs forward pass of the model.

        Args:
            input (torch.Tensor): Input data.

        Returns:
            torch.Tensor: Output of the model.

        """
        output_base = self.baseline_model(input_map)
        x = self.update_model(input_map)
        output_booster = self.denseblock(torch.concat([output_base, x], dim=1))

        output = output_base + output_booster

        return output
