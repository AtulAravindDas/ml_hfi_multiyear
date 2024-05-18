"""
Convolutional Neural Network (CNN) model for regression tasks.

Functions
---------

Classes
---------
CNNModel(base.base_model.BaseModel)
    A class representing a torch-based model for a specific task.

"""

import torch
from torchvision.transforms import v2

from base.base_model import BaseModel
from base.base_model import RescaleLayer
from base.base_model import dense_couplet, conv_block, dense_block


class CNNModel(BaseModel):
    """
    TorchModel class represents a torch-based model for a specific task.

    Args:
        config (dict): Configuration parameters for the model.

    Attributes:
        config (dict): Configuration parameters for the model.
        input_shape (tuple): Shape of the input data.
        augmentation (torch.nn.Sequential): Augmentation layers.
        conv_block (torch.nn.Module): CNN block.
        flat (torch.nn.Flatten): Flat layer.
        dropout (torch.nn.Dropout): Dropout layer.
        denseblock (torch.nn.Module): Dense blocks.
        skip_channels (list): List of skip channels.
        rescale_input (RescaleLayer): Rescaling layer for input data.
        rescale_target (RescaleLayer): Rescaling layer for target data.
        output (torch.nn.Module): Output layers.

    Methods:
        forward(input): Performs forward pass of the model.

    """

    def __init__(self, config):
        """
        Initializes the TorchModel.

        Args:
            config (dict): Configuration parameters for the model.

        """
        super().__init__()

        self.config = config
        self.input_shape = (
            config["scene_width_landsat"],
            config["scene_width_landsat"],
            len(config["data"]["channels"]),
        )

        assert (
            len(self.config["architecture"]["cnn_activation"])
            == len(self.config["architecture"]["kernel_size"])
            == len(self.config["architecture"]["filters"])
        )
        assert len(self.config["architecture"]["dense_units"]) == len(
            self.config["architecture"]["dense_activations"]
        )

        # DEFINE MODEL LAYERS

        # Augmentation layers
        self.augmentation = torch.nn.Sequential(
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
        )

        # DEFINE CNN BLOCKS and RESCALE LAYERS
        # Build custom CNN block
        self.rescale_input = RescaleLayer((1.0 / 255.0), 0.0)
        self.skip_channels = config["architecture"]["skip_channels"]

        # CNN block
        self.base_cnn_block = conv_block(
            [self.input_shape[-1], *config["architecture"]["filters"][:-1]],
            [*config["architecture"]["filters"]],
            [*config["architecture"]["cnn_activation"]],
            [*config["architecture"]["kernel_size"]],
        )

        # Flat layer
        self.flat = torch.nn.Flatten(start_dim=1)

        # Dropout layer
        self.dropout = torch.nn.Dropout(p=config["architecture"]["dropout"])

        # Dense blocks
        self.denseblock = dense_block(
            config["architecture"]["dense_units"],
            config["architecture"]["dense_activations"],
            in_features=config["architecture"]["dense_in"],
        )

        # Output scaling layer
        self.rescale_target = RescaleLayer(100.0, 0.0)

        # Output layers
        self.output = dense_couplet(
            in_features=config["architecture"]["final_in"],
            out_features=1,
            act_fun=self.config["architecture"]["final_activation"],
        )

    def forward(self, input):
        """
        Performs forward pass of the model.

        Args:
            input (torch.Tensor): Input data.

        Returns:
            torch.Tensor: Output of the model.

        """
        # data augmentation
        if self.training:
            input = self.augmentation(input)

        # rescale input
        input = self.rescale_input(input)

        # Base CNN Block
        x = self.base_cnn_block(input)

        # flat layer
        x = self.flat(x)

        # skip connections
        input_flat_chA = self.flat(input[:, self.skip_channels[0], :, :])
        input_flat_chB = self.flat(input[:, self.skip_channels[1], :, :])
        x = torch.cat((x, input_flat_chA, input_flat_chB), dim=-1)

        # dropout layer
        x = self.dropout(x)

        # dense blocks
        x = self.denseblock(x)

        # output layer
        x = self.output(x)
        x = self.rescale_target(x)

        return x
