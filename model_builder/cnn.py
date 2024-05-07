"""
Network modules for pytorch models.

Functions
---------
conv_couplet(in_channels, out_channels, act_fun, *args, **kwargs)
    Create a sequential module consisting of a convolutional layer, activation function, and max pooling layer.

dense_lazy_couplet(out_features, act_fun, *args, **kwargs)
    Create a sequential module consisting of a lazy linear layer and activation function.

conv_block(in_channels, out_channels, act_fun, kernel_size)
    Create a sequential module consisting of multiple conv_couplet modules.

dense_block(out_features, act_fun)
    Create a sequential module consisting of multiple dense_lazy_couplet modules.

Classes
---------
RescaleLayer
    A class representing a rescaling layer.

TorchModel(base.base_model.BaseModel)
    A class representing a torch model.

"""

import torch
from torchvision.transforms import v2

from base.base_model import BaseModel
from base.base_model import RescaleLayer
from base.base_model import dense_couplet, conv_block, dense_block


# https://github.com/FrancescoSaverioZuppichini/Pytorch-how-and-when-to-use-Module-Sequential-ModuleList-and-ModuleDict


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
        rescale_input (RescaleLayer): Rescaling layer for input data.
        rescale_target (RescaleLayer): Rescaling layer for target data.
        output (torch.nn.Module): Output layers.

    Methods:
        forward(input): Performs forward pass of the model.
        predict(dataloader, device): Makes predictions using the model.

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
        self.skip_channels = (2, -1)

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
        # TODO: make it a choice how many final dense blocks to use
        self.denseblockA = dense_block(
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

        # skip connection
        input_flat_chA = self.flat(input[:, self.skip_channels[0], :, :])
        input_flat_chB = self.flat(input[:, self.skip_channels[1], :, :])
        x = torch.cat((x, input_flat_chA, input_flat_chB), dim=-1)

        # dropout layer
        x = self.dropout(x)

        # dense blocks
        x = self.denseblockA(x)

        # output layer
        x = self.output(x)
        x = self.rescale_target(x)

        return x
