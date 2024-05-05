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
import numpy as np
from torchvision.transforms import v2
from itertools import islice
import time

from base.base_model import BaseModel
import utils.utils as utils

import torchvision.models as models
import torch.nn as nn
from torchinfo import summary


# https://github.com/FrancescoSaverioZuppichini/Pytorch-how-and-when-to-use-Module-Sequential-ModuleList-and-ModuleDict


def conv_couplet(in_channels, out_channels, act_fun, *args, **kwargs):
    return torch.nn.Sequential(
        torch.nn.Conv2d(in_channels, out_channels, *args, **kwargs),
        getattr(torch.nn, act_fun)(),
        torch.nn.MaxPool2d(kernel_size=(2, 2), ceil_mode=True),
    )


def dense_lazy_couplet(out_features, act_fun, *args, **kwargs):
    return torch.nn.Sequential(
        torch.nn.LazyLinear(out_features=out_features, bias=True),
        getattr(torch.nn, act_fun)(),
    )


def dense_couplet(in_features, out_features, act_fun, *args, **kwargs):
    return torch.nn.Sequential(
        torch.nn.Linear(in_features=in_features, out_features=out_features, bias=True),
        getattr(torch.nn, act_fun)(),
    )


def conv_block(in_channels, out_channels, act_fun, kernel_size):
    block = [
        conv_couplet(in_channels, out_channels, act_fun, kernel_size, padding="same")
        for in_channels, out_channels, act_fun, kernel_size in zip(
            [*in_channels],
            [*out_channels],
            [*act_fun],
            [*kernel_size],
        )
    ]
    return torch.nn.Sequential(*block)


def dense_block(out_features, act_fun, in_features=None):
    if in_features is None:
        block = [
            dense_lazy_couplet(out_channels, act_fun)
            for out_channels, act_fun in zip([*out_features], [*act_fun])
        ]
        return torch.nn.Sequential(*block)
    else:
        block = [
            dense_couplet(in_features, out_features, act_fun)
            for in_features, out_features, act_fun in zip(
                [*in_features], [*out_features], [*act_fun]
            )
        ]
        return torch.nn.Sequential(*block)


class RescaleLayer:
    def __init__(self, scale, offset):
        self.offset = offset
        self.scale = scale

    def __call__(self, x):
        x = torch.multiply(x, self.scale)
        x = torch.add(x, self.offset)
        return x


class TorchModel(BaseModel):
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
        self.isresnet = self.config["architecture"].get("resnet", False)

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
        # Decide whether to build custom CNN blocks or use ResNet18
        if self.isresnet:
            # Build ResNet18 block
            # scaling specific to resnet and torchvision
            self.rescale_input = RescaleLayer((1.0 / (255.0 * 0.22)), -0.44)
            self.skip_channels = (1, -1)

            if config["architecture"].get("resnet_pretrained", True):
                layers = list(
                    models.resnet18(
                        weights="ResNet18_Weights.IMAGENET1K_V1"
                    ).children()
                )[: config["architecture"]["resnet_drop_layer"]]
            else:
                layers = list(
                    models.resnet18(
                        weights=None,
                    ).children()
                )[: config["architecture"]["resnet_drop_layer"]]
            self.base_cnn_block = nn.Sequential(*layers)

            # Set trainable Resnet layers
            for param in self.base_cnn_block.parameters():
                param.requires_grad = config["architecture"]["resnet_trainable"]

        else:
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
        # self.denseblockB = dense_block(
        #     config["architecture"]["dense_units"],
        #     config["architecture"]["dense_activations"],
        #     in_features=config["architecture"]["dense_units"],
        # )

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
        # x = self.denseblockB(x)

        # output layer
        x = self.output(x)
        x = self.rescale_target(x)

        return x

    def predict(self, dataloader, device="cpu"):
        """
        Makes predictions using the model.

        Args:
            dataloader (torch.utils.data.DataLoader): DataLoader for the dataset.
            device (str): Device to use for predictions. Default is "cpu".

        Returns:
            numpy.ndarray: Array of predictions.

        """
        self.to(device)
        self.eval()
        with torch.inference_mode():

            output = np.zeros((len(dataloader.dataset.sample_files), 1))
            start_time = time.time()

            for batch_idx, (data, target) in enumerate(dataloader):

                data, target = data.to(device), target.to(device)
                out = self(data).to("cpu")

                # save predictions to output
                # cannot used batch_size = len(out) because of quicklook settings
                batch_size = self.config["inference"]["batch_size"]

                if self.config["inference"]["quicklook"]:
                    output[
                        batch_idx
                        * batch_size : (batch_idx + 1)
                        * batch_size : self.config["inference"]["quicklook_skiplen"]
                    ] = out
                else:
                    output[batch_idx * batch_size : (batch_idx + 1) * batch_size] = out

                if batch_idx % 1000 == 0:
                    execution_time = time.time() - start_time
                    print(
                        f"batch {batch_idx} of {int(np.ceil(output.shape[0] / batch_size))} - "
                        f"{execution_time:.3f}s - "
                        f"{execution_time / ((batch_idx + 1) * batch_size):.9f}s/sample"
                    )

            output = np.asarray(output)

            end_time = time.time()
            execution_time = end_time - start_time
            print(f"\nExecution time: {execution_time:.3f}s")
            print(f"Number samples: {output.shape[0]}")
            print(f"Time per sample: {execution_time / output.shape[0]:.7f}s \n")

        return output
