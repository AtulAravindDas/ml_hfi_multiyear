"""
Base model modules for pytorch models.

This module contains utility functions and a base class for creating PyTorch models.

Classes
---------
BaseModel(torch.nn.Module): Base class for all models.

Functions
---------
conv_couplet(in_channels, out_channels, act_fun, *args, **kwargs):
    Creates a convolutional couplet.
dense_lazy_couplet(out_features, act_fun, *args, **kwargs):
    Creates a lazy dense couplet.
dense_couplet(in_features, out_features, act_fun, *args, **kwargs):
    Creates a dense couplet.
conv_block(in_channels, out_channels, act_fun, kernel_size):
    Creates a block of convolutional couplets.
dense_block(out_features, act_fun, in_features=None):
    Creates a block of dense couplets.
RescaleLayer:
    A rescaling layer class.

"""

import torch
import numpy as np
from abc import abstractmethod
import time

# Utility functions


def conv_couplet(in_channels, out_channels, act_fun, *args, **kwargs):
    """
    Creates a convolutional couplet.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        act_fun (str): Activation function to use.
        *args: Additional positional arguments for torch.nn.Conv2d.
        **kwargs: Additional keyword arguments for torch.nn.Conv2d.

    Returns:
        torch.nn.Sequential: Sequential module containing the convolutional couplet.

    """
    return torch.nn.Sequential(
        torch.nn.Conv2d(in_channels, out_channels, *args, **kwargs),
        getattr(torch.nn, act_fun)(),
        torch.nn.MaxPool2d(kernel_size=(2, 2), ceil_mode=True),
    )


def dense_lazy_couplet(out_features, act_fun, *args, **kwargs):
    """
    Creates a lazy dense couplet.

    Args:
        out_features (int): Number of output features.
        act_fun (str): Activation function to use.
        *args: Additional positional arguments for torch.nn.LazyLinear.
        **kwargs: Additional keyword arguments for torch.nn.LazyLinear.

    Returns:
        torch.nn.Sequential: Sequential module containing the lazy dense couplet.

    """
    return torch.nn.Sequential(
        torch.nn.LazyLinear(out_features=out_features, bias=True),
        getattr(torch.nn, act_fun)(),
    )


def dense_couplet(in_features, out_features, act_fun, *args, **kwargs):
    """
    Creates a dense couplet.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        act_fun (str): Activation function to use.
        *args: Additional positional arguments for torch.nn.Linear.
        **kwargs: Additional keyword arguments for torch.nn.Linear.

    Returns:
        torch.nn.Sequential: Sequential module containing the dense couplet.

    """
    return torch.nn.Sequential(
        torch.nn.Linear(in_features=in_features, out_features=out_features, bias=True),
        getattr(torch.nn, act_fun)(),
    )


def conv_block(in_channels, out_channels, act_fun, kernel_size):
    """
    Creates a block of convolutional couplets.

    Args:
        in_channels (list): List of input channel sizes.
        out_channels (list): List of output channel sizes.
        act_fun (list): List of activation functions to use.
        kernel_size (list): List of kernel sizes.

    Returns:
        torch.nn.Sequential: Sequential module containing the block of convolutional couplets.

    """
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
    """
    Creates a block of dense couplets.

    Args:
        out_features (list): List of output feature sizes.
        act_fun (list): List of activation functions to use.
        in_features (list, optional): List of input feature sizes. Defaults to None.

    Returns:
        torch.nn.Sequential: Sequential module containing the block of dense couplets.

    """
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
    """
    Rescaling layer class.

    Args:
        scale (float): Scaling factor.
        offset (float): Offset value.

    """

    def __init__(self, scale, offset):
        self.offset = offset
        self.scale = scale

    def __call__(self, x):
        """
        Rescale the input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Rescaled tensor.

        """
        x = torch.multiply(x, self.scale)
        x = torch.add(x, self.offset)
        return x


class BaseModel(torch.nn.Module):
    """
    Base class for all models.

    This class provides a base implementation for creating PyTorch models.

    Methods
    -------
    forward(*inputs):
        Forward pass logic.

    freeze_layers(freeze_id, verbose=False):
        Freeze layers in the model.

    predict(dataloader, device="cpu"):
        Makes predictions using the model.

    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, *inputs):
        """
        Forward pass logic.

        This method should be overridden by subclasses.

        Returns:
            Model output.

        """
        raise NotImplementedError

    def __str__(self):
        """
        Model prints with number of trainable parameters.

        Returns:
            str: Model representation with number of trainable parameters.

        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + "\nTrainable parameters: {}".format(params)

    def freeze_layers(self, freeze_id, verbose=False):
        """
        Freeze layers in the model.

        Args:
            freeze_id (str): Identifier for the layers to freeze.
            verbose (bool, optional): Whether to print verbose information. Defaults to False.

        """
        params = self.state_dict()
        params.keys()

        for name, param in self.named_parameters():
            if freeze_id in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

        if verbose:
            for name, param in self.named_parameters():
                print("-" * 20)
                print(f"name: {name}, ")
                print(str(param.numel()))
                print(", train: ")
                print(param.requires_grad)

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
