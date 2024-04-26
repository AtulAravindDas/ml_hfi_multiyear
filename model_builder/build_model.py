"""Network modules for pytorch models.

Functions
---------
conv_couplet(in_channels, out_channels, act_fun, *args, **kwargs)
dense_lazy_couplet(out_features, act_fun, *args, **kwargs)
conv_block(in_channels, out_channels, act_fun, kernel_size)
dense_block(out_features, act_fun)


Classes
---------
RescaleLayer()
TorchModel(base.base_model.BaseModel)

"""

import torch
import numpy as np
from torchvision.transforms import v2
from base.base_model import BaseModel
import utils.utils as utils

import time

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
    def __init__(self, config):
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
        assert len(self.config["architecture"]["dense_units"]) == len(self.config["architecture"]["dense_activations"])

        # Augmentation layers
        self.augmentation = torch.nn.Sequential(
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
        )

        # CNN block
        self.conv_block = conv_block(
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

        # Rescaling layers
        self.rescale_input = RescaleLayer(1.0 / 255.0, 0.0)
        self.rescale_target = RescaleLayer(100.0, 0.0)

        # Output layers
        self.output = dense_couplet(
            in_features=config["architecture"]["final_in"],
            out_features=1,
            act_fun=self.config["architecture"]["final_activation"],
        )

    def forward(self, input):

        # rescale input
        input = self.rescale_input(input)
        x = input

        # data augmentation
        if self.training:
            x = self.augmentation(x)

        # CNN block
        x = self.conv_block(x)

        # flat layer
        x = self.flat(x)

        # skip connection
        input_flat = self.flat(input[:, 2, :, :])
        x = torch.cat((x, input_flat), dim=-1)

        # dropout layer
        x = self.dropout(x)

        # dense block
        x = self.denseblock(x)

        # output layer
        x = self.output(x)
        x = self.rescale_target(x)

        return x

    def predict(self, dataloader, device="cpu"):

        self.to(device)
        self.eval()
        with torch.inference_mode():

            output = np.zeros((len(dataloader.dataset.sample_files), 1))
            start_time = time.time()
            for batch_idx, (data, target) in enumerate(dataloader):

                data, target = data.to(device), target.to(device)
                out = self(data).to("cpu")

                batch_size = len(out)
                output[batch_idx * batch_size : (batch_idx + 1) * batch_size] = out

                if batch_idx % 100 == 0:
                    execution_time = time.time() - start_time
                    print(
                        f"batch {batch_idx} of {int(np.ceil(output.shape[0] / batch_size))} - "
                        f"{execution_time:.3f}s - "
                        f"{execution_time/((batch_idx+1)*batch_size):.7f}s/sample"
                    )

            output = np.asarray(output)

            end_time = time.time()
            execution_time = end_time - start_time
            print(f"\nExecution time: {execution_time:.3f}s")
            print(f"Number samples: {output.shape[0]}")
            print(f"Time per sample: {execution_time/output.shape[0]:.7f}s")

        return output
