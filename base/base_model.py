"""Base model modules for pytorch models.

Classes
---------
BaseModel(torch.nn.Module)

"""

import torch
import numpy as np
from abc import abstractmethod
import time


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


class BaseModel(torch.nn.Module):
    """
    Base class for all models.
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, *inputs):
        """
        Forward pass logic

        :return: Model output
        """
        raise NotImplementedError

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return (
            super().__str__()
            + "\nTrainable parameters: {}".format(params)
        )

    def freeze_layers(self, freeze_id, verbose=False):
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
