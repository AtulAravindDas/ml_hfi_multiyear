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

import methods

# https://github.com/FrancescoSaverioZuppichini/Pytorch-how-and-when-to-use-Module-Sequential-ModuleList-and-ModuleDict

directory_paths = methods.get_directories()
SAVE_MODEL_DIRECTORY = directory_paths["save_model_dir"]


def get_model(config):
    model = TorchModel(config)

    checkpoint_dir = SAVE_MODEL_DIRECTORY + config["exp_name"] + "/"
    # model.load_weights(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()
    return model


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
            len(config["channels"]),
        )

        assert (
            len(self.config["cnn_activation"])
            == len(self.config["kernel_size"])
            == len(self.config["filters"])
        )
        assert len(self.config["dense_units"]) == len(self.config["dense_activations"])

        # CNN block
        self.conv_block = conv_block(
            [self.input_shape[-1], *config["filters"][:-1]],
            [*config["filters"]],
            [*config["cnn_activation"]],
            [*config["kernel_size"]],
        )

        # Flat layer
        self.flat = torch.nn.Flatten(start_dim=1)

        # Dense blocks
        self.denseblock = dense_block(
            config["dense_units"],
            config["dense_activations"],
            in_features=config["dense_in"],
        )

        # Rescaling layers
        self.rescale_input = RescaleLayer(1.0 / 255.0, 0.0)
        self.rescale_target = RescaleLayer(100.0, 0.0)

        # Output layers
        self.output = dense_couplet(
            in_features=config["final_in"],
            out_features=1,
            act_fun=self.config["final_activation"],
        )

    def forward(self, input):

        # rescale input
        x_scaled = self.rescale_input(input)

        # data augmentation
        x = v2.RandomHorizontalFlip()(x_scaled)
        x = v2.RandomVerticalFlip()(x)

        # CNN block
        x = self.conv_block(x)

        # flat layer
        x = self.flat(x)

        # skip connection
        x_scaled_flat = self.flat(x_scaled[:, 2, :, :])
        x = torch.cat((x, x_scaled_flat), dim=-1)

        # dropout layer
        x = torch.nn.Dropout(p=self.config["dropout"])(x)

        # dense block
        x = self.denseblock(x)

        # output layer
        x = self.output(x)
        x = self.rescale_target(x)

        return x

    def predict(self, dataset=None, dataloader=None, batch_size=128, device="cpu"):

        if (dataset is None) & (dataloader is None):
            raise ValueError("both dataset and dataloader cannot be none.")

        if (dataset is not None) & (dataloader is not None):
            raise ValueError(
                "dataset and dataloader cannot both be defined. choose one."
            )

        if dataset is not None:
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
            )

        self.to(device)
        self.eval()
        with torch.inference_mode():

            output = None
            for batch_idx, (data, target) in enumerate(dataloader):
                input, input_unit, target = (
                    data[0].to(device),
                    data[1].to(device),
                    target.to(device),
                )

                out = self(input, input_unit).to("cpu").numpy()
                if output is None:
                    output = out
                else:
                    output = np.concatenate((output, out), axis=0)

        return output
