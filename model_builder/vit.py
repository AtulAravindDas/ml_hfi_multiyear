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
from vit_pytorch import simple_vit

from base.base_model import BaseModel
from base.base_model import RescaleLayer


# https://github.com/lucidrains/vit-pytorch?tab=readme-ov-file
class ViTModel(BaseModel):
    """
    TorchModel class represents a torch-based model for a specific task.

    Args:
        config (dict): Configuration parameters for the model.

    Attributes:
        config (dict): Configuration parameters for the model.

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

        # DEFINE MODEL LAYERS

        # Augmentation layers
        self.augmentation = torch.nn.Sequential(
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
        )

        self.rescale_input = RescaleLayer((1.0 / 255.0), 0.0)
        # self.skip_channels = (2, -1)

        self.vit = simple_vit.SimpleViT(
            num_classes=1,
            image_size=max(self.input_shape),
            channels=config["architecture"]["channels"],
            patch_size=config["architecture"]["patch_size"],
            dim=config["architecture"]["dim"],
            depth=config["architecture"]["depth"],
            heads=config["architecture"]["heads"],
            mlp_dim=config["architecture"]["mlp_dim"],
        )

        # Output scaling layer
        self.sigmoid = torch.nn.Sigmoid()
        self.rescale_target = RescaleLayer(100.0, 0.0)

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

        # ViT
        x = self.vit(input)

        # output layer
        x = self.sigmoid(x)
        x = self.rescale_target(x)

        return x

