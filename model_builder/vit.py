"""
Vision Transformer (ViT) model for regression tasks.

Functions
---------
None

Classes
---------
ViTModel(BaseModel)
    TorchModel class represents a torch-based model for a specific task.

"""

import torch
from torchvision.transforms import v2
from vit_pytorch import simple_vit
from vit_pytorch.cct import cct_8 as cct_module

from base.base_model import BaseModel
from base.base_model import RescaleLayer


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

        # self.vit = simple_vit.SimpleViT(
        #     num_classes=1,
        #     image_size=max(self.input_shape),
        #     channels=config["architecture"]["channels"],
        #     patch_size=config["architecture"]["patch_size"],
        #     dim=config["architecture"]["dim"],
        #     depth=config["architecture"]["depth"],
        #     heads=config["architecture"]["heads"],
        #     mlp_dim=config["architecture"]["mlp_dim"],
        # )

        self.vit = cct_module(
            img_size=max(self.input_shape),
            n_conv_layers=config["architecture"]["n_conv_layers"],
            kernel_size=config["architecture"]["kernel_size"],
            stride=2,
            padding=3,
            pooling_kernel_size=3,
            pooling_stride=2,
            pooling_padding=1,
            num_classes=1,
            positional_embedding="learnable",  # ['sine', 'learnable', 'none']
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
