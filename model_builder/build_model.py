"""
Network modules to build pytorch models.

Functions
---------
ModelBuilder(config)

Classes
---------
"""


def ModelBuilder(config):
    """
    Build a model based on the configuration.

    Args:
        config (dict): Configuration parameters for the model.

    Returns:
        torch.nn.Module: A torch model.

    Raises:
        ValueError: If the model type specified in the configuration is not recognized.

    """
    if config["architecture"]["type"] == "cnn":
        from model_builder.cnn import CNNModel
        print("Building CNNModel model.")
        return CNNModel(config)

    elif config["architecture"]["type"] == "resnet":
        from model_builder.resnet import ResNetModel
        print("Building ResNetModel model.")
        return ResNetModel(config)

    elif config["architecture"]["type"] == "vit":
        from model_builder.vit import ViTModel
        print("Building ViTModel model.")
        return ViTModel(config)

    else:
        raise ValueError(f"Model type {config['architecture']['type']} not recognized.")
