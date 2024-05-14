"""
Utility classes and functions.

Functions
---------
get_directories(machine)
    Get the directory paths from the "utils/directories.json" file.

get_default_filepaths()
    Get the default file paths from the "utils/default_filepaths.json" file.

prepare_device(device="gpu", device_id=0)
    Set up the GPU device if available and return the device.

get_model_name(expname, seed)
    Generate the model name based on the experiment name and seed.

get_model_dir(expname, model_name, machine="falco")
    Get the directory path for saving the model based on the experiment name and model name.

get_predictions_filename(config, landsat_name)
    Get the file name for saving the predictions based on the experiment configuration and Landsat name.

get_prediction_dir(expname, model_name, machine="falco")
    Get the directory path for saving the predictions based on the experiment name and model name.

save_training_tags(config, tags_train, dict_train, tags_val, dict_val)
    Save the training tags to the specified directory based on the experiment configuration.

load_training_tags(config)
    Load the training tags from the specified directory based on the experiment configuration.

save_torch_model(model, config)
    Save the PyTorch model to the specified directory based on the experiment configuration.

load_torch_model(model, filename)
    Load the PyTorch model from the specified file and return the loaded model.

load_model(config, clean=True)
    Load the TorchModel based on the experiment configuration and return the loaded model.

get_config(expname)
    Get the experiment configuration based on the experiment name.

Classes
---------
MetricTracker
    A class for tracking and calculating metrics during training.

"""

import os
import json
import torch
import numpy as np
import pickle


def get_directories(machine):
    """Get the directory paths from the "utils/directories.json" file."""
    with open("utils/directories.json") as f:
        directories = json.load(f)
    return directories[machine]


def get_default_filepaths():
    """Get the default file paths from the "utils/default_filepaths.json" file."""
    with open("utils/default_filepaths.json") as f:
        filepaths = json.load(f)
    return filepaths


def prepare_device(device="gpu", device_id=0):
    """
    Set up the GPU device if available and return the device.

    Parameters
    ----------
    device : str, optional
        The device to use, either "gpu" or "cpu" (default is "gpu").

    Returns
    -------
    torch.device
        The device to be used for training.

    Raises
    ------
    NotImplementedError
        If the specified device is not supported.
    """
    if device == "gpu":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda:" + str(device_id))
        else:
            print(
                "Warning: CUDA and/or MPS devices are not available. Training will be performed on CPU."
            )
            device = torch.device("cpu")
    elif device == "cpu":
        device = torch.device("cpu")
    else:
        raise NotImplementedError

    return device


def get_model_name(expname, seed):
    """
    Generate the model name based on the experiment name and seed.

    Parameters
    ----------
    expname : str
        The experiment name.
    seed : int
        The seed value.

    Returns
    -------
    str
        The generated model name.
    """
    return expname + "_seed" + str(seed)


def get_model_dir(expname, model_name, machine="falco"):
    """
    Get the directory path for saving the model based on the experiment name and model name.

    Parameters
    ----------
    expname : str
        The experiment name.
    model_name : str
        The model name.

    Returns
    -------
    str
        The directory path for saving the model.
    """
    directory_paths = get_directories(machine)
    MODEL_DIR = directory_paths["save_model_dir"]

    dir = MODEL_DIR + expname + "/" + model_name + "/"
    os.makedirs(dir, exist_ok=True)

    return dir


def get_predictions_filename(config, landsat_name):
    """
    Get the file name for saving the predictions based on the experiment configuration and Landsat name.

    Parameters
    ----------
    config : dict
        The experiment configuration.
    landsat_name : str
        The Landsat name.

    Returns
    -------
    str
        The file name for saving the predictions.
    """
    model_name = get_model_name(config["expname"], config["seed"])
    dir = get_prediction_dir(config["expname"], model_name, machine=config["machine"])
    return dir + "predictions_" + landsat_name + ".tif"


def get_prediction_dir(expname, model_name, machine="falco"):
    """
    Get the directory path for saving the predictions based on the experiment name and model name.

    Parameters
    ----------
    expname : str
        The experiment name.
    model_name : str
        The model name.

    Returns
    -------
    str
        The directory path for saving the predictions.
    """
    directory_paths = get_directories(machine)
    PREDICTION_DIR = directory_paths["predictions_dir"]

    dir = PREDICTION_DIR + expname + "/" + model_name + "/"
    os.makedirs(dir, exist_ok=True)

    return dir


def save_training_tags(config, tags_train, dict_train, tags_val, dict_val):
    """
    Save the training tags to the specified directory based on the experiment configuration.

    Parameters
    ----------
    config : dict
        The experiment configuration.
    tags_train : list
        The training tags.
    tags_val : list
        The validation tags.
    """
    import utils

    dir = utils.utils.get_directories(config["machine"])["tags_dir"]
    model_name = get_model_name(config["expname"], config["seed"])

    with open(dir + model_name + "_tags_train.pkl", "wb") as f:
        pickle.dump(tags_train, f)
    with open(dir + model_name + "_tags_dict_train.pkl", "wb") as f:
        pickle.dump(dict_train, f)
    with open(dir + model_name + "_tags_val.pkl", "wb") as f:
        pickle.dump(tags_val, f)
    with open(dir + model_name + "_tags_dict_val.pkl", "wb") as f:
        pickle.dump(dict_val, f)

    print("saved training tags.")


def load_training_tags(config):
    """
    Load the training tags from the specified directory based on the experiment configuration.

    Parameters
    ----------
    config : dict
        The experiment configuration.

    Returns
    -------
    tuple
        A tuple containing the loaded training tags and validation tags.
    """
    import utils

    dir = utils.utils.get_directories(config["machine"])["tags_dir"]

    config_tags_name = config["data"].get("tags_loadname", None)
    if config_tags_name is None:
        config_tags_name = get_model_name(config["expname"], config["seed"])

    if not os.path.exists(dir + config_tags_name + "_tags_train.pkl"):
        return None, None, None, None
    if not os.path.exists(dir + config_tags_name + "_tags_dict_train.pkl"):
        return None, None, None, None
    if not os.path.exists(dir + config_tags_name + "_tags_val.pkl"):
        return None, None, None, None
    if not os.path.exists(dir + config_tags_name + "_tags_dict_val.pkl"):
        return None, None, None, None

    with open(dir + config_tags_name + "_tags_train.pkl", "rb") as f:
        tags_train = pickle.load(f)
    with open(dir + config_tags_name + "_tags_dict_train.pkl", "rb") as f:
        tags_dict_train = pickle.load(f)
    with open(dir + config_tags_name + "_tags_val.pkl", "rb") as f:
        tags_val = pickle.load(f)
    with open(dir + config_tags_name + "_tags_dict_val.pkl", "rb") as f:
        tags_dict_val = pickle.load(f)

    return tags_train, tags_dict_train, tags_val, tags_dict_val


def save_torch_model(model, config, epoch=None):
    """
    Save the PyTorch model to the specified directory based on the experiment configuration.

    Parameters
    ----------
    model : TorchModel
        The PyTorch model.
    config : dict
        The experiment configuration.
    """
    model_name = get_model_name(config["expname"], config["seed"])
    if epoch is not None:
        model_name = model_name + "_epoch" + str(epoch).zfill(3)
    dir = get_model_dir(config["expname"], model_name, machine=config["machine"])

    torch.save(model.state_dict(), dir + model_name + ".pt")


def load_torch_model(model, filename):
    """
    Load the PyTorch model from the specified file and return the loaded model.

    Parameters
    ----------
    model : TorchModel
        The PyTorch model.
    filename : str
        The file name of the saved model.

    Returns
    -------
    TorchModel
        The loaded PyTorch model.
    """
    model.load_state_dict(torch.load(filename, map_location=torch.device("cpu")))
    model.eval()
    return model


def load_model(config, clean=True):
    """
    Load the TorchModel based on the experiment configuration and return the loaded model.

    Parameters
    ----------
    config : dict
        The experiment configuration.
    clean : bool, optional
        Whether to load a clean model or the saved model (default is True).

    Returns
    -------
    TorchModel
        The loaded TorchModel.
    """
    from model_builder.build_model import ModelBuilder

    model = ModelBuilder(config)

    if clean:
        return model
    else:
        try:
            model_name = get_model_name(config["expname"], config["seed"])
            dir = get_model_dir(
                config["expname"], model_name, machine=config["machine"]
            )

            model = load_torch_model(model, dir + model_name + ".pt")
            print("Loading model from: ", dir + model_name + ".pt\n")
            return model
        except:
            print("Saved model not found.")
            if config["mode"] == "inference":
                raise FileNotFoundError
            else:
                print("Using a clean model.\n")
                return model


def get_config(expname):
    """
    Get the experiment configuration based on the experiment name.

    Parameters
    ----------
    expname : str
        The experiment name.

    Returns
    -------
    dict
        The experiment configuration.
    """
    basename = "exp"

    # GET CONFIG
    with open("configs/config_" + expname[len(basename) :] + ".json") as f:
        config = json.load(f)

    assert (
        config["expname"] == basename + expname[len(basename) :]
    ), "Exp_Name must be equal to config[expname]"

    # SET CONFIG MODE
    config["mode"] = None

    # GET CONSTANTS
    with open("utils/constants.json") as f:
        constants = json.load(f)

    config["tile_len_deg"] = constants["TILE_LEN_DEG"]
    config["landsat_pixel_to_deg"] = constants["LANDSAT_PIXEL_TO_DEG"]
    config["landsat_to_hfi_ratio"] = constants["LANDSAT_TO_HII_RATIO"]

    # CHECK IF QUICKLOOK and CHANGE BATCH SIZE
    if config["inference"]["quicklook"]:
        config["inference"]["batch_size"] = config["inference"]["batch_size"] * (
            config["inference"]["quicklook_skiplen"] - 1
        )

    # SET SCENE WIDTH
    assert (
        config["data"]["scene_width"] % 2 == 1
    ), "the scene_width must be an odd number in units of hfi pixels"
    config["scene_width_landsat"] = int(
        config["data"]["scene_width"] * config["landsat_to_hfi_ratio"]
    )

    return config


class MetricTracker:
    """
    A class for tracking and calculating metrics during training.

    Parameters
    ----------
    *keys : str
        The keys for the metrics to be tracked.

    Attributes
    ----------
    history : dict
        A dictionary to store the history of the tracked metrics.

    Methods
    -------
    reset()
        Reset the history of the tracked metrics.

    update(key, value)
        Update the history of the specified metric with the given value.

    result()
        Calculate the average of the tracked metrics.

    print(idx=None)
        Print the tracked metrics with their corresponding values.
    """

    def __init__(self, *keys):
        self.history = dict()
        for k in keys:
            self.history[k] = []
        self.reset()

    def reset(self):
        """Reset the history of the tracked metrics."""
        for key in self.history:
            self.history[key] = []

    def update(self, key, value):
        """
        Update the history of the specified metric with the given value.

        Parameters
        ----------
        key : str
            The key of the metric.
        value : float
            The value of the metric.
        """
        if key in self.history:
            self.history[key].append(value)

    def result(self):
        """Calculate the average of the tracked metrics."""
        for key in self.history:
            self.history[key] = np.nanmean(self.history[key])

    def print(self, idx=None):
        """
        Print the tracked metrics with their corresponding values.

        Parameters
        ----------
        idx : int, optional
            The index of the metric value to be printed (default is None).
        """
        for key in self.history.keys():
            if idx is None:
                print(f"  {key} = {self.history[key]:.5f}")
            else:
                print(f"  {key} = {self.history[key][idx]:.5f}")


def set_random_seeds(seed=42):
    """
    Set the random seeds for reproducibility.

    Parameters
    ----------
    seed : int, optional
        The seed value (default is 42).
    """
    import torch
    import random

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
