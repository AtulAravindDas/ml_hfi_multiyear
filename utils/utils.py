"""Utility classes and functions.

Functions
---------
prepare_device(device="gpu")
save_torch_model(model, filename)
load_torch_model(model, filename)
get_config(exp_name)

Classes
---------
MetricTracker()

"""

import json
import torch
import numpy as np


def get_directories():
    with open("utils/directories.json") as f:
        directories = json.load(f)
    return directories


def get_model_name(expname, seed):
    return expname + "_seed" + str(seed)


def prepare_device(device="gpu"):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    if device == "gpu":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            print("Warning: MPS device not found." "Training will be performed on CPU.")
            device = torch.device("cpu")
    elif device == "cpu":
        device = torch.device("cpu")
    else:
        raise NotImplementedError

    return device


def save_torch_model(model, filename):
    torch.save(model.state_dict(), filename)


def load_torch_model(model, filename):
    model.load_state_dict(torch.load(filename))
    model.eval()
    return model


def get_config(exp_name):

    basename = "exp_"

    # GET CONFIG
    with open("configs/config_" + exp_name[len(basename) :] + ".json") as f:
        config = json.load(f)

    assert (
        config["exp_name"] == basename + exp_name[len(basename) :]
    ), "Exp_Name must be equal to config[exp_name]"

    # GET CONSTANTS
    with open("utils/constants.json") as f:
        constants = json.load(f)

    config["tile_len_deg"] = constants["TILE_LEN_DEG"]
    config["landsat_pixel_to_deg"] = constants["LANDSAT_PIXEL_TO_DEG"]
    config["landsat_to_hfi_ratio"] = constants["LANDSAT_TO_HII_RATIO"]

    # SET SCENE WIDTH
    assert config["scene_width"] % 2 == 1, "the scene_width must be an odd number in units of hfi pixels"
    config["scene_width_landsat"] = int(config["scene_width"] * config["landsat_to_hfi_ratio"])

    return config


class MetricTracker:
    def __init__(self, *keys):

        self.history = dict()
        for k in keys:
            self.history[k] = []
        self.reset()

    def reset(self):
        for key in self.history:
            self.history[key] = []

    def update(self, key, value):
        if key in self.history:
            self.history[key].append(value)

    def result(self):
        for key in self.history:
            self.history[key] = np.nanmean(self.history[key])

    def print(self, idx=None):
        for key in self.history.keys():
            if idx is None:
                print(f"  {key} = {self.history[key]:.5f}")
            else:
                print(f"  {key} = {self.history[key][idx]:.5f}")
