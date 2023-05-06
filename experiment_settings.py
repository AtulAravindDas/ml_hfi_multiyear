"""Experimental settings

Functions
---------
get_settings(experiment_name)
"""
import numpy as np

__author__ = "Elizabeth A. Barnes and Randal J. Barnes"
__date__ = "05 May 2023"


def get_settings(experiment_name):
    experiments = {

        "exp0": {

            "training_years": (2013, 2010, 2005),
            "validation_years": (0, 0),
            "channels": (2, 3, 4),
            "channel_norms": (255., 255., 255.),
            "n_batches": (100, 20),  # (training_size, validation_size)
            "scene_width": 3,  # in units of HFI pixels, so 3x3
            "training_ibounds": (9697, 9787, 28915, 29005),  # (ilat0, ilat1, ilon0, ilon1)

            "layers_units": (64, 128, 128, 128, 128),
            "kernel_size": 3,
            "max_pool_stride": (2, 2),  # (pool size, stride length)
            "dense_units": 32,
            "learning_rate": 0.001,
            "activation": "relu",
            "activation_output": "sigmoid",
            "dropout": 0.5,
            "rng_seed": 33,
            "patience": 3,
            "batch_size": 128,
            "max_epochs": 10_000,

            "aug_randomflip": True,
        },

    }

    exp_dict = experiments[experiment_name]
    exp_dict['exp_name'] = experiment_name

    assert len(exp_dict["channels"]) == len(exp_dict["channel_norms"]), "channel and channel_norms must have equal lengths"

    return exp_dict
