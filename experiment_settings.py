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
            "channels": (2, 3, 4),
            "channel_norms": (255., 255., 255.),
            "nbatches": (200, 10),  # (training_batches, validation_batches)
            "scene_width": 3,  # in units of HFI pixels, so 3x3
            "training_ibounds": (9697, 9787, 28915, 29005),  # (ilat0, ilat1, ilon0, ilon1)
            "rng_seed": 33,

            "layers_units": (64, 128, 128, 128, 128),
            "kernel_size": 3,
            "max_pool_stride": (2, 2),  # (pool size, stride length)
            "dense_units": 32,
            "aug_randomflip": True,

            "learning_rate": 0.001,
            "dropout": 0.5,
            "patience": 3,
            "batch_size": 32,
            "max_epochs": 3,
            "early_stopping": False,
        },

    }

    exp_dict = experiments[experiment_name]
    exp_dict['exp_name'] = experiment_name

    assert len(exp_dict["channels"]) == len(exp_dict["channel_norms"]), "channel and channel_norms must have equal lengths"

    return exp_dict
