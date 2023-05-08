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

            "training_years": (2005, 2013,),  # (2000, 2005, 2010, 2013)
            "testing_years": (2010, ),
            "channels": (1, 2, 3, 4, 5, 6),  # indexing starts at 1
            "nbatches": (200, 25),  # (training_batches, validation_batches)
            "scene_width": 3,  # in units of HFI pixels, so 3x3
            "training_ibounds": (9697, 9787, 28915, 29005),  # (ilat0, ilat1, ilon0, ilon1) --> max bounds: (0, 15683, 0, 36390)
            "rng_seed": 33,

            "layers_units": (64, 128, 128, 128, 128),
            "kernel_size": 3,
            "max_pool_stride": (2, 2),  # (pool size, stride length)
            "dense_units": 32,
            "aug_randomflip": True,

            "learning_rate": 0.001,
            "dropout": 0.25,
            "patience": 3,
            "batch_size": 32,
            "max_epochs": 1_000,
            "early_stopping": True,

            "save_best_only": True,
        },

    }

    exp_dict = experiments[experiment_name]
    exp_dict['exp_name'] = experiment_name

    return exp_dict
