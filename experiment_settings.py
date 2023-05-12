"""Experimental settings

Functions
---------
get_settings(experiment_name)
"""
import numpy as np

__author__ = "Elizabeth A. Barnes and Randal J. Barnes"
__date__ = "09 May 2023"


def get_settings(experiment_name):
    experiments = {

        "exp0": {

            "training": True,
            "training_years": (2015, 2016, 2017, 2018),  # (2000, 2005, 2010, 2013)
            "testing_year": 2020,
            "tilename": "-6lat_106lon",
            "latlon_bounds": (-6.04, -6.92, 106.05, 106.93),  # (lat0, lat1, lon0, lon1)
            # "latlon_bounds": (-6.1, -6.34, 106.6, 106.93),  # (lat0, lat1, lon0, lon1)
            "channels": (1, 2, 3, 4, 5, 6),  # indexing starts at 1, channel 7 = WATER_MASK
            "nbatches": (200, 50),  # (training_batches, validation_batches)
            "scene_width": 114,  # in units of landsat pixels
            "rng_seed": 33,

            "layers_units": (64, 128, 128, 128, 128),
            "kernel_size": 3,
            "max_pool_stride": (2, 2),  # (pool size, stride length)
            "dense_units": 32,

            "kluge_value_for_zero": -0.2,
            "aug_randomflip": True,
            "input_noise": 10,  # in units of rgb values
            "sample_weights_alpha": 10.0,
            "subsample": True,

            "learning_rate": 0.001,
            "dropout": 0.50,
            "patience": 3,
            "batch_size": 32,
            "max_epochs": 1_000,
            "early_stopping": True,

            "pickup_where_leftoff": False,
            "save_best_only": True,
        },


    }

    exp_dict = experiments[experiment_name]
    exp_dict['exp_name'] = experiment_name

    assert exp_dict["scene_width"] % 3 == 0, "The scene_width must be divisible by 3."

    return exp_dict
