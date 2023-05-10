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

            "training_years": (2015, 2016, 2017, 2018),  # (2000, 2005, 2010, 2013)
            "testing_year": 2020,
            "tilename": "06S_106E",
            "latlon_bounds": (-6.04, -6.92, 106.05, 106.93),  # (lat0, lat1, lon0, lon1)
            "channels": (1, 2, 3, 4, 5, 6),  # indexing starts at 1
            "nbatches": (200, 25),  # (training_batches, validation_batches)
            "scene_width": 114,  # in units of landsat pixels
            "rng_seed": 33,

            "layers_units": (64, 128, 128, 128, 128),
            "kernel_size": 3,
            "max_pool_stride": (2, 2),  # (pool size, stride length)
            "dense_units": 32,
            "aug_randomflip": True,

            "kluge_value_for_zero": -0.2,
            "learning_rate": 0.001,
            "dropout": 0.5,
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

    return exp_dict
