"""Experimental settings

Functions
---------
get_settings(experiment_name)
"""
import numpy as np

__author__ = "Elizabeth A. Barnes and Randal J. Barnes"
__date__ = "09 May 2023"


# DECLARE CONSTANTS
TILE_LEN_DEG = 1.
LANDSAT_TO_HII_RATIO = 10.
LANDSAT_PIXEL_TO_DEG = 0.00026949


def get_settings(experiment_name):
    experiments = {

        "exp0": {
            "mode": None,  # "training" or "inference" as set in the code itself
            "training_years": (2015, 2016, 2017, 2018, 2019),  # (2000, 2005, 2010, 2013)
            "inference_years": (2020, ),

            "training_region": (-90, 90, -180, 180),  # (lat_south, lat_north, lon_west, lon_east)
            "inference_region": (-90, 90, -180, 180),  # (lat_south, lat_north, lon_west, lon_east)

            "channels": (1, 2, 3, 4, 5, 6),  # indexing starts at 1, channel 7 = WATER_MASK
            "scene_width": 114,  # in units of landsat pixels
            "nbatches": (500, 50),  # (training_batches, validation_batches per landsat tile)
            "batches_per_epoch": 500,
            "rng_seed": 33,

            "layers_units": (64, 128, 128, 128, 128),
            "kernel_size": 3,
            "max_pool_stride": (2, 2),  # (pool size, stride length)
            "dense_units": 32,

            "kluge_value_for_zero": -.2,  # -0.2,
            "kluge_value_for_one": 0.,  # -0.2,
            "aug_randomflip": True,
            "input_noise": 10,  # in units of rgb values
            "sample_weights_alpha": 25.0,
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
    exp_dict["tile_len_deg"] = TILE_LEN_DEG
    exp_dict["landsat_to_hfi_ratio"] = LANDSAT_TO_HII_RATIO
    exp_dict["landsat_pixel_to_deg"] = LANDSAT_PIXEL_TO_DEG

    assert exp_dict["scene_width"] % 3 == 0, "The scene_width must be divisible by 3."

    return exp_dict
