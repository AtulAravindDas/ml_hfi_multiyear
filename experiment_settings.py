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
            "scene_width": 11,  # in units of HFI, must be ODD. 30, 50, 70, 90, 110, 130
            "nbatches": (750, 25),  # (training_batches, validation_batches per landsat tile)
            "batches_per_epoch": 750,
            "rng_seed": 33,

            "layers_units": (64, 128, 128, 128, 128),
            "kernel_size": 3,
            "max_pool_stride": (2, 2),  # (pool size, stride length)
            "dense_units": 32,

            "loss": "DenseWeightMSE",
            "kluge_value_for_zero": 0.,  # -0.2,
            "kluge_value_for_one": 0.,  # -0.2,
            "aug_randomflip": True,
            "input_noise": 10,  # in units of rgb values
            "sample_weights_alpha": 15.0,
            "subsample": True,

            "learning_rate": 0.001,
            "dropout": 0.10,
            "patience": 3,
            "batch_size": 32,
            "max_epochs": 1_000,
            "early_stopping": True,

            "pickup_where_leftoff": False,
            "save_best_only": True,
        },

        "exp1": {
            "mode": None,  # "training" or "inference" as set in the code itself
            "training_years": (2015, 2016, 2017, 2018, 2019),  # (2000, 2005, 2010, 2013)
            "inference_years": (2020, ),

            "training_region": (-90, 90, -180, 180),  # (lat_south, lat_north, lon_west, lon_east)
            "inference_region": (-90, 90, -180, 180),  # (lat_south, lat_north, lon_west, lon_east)

            "channels": (1, 2, 3, 4, 5, 6),  # indexing starts at 1, channel 7 = WATER_MASK
            "scene_width": 11,  # in units of HFI, must be ODD. 30, 50, 70, 90, 110, 130
            "nbatches": (750, 25),  # (training_batches, validation_batches per landsat tile)
            "batches_per_epoch": 750,
            "rng_seed": 33,

            "layers_units": (64, 128, 128, 128, 128),
            "kernel_size": 3,
            "max_pool_stride": (2, 2),  # (pool size, stride length)
            "dense_units": 32,

            "loss": "DenseWeightMSE",
            "extra_denseweight_high": (80, 1.3),
            "kluge_value_for_zero": 0.,  # -0.2,
            "kluge_value_for_one": 0.,  # -0.2,
            "aug_randomflip": True,
            "input_noise": 10,  # in units of rgb values
            "sample_weights_alpha": 25.0,
            "subsample": True,

            "learning_rate": 0.001,
            "dropout": 0.10,
            "patience": 3,
            "batch_size": 32,
            "max_epochs": 1_000,
            "early_stopping": True,

            "pickup_where_leftoff": False,
            "save_best_only": True,
        },

        "exp2": {
            "mode": None,  # "training" or "inference" as set in the code itself
            "training_years": (2015, 2016, 2017, 2018, 2019),  # (2000, 2005, 2010, 2013)
            "inference_years": (2020, ),

            "training_region": (-90, 90, -180, 180),  # (lat_south, lat_north, lon_west, lon_east)
            "inference_region": (-90, 90, -180, 180),  # (lat_south, lat_north, lon_west, lon_east)

            "channels": (1, 2, 3, 4, 5, 6),  # indexing starts at 1, channel 7 = WATER_MASK
            "scene_width": 11,  # in units of HFI, must be ODD. 30, 50, 70, 90, 110, 130
            "nbatches": (750, 25),  # (training_batches, validation_batches per landsat tile)
            "batches_per_epoch": 750,
            "rng_seed": 33,

            "layers_units": (64, 128, 128, 128, 128),
            "kernel_size": 3,
            "max_pool_stride": (2, 2),  # (pool size, stride length)
            "dense_units": 32,

            "loss": "DenseDualWeightMSE",
            "loss_params": (0.2, 0.0),  # gamma power, offset
            "kluge_value_for_zero": 0.,  # -0.2,
            "kluge_value_for_one": 0.,  # -0.2,
            "aug_randomflip": True,
            "input_noise": 10,  # in units of rgb values
            "sample_weights_alpha": 15.0,
            "subsample": True,

            "learning_rate": 0.001,
            "dropout": 0.10,
            "patience": 3,
            "batch_size": 32,
            "max_epochs": 1_000,
            "early_stopping": True,

            "pickup_where_leftoff": False,
            "save_best_only": True,
        },

        "exp3": {
            "mode": None,  # "training" or "inference" as set in the code itself
            "training_years": (2015, 2016, 2017, 2018, 2019),  # (2000, 2005, 2010, 2013)
            "inference_years": (2020, ),

            "training_region": (-90, 90, -180, 180),  # (lat_south, lat_north, lon_west, lon_east)
            "inference_region": (-90, 90, -180, 180),  # (lat_south, lat_north, lon_west, lon_east)

            "channels": (1, 2, 3, 4, 5, 6),  # indexing starts at 1, channel 7 = WATER_MASK
            "scene_width": 11,  # in units of HFI, must be ODD. 30, 50, 70, 90, 110, 130
            "nbatches": (750, 25),  # (training_batches, validation_batches per landsat tile)
            "batches_per_epoch": 750,
            "rng_seed": 33,

            "layers_units": (64, 128, 128, 128, 128),
            "kernel_size": 3,
            "max_pool_stride": (2, 2),  # (pool size, stride length)
            "dense_units": 32,

            "loss": "DenseDualWeightMSE",
            "loss_params": (0.2, 50.),  # gamma power, offset
            "kluge_value_for_zero": 0.,  # -0.2,
            "kluge_value_for_one": 0.,  # -0.2,
            "aug_randomflip": True,
            "input_noise": 10,  # in units of rgb values
            "sample_weights_alpha": 15.0,
            "subsample": True,

            "learning_rate": 0.001,
            "dropout": 0.10,
            "patience": 3,
            "batch_size": 32,
            "max_epochs": 1_000,
            "early_stopping": True,

            "pickup_where_leftoff": False,
            "save_best_only": True,
        },

        "exp4": {
            "mode": None,  # "training" or "inference" as set in the code itself
            "training_years": (2015, 2016, 2017, 2018, 2019),  # (2000, 2005, 2010, 2013)
            "inference_years": (2020, ),

            "training_region": (-90, 90, -180, 180),  # (lat_south, lat_north, lon_west, lon_east)
            "inference_region": (-90, 90, -180, 180),  # (lat_south, lat_north, lon_west, lon_east)

            "channels": (1, 2, 3, 4, 5, 6),  # indexing starts at 1, channel 7 = WATER_MASK
            "scene_width": 11,  # in units of HFI, must be ODD. 30, 50, 70, 90, 110, 130
            "nbatches": (750, 25),  # (training_batches, validation_batches per landsat tile)
            "batches_per_epoch": 750,
            "rng_seed": 33,

            "layers_units": (64, 128, 128, 128, 128),
            "kernel_size": 3,
            "max_pool_stride": (2, 2),  # (pool size, stride length)
            "dense_units": 32,

            "loss": "DenseDualWeightMSE",
            "loss_params": (0.2, 50.),  # gamma power, offset
            "kluge_value_for_zero": 0.,  # -0.2,
            "kluge_value_for_one": 0.,  # -0.2,
            "aug_randomflip": True,
            "input_noise": 10,  # in units of rgb values
            "sample_weights_alpha": 15.0,
            "subsample": True,

            "learning_rate": 0.001,
            "dropout": 0.10,
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
    exp_dict["landsat_pixel_to_deg"] = LANDSAT_PIXEL_TO_DEG
    exp_dict["landsat_to_hfi_ratio"] = LANDSAT_TO_HII_RATIO

    assert exp_dict["scene_width"] % 2 == 1, "the scene_width must be an odd number in units of hfi pixels"
    exp_dict["scene_width_landsat"] = int(exp_dict["scene_width"] * exp_dict["landsat_to_hfi_ratio"])

    return exp_dict
