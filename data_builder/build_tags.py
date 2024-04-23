"""Build the data.

Classes
---------
data_generator

Functions
---------
get_tags(config)
build_dataset(config, sample_years, sample_lats, sample_lons)
make_sample_list(config,)
data_generator.get_input_data(self, years, sample_lats, sample_lons)
data_generator.get_output_data(self, years, sample_lats, sample_lons)
"""

import os
import numpy as np
from sklearn.utils.class_weight import compute_sample_weight
import rasterio
from rasterio.windows import Window
from rasterio.transform import Affine
import methods
import time

from data_builder import read_landsat


directory_paths = methods.get_directories()
SAVE_MODEL_DIRECTORY = directory_paths["save_model_dir"]
DATA_DIRECTORY = directory_paths["data_dir"]
LANDSAT_DIRECTORY = directory_paths["landsat_dir"]
PREDICTIONS_DIRECTORY = directory_paths["predictions_dir"]

DEFAULT_MASK_FILENAME = DATA_DIRECTORY + "hii_coastal_buffer_mask.tif"


def get_tags(config):
    if config["mode"] == "training":
        tags_train, tags_val = get_training_tags(config)
    elif config["mode"] == "inference":
        tags_train, tags_val = get_inference_tags(config)
    else:
        raise NotImplementedError("no such mode.")

    return tags_train, tags_val


def get_training_tags(config):
    rng = np.random.default_rng(config["rng_seed"])

    (
        lat_s_bound,
        lat_n_bound,
        lon_w_bound,
        lon_e_bound,
    ) = read_landsat.get_landsat_bounds(config, region=config["training_region"])
    print(lat_s_bound, lat_n_bound, lon_w_bound, lon_e_bound)
    print("\n")

    with rasterio.open(DEFAULT_MASK_FILENAME) as buffer_mask:
        sample_lats = []
        sample_lons = []
        sample_years = []

        for latfile in np.arange(
            lat_s_bound + config["tile_len_deg"],
            lat_n_bound + config["tile_len_deg"],
            config["tile_len_deg"],
        ):
            for lonfile in np.arange(
                lon_w_bound, lon_e_bound, config["tile_len_deg"]
            ):
                # check that landsat file even exists
                landsat_filenames = read_landsat.get_input_filename(
                    config["training_years"],
                    np.ones(len(config["training_years"])) * latfile,
                    np.ones(len(config["training_years"])) * lonfile,
                    config,
                )
                file_flag = False
                for file in landsat_filenames:
                    if os.path.isfile(LANDSAT_DIRECTORY + file + ".tif") is False:
                        file_flag = True
                        break
                if file_flag:
                    # print(f"skipping landsat file that does not exist: {file}")
                    continue

                print(landsat_filenames[0])

                # get indices for region and tile
                tile_bounds = (
                    latfile - config["tile_len_deg"],
                    latfile,
                    lonfile,
                    lonfile + config["tile_len_deg"],
                )
                ilat_s, ilat_n, ilon_w, ilon_e = methods.get_tile_indices(
                    buffer_mask, tile_bounds
                )
                ilat_s, ilat_n, ilon_w, ilon_e = methods.trim_hfi_region(
                    (ilat_s, ilat_n, ilon_w, ilon_e),
                    buffer_mask,
                    region=config["training_region"],
                )

                # get tag indices that are not water or on edges
                window = Window.from_slices((ilat_n, ilat_s + 1), (ilon_w, ilon_e + 1))
                mask = buffer_mask.read(1, window=window)
                land_pixels = np.sum(mask)
                frac_land = land_pixels / (mask.shape[0] * mask.shape[1])

                land_indices = np.argwhere(mask)
                ilat_grid, ilon_grid = (
                    land_indices[:, 0] + window.row_off,
                    land_indices[:, 1] + window.col_off,
                )

                subsample_lons, subsample_lats = buffer_mask.xy(
                    ilat_grid, ilon_grid, offset="ul"
                )
                subsample_lons, subsample_lats = np.asarray(subsample_lons), np.asarray(
                    subsample_lats
                )

                # Remove all possibilities of edges and corners
                edge_width = (
                    (np.ceil((config["scene_width"] - 1) / 2) + 1)
                    * config["landsat_to_hfi_ratio"]
                    * config["landsat_pixel_to_deg"]
                )

                i_nonedge = np.flatnonzero(
                    np.logical_and(
                        np.abs(
                            np.abs(subsample_lons)
                            - np.rint(np.abs(subsample_lons) / config["tile_len_deg"])
                            * config["tile_len_deg"]
                        )
                        > edge_width,
                        np.abs(
                            np.abs(subsample_lats)
                            - np.rint(np.abs(subsample_lats) / config["tile_len_deg"])
                            * config["tile_len_deg"]
                        )
                        > edge_width,
                    )
                )
                subsample_lats = subsample_lats[i_nonedge]
                subsample_lons = subsample_lons[i_nonedge]

                if len(subsample_lats) < config["batch_size"]:
                    continue

                nbatches = int(np.ceil(np.sum(config["nbatches"]) * frac_land))
                nsamples = int(config["batch_size"] * nbatches)

                # grab the samples
                isamples = rng.choice(
                    range(len(subsample_lats)), size=nsamples, replace=False
                )
                subsample_lats, subsample_lons = (
                    subsample_lats[isamples],
                    subsample_lons[isamples],
                )

                subsample_years = np.repeat(
                    np.random.choice(
                        config["training_years"], size=nbatches, replace=True
                    ),
                    config["batch_size"],
                )
                assert len(subsample_years) == len(
                    subsample_lats
                ), "sample years and locations must be the same length"

                # append to list across tiles
                sample_lats = sample_lats + subsample_lats.tolist()
                sample_lons = sample_lons + subsample_lons.tolist()
                sample_years = sample_years + subsample_years.tolist()

                # print meta data
                print(f"...{frac_land.round(3) = }, # samples = {len(subsample_years)}")

    assert len(sample_lats) > 0, "you have no training data."

    # Turn into numpy arrays
    sample_lats, sample_lons, sample_years = (
        np.asarray(sample_lats),
        np.asarray(sample_lons),
        np.asarray(sample_years, dtype="int"),
    )

    # SPLIT INTO TRAINING AND VALIDATION SETS
    nbatches = int(len(sample_lats) // config["batch_size"])
    p = config["nbatches"][0] / np.sum(config["nbatches"])
    train_bool = np.repeat(
        np.random.choice([0, 1], size=nbatches, replace=True, p=[1.0 - p, p]),
        config["batch_size"],
    )

    isample = np.flatnonzero(train_bool == 1)
    taglat_train, taglon_train, tagyear_train = (
        sample_lats[isample],
        sample_lons[isample],
        sample_years[isample],
    )
    isample = np.flatnonzero(train_bool == 0)
    taglat_val, taglon_val, tagyear_val = (
        sample_lats[isample],
        sample_lons[isample],
        sample_years[isample],
    )

    # GET FILENAMES
    tagfile_train = read_landsat.get_input_filename(
        tagyear_train, taglat_train, taglon_train, config
    )
    tagfile_val = read_landsat.get_input_filename(
        tagyear_val, taglat_val, taglon_val, config
    )

    # Put into nice packages
    tags_train = (tagyear_train, taglat_train, taglon_train, tagfile_train)
    tags_val = (tagyear_val, taglat_val, taglon_val, tagfile_val)

    # PRINT META DATA
    print(
        f"\ntotal training samples = {len(tagyear_train)}, total validation samples = {len(tagyear_val)}\n"
    )

    return tags_train, tags_val


def get_inference_tags(config):
    with rasterio.open(DEFAULT_MASK_FILENAME) as buffer_mask:
        ilat_s, ilat_n, ilon_w, ilon_e = methods.get_tile_indices(
            buffer_mask, config["tile"]
        )
        ilat_s, ilat_n, ilon_w, ilon_e = methods.trim_hfi_region(
            (ilat_s, ilat_n, ilon_w, ilon_e),
            buffer_mask,
            region=config["inference_region"],
        )
        window = Window.from_slices((ilat_n, ilat_s + 1), (ilon_w, ilon_e + 1))
        mask = buffer_mask.read(1, window=window)
        land_pixels = np.sum(mask)
        if land_pixels == 0:
            return ([], [], [], []), None

        ilat_grid, ilon_grid = np.meshgrid(
            np.arange(ilat_n, ilat_s + 1), np.arange(ilon_w, ilon_e + 1), indexing="ij"
        )
        print("output region shape = " + str(ilon_grid.shape))

        sample_lons, sample_lats = buffer_mask.xy(
            np.ndarray.flatten(ilat_grid, order="C"),
            np.ndarray.flatten(ilon_grid, order="C"),
            offset="ul",
        )
        # print(np.min(sample_lats), np.max(sample_lats), np.min(sample_lons), np.max(sample_lons))

    sample_lons, sample_lats = np.asarray(sample_lons), np.asarray(sample_lats)

    tagyear_inf = np.asarray(
        np.ones(shape=sample_lats.shape) * config["inference_years"][0],
        dtype=int,
    )
    tagfile_inf = read_landsat.get_input_filename(
        tagyear_inf, sample_lats, sample_lons, config
    )

    # PRINT SIZES
    n_inference = tagyear_inf.shape
    print(f"{n_inference = }")
    assert len(n_inference) > 0, "you have no data to predict."

    # Put into a nice package
    tags = (tagyear_inf, sample_lats, sample_lons, tagfile_inf)
    return tags, None
