"""Build the data.

Classes
---------
data_generator

Functions
---------
get_tags(config)
    Retrieves the training and validation tags based on the provided configuration.
build_dataset(config, sample_years, sample_lats, sample_lons)
    Builds the dataset using the provided configuration and sample years, latitudes, and longitudes.
make_sample_list(config,)
    Creates a list of samples based on the provided configuration.
data_generator.get_input_data(self, years, sample_lats, sample_lons)
    Retrieves the input data for the data generator based on the provided years, latitudes, and longitudes.
data_generator.get_output_data(self, years, sample_lats, sample_lons)
    Retrieves the output data for the data generator based on the provided years, latitudes, and longitudes.
"""

import os
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.transform import Affine
import time
from collections import defaultdict

from utils import utils
from data_builder import read_landsat
from data_builder import data_methods


default_filepaths = utils.get_default_filepaths()
DEFAULT_MASK_FILEPATH = default_filepaths["mask_filepath"]
DEFAULT_HII_FILEPATH = default_filepaths["hii_filepath"]


def get_tags(config):

    if config["mode"] == "training":

        tags_train, tags_val = utils.load_training_tags(config)

        if tags_train is None or tags_val is None:
            tags_train, tags_val = get_training_tags(config)
            utils.save_training_tags(config, tags_train, tags_val)

        print(f"# Training Samples:   {len(tags_train[0])}")
        print(f"# Validation Samples: {len(tags_val[0])}")

    elif config["mode"] == "inference":
        tags_train, tags_val = get_inference_tags(config)
    else:
        raise NotImplementedError("no such mode.")

    return tags_train, tags_val


def get_training_tags(config):

    directory_paths = utils.get_directories(config["machine"])
    rng = np.random.default_rng(config["seed"] + 33)

    lats_lons_dict = data_methods.get_lats_lons_list(
        region=config["data"]["training_region"], tile_len_deg=config["tile_len_deg"]
    )

    print(
        f"\nTraining Tile Bounds: \n"
        f"  {lats_lons_dict['lats'] = } \n"
        f"  {lats_lons_dict['lons'] = } "
    )

    with rasterio.open(DEFAULT_MASK_FILEPATH) as buffer_mask:
        print(
            f"Creating mask from {DEFAULT_MASK_FILEPATH} "
            f"and class bins from {DEFAULT_HII_FILEPATH}"
        )

        sample_lats = []
        sample_lons = []
        sample_years = []

        for lats_list, lons_list in zip(lats_lons_dict["lats"], lats_lons_dict["lons"]):
            print(lats_list, lons_list)

            for latfile in lats_list:
                for lonfile in lons_list:

                    # check that landsat file even exists
                    landsat_filenames = read_landsat.get_input_filename(
                        config["data"]["training_years"],
                        np.ones(len(config["data"]["training_years"])) * latfile,
                        np.ones(len(config["data"]["training_years"])) * lonfile,
                        config,
                    )
                    file_flag = False
                    for file in landsat_filenames:
                        if (
                            os.path.isfile(
                                directory_paths["landsat_dir"] + file + ".tif"
                            )
                            is False
                        ):
                            file_flag = True
                            break
                    if file_flag:
                        # print(f"skipping landsat file that does not exist: {file}")
                        continue

                    print(f"Building tags for {landsat_filenames}")

                    # get indices for region and tile
                    tile_bounds = (
                        latfile - config["tile_len_deg"],
                        latfile,
                        lonfile,
                        lonfile + config["tile_len_deg"],
                    )
                    ilat_s, ilat_n, ilon_w, ilon_e = data_methods.get_tile_indices(
                        buffer_mask, tile_bounds
                    )
                    ilat_s, ilat_n, ilon_w, ilon_e = data_methods.trim_hfi_region(
                        (ilat_s, ilat_n, ilon_w, ilon_e),
                        buffer_mask,
                        region=config["data"]["training_region"],
                    )

                    # get tag indices that are not water or on edges
                    window = Window.from_slices(
                        (ilat_n, ilat_s + 1), (ilon_w, ilon_e + 1)
                    )
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
                    subsample_lons, subsample_lats = np.asarray(
                        subsample_lons
                    ), np.asarray(subsample_lats)

                    # Remove all possibilities of edges and corners
                    edge_width = (
                        (np.ceil((config["data"]["scene_width"] - 1) / 2) + 1)
                        * config["landsat_to_hfi_ratio"]
                        * config["landsat_pixel_to_deg"]
                    )

                    i_nonedge = np.flatnonzero(
                        np.logical_and(
                            np.abs(
                                np.abs(subsample_lons)
                                - np.rint(
                                    np.abs(subsample_lons) / config["tile_len_deg"]
                                )
                                * config["tile_len_deg"]
                            )
                            > edge_width,
                            np.abs(
                                np.abs(subsample_lats)
                                - np.rint(
                                    np.abs(subsample_lats) / config["tile_len_deg"]
                                )
                                * config["tile_len_deg"]
                            )
                            > edge_width,
                        )
                    )
                    subsample_lats = subsample_lats[i_nonedge]
                    subsample_lons = subsample_lons[i_nonedge]

                    if len(subsample_lats) < config["trainer"]["batch_size"]:
                        continue

                    # SUBSAMPLE FOR EQUAL SAMPLING ACROSS VALUES

                    # Subsample to deal with imbalanced data
                    with rasterio.open(DEFAULT_HII_FILEPATH) as tif:
                        array = tif.sample(
                            [*zip(subsample_lons, subsample_lats)], indexes=1
                        )
                        array = np.ndarray.flatten(np.asarray(list(array)))
                    percentage_sampling = config["data"]["percentage_sampling"]

                    # Calculate the histogram of the array to determine the distribution
                    hist, bins = np.histogram(array, bins=np.arange(11) * 10)

                    # Calculate the total number of samples
                    total_samples = len(array)

                    # Calculate the minimum number of samples per bin based on the minimum percentage
                    min_samples_threshold = int(total_samples * percentage_sampling)

                    # this ensures that the max is 'percentage_sampling'%, if there are not, we just take what is available but no more than 'percentage_sampling'% equally divided in the number of bins
                    max_samples_per_bin = int(min_samples_threshold // len(bins))

                    # Initialize list to store indices to keep
                    indices_to_keep = []

                    # Iterate over each bin
                    for i in range(len(bins) - 1):
                        bin_start = bins[i]
                        bin_end = bins[i + 1]
                        bin_indices = np.where(
                            (array >= bin_start) & (array < bin_end)
                        )[0]

                        # Pull samples from each bin to create balanced decile bins
                        if (
                            len(bin_indices) < max_samples_per_bin
                            and config["data"]["oversample_rare_bins"]
                            and len(bin_indices) > 0
                        ):
                            # If the number of samples in the bin is less than the minimum threshold,
                            # we sample more.
                            assert (
                                config["data"]["oversample_rate"] > 1
                            ), "oversample_rate must be > 1"

                            num_samples_to_keep = int(
                                np.min(
                                    [
                                        max_samples_per_bin,
                                        len(bin_indices)
                                        * config["data"]["oversample_rate"],
                                    ]
                                )
                            )
                            indices_to_keep.extend(
                                rng.choice(
                                    bin_indices, num_samples_to_keep, replace=True
                                )
                            )
                        else:
                            # Do not to exceed the number of available samples and sample what is available
                            num_samples_to_keep = min(
                                max_samples_per_bin, len(bin_indices)
                            )
                            indices_to_keep.extend(
                                rng.choice(
                                    bin_indices, num_samples_to_keep, replace=False
                                )
                            )

                        print(
                            f"   {bin_start:<3}- {bin_end:<3}: n_total={len(bin_indices):<10}"
                            f"n_keep={num_samples_to_keep:<10}"
                        )

                    # indices to keep f(#samples grabbed)
                    subsample_lats, subsample_lons = (
                        subsample_lats[
                            indices_to_keep
                        ],  # indices_to_keep instead of isamples
                        subsample_lons[
                            indices_to_keep
                        ],  # indices_to_keep instead of isamples
                    )

                    subsample_years = rng.choice(
                        config["data"]["training_years"],
                        size=len(indices_to_keep),
                        replace=True,
                    )

                    assert len(subsample_years) == len(
                        subsample_lats
                    ), "sample years and locations must be the same length"

                    # append to list across tiles
                    sample_lats.extend(subsample_lats.tolist())
                    sample_lons.extend(subsample_lons.tolist())
                    sample_years.extend(subsample_years.tolist())

                    # print meta data
                    print(
                        f"frac_land={frac_land.round(3)}, #samples = {len(subsample_years)}\n"
                    )

    assert (
        len(sample_lats) > 0
    ), "You have no training data. Is config['machine'] set correctly? This is a common error."

    # Turn into numpy arrays
    sample_lats, sample_lons, sample_years = (
        np.asarray(sample_lats),
        np.asarray(sample_lons),
        np.asarray(sample_years, dtype="int"),
    )

    # SPLIT INTO TRAINING AND VALIDATION SETS
    nbatches = int(len(sample_lats) // config["trainer"]["batch_size"])
    p = config["data"]["val_frac"]
    train_bool = np.repeat(
        np.random.choice([0, 1], size=nbatches, replace=True, p=[p, 1.0 - p]),
        config["trainer"]["batch_size"],
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

    # Clean the tags
    tags_train, tags_val = clean_tags(
        tags_train, tags_val, config["trainer"]["batch_size"]
    )

    return tags_train, tags_val


def clean_tags(tags_train, tags_val, min_count=0):
    """
    Clean the tags based on a threshold value.

    Args:
        tags_train (tuple): A tuple containing the training tags.
        tags_val (tuple): A tuple containing the validation tags.
        min_count (int): The min_count value for cleaning the tags (batch_size)

    Returns:
        tuple: A tuple containing the cleaned training tags and cleaned validation tags.
    """

    def indices_below_count(array, min_count):
        return [i for i, element in enumerate(array) if element <= min_count]

    def clean(tags):
        tags_dict = defaultdict(list)

        for index, filename in enumerate(tags[-1]):
            tags_dict[filename].append(index)

        # Convert lists to numpy arrays
        for key in tags_dict:
            tags_dict[key] = np.asarray(tags_dict[key])

        sample_years, sample_lats, sample_lons, sample_files = tags

        acum = np.zeros(len(sample_files), dtype=float)  # Preallocate memory for acum
        for i, tile_key in enumerate(sample_files):
            acum[i] = len(tags_dict[tile_key])

        ind = indices_below_count(acum, min_count)

        tags_clean = list(tags)

        for i in range(3):  # Loop to modify the first three lists in tags
            tags_clean[i] = np.delete(tags[i], ind)

        tags_clean[3] = [tags[3][i] for i in range(len(tags[3])) if i not in ind]

        # Convert the modified list back to a tuple if necessary
        return tuple(tags_clean)

    tags_train_clean = clean(tags_train)
    tags_val_clean = clean(tags_val)

    return tags_train_clean, tags_val_clean


def get_inference_tags(config):
    with rasterio.open(DEFAULT_MASK_FILEPATH) as buffer_mask:
        ilat_s, ilat_n, ilon_w, ilon_e = data_methods.get_tile_indices(
            buffer_mask, config["tile"]
        )
        ilat_s, ilat_n, ilon_w, ilon_e = data_methods.trim_hfi_region(
            (ilat_s, ilat_n, ilon_w, ilon_e),
            buffer_mask,
            region=config["data"]["inference_region"],
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

    sample_lons, sample_lats = np.asarray(sample_lons), np.asarray(sample_lats)

    tagyear_inf = np.asarray(
        np.ones(shape=sample_lats.shape) * config["data"]["inference_years"][0],
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
