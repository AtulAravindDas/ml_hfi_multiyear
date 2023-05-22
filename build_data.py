"""Build the data.

Classes
---------
data_generator

Functions
---------
get_tags(settings)
build_tf_dataset(settings, sample_years, sample_lats, sample_lons, batch_size)
make_sample_list(settings,)
data_generator.get_input_data(self, years, sample_lats, sample_lons)
data_generator.get_output_data(self, years, sample_lats, sample_lons)
"""
import os
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_sample_weight
import rasterio
from rasterio.windows import Window
from rasterio.transform import Affine
import methods

# from methods import permute_shuffle_sample_list
import read_landsat


directory_paths = methods.get_directories()
SAVE_MODEL_DIRECTORY = directory_paths["save_model_dir"]
DATA_DIRECTORY = directory_paths["data_dir"]
LANDSAT_DIRECTORY = directory_paths["landsat_dir"]
PREDICTIONS_DIRECTORY = directory_paths["predictions_dir"]

DEFAULT_MASK_FILENAME = DATA_DIRECTORY + "hii_coastal_buffer_mask.tif"
DEFAULT_FILENAME = "hii_2020-01-01_uint8.tif"


def build_tf_dataset(settings, tags, batch_size, mode=None):
    # override mode if desired
    if mode is None:
        mode = settings["mode"]

    # create tag dictionary for training
    tags_dict = {}
    sample_years, sample_lats, sample_lons, sample_files = tags
    for filename in np.unique(sample_files):
        i = [index for (index, item) in enumerate(sample_files) if item == filename]
        tags_dict[filename] = (
            sample_years[i],
            sample_lats[i],
            sample_lons[i],
            np.asarray(sample_files)[i],
        )

    # make data generator class
    data_gen = data_generator(settings, tags_dict)

    # create initial tf datasets
    input_tfds = tf.data.Dataset.from_tensor_slices(tags)
    output_tfds = tf.data.Dataset.from_tensor_slices(tags)

    # shuffle the data
    if mode == "training":
        input_tfds = input_tfds.batch(batch_size).shuffle(
            # buffer_size=int(len(tags[0]) / batch_size),
            buffer_size=2 * len(tags[0]),
            reshuffle_each_iteration=True,
            seed=settings["rng_seed"],
        )
        output_tfds = output_tfds.batch(batch_size).shuffle(
            buffer_size=2 * len(tags[0]),
            # buffer_size=int(len(tags[0]) / batch_size),
            reshuffle_each_iteration=True,
            seed=settings["rng_seed"],
        )

    elif mode == "inference":
        input_tfds = input_tfds.batch(batch_size)
        output_tfds = output_tfds.batch(batch_size)

    else:
        raise NotImplementedError("no such mode.")

    # use the mapping function to map sample tags to the data generator functions
    input_tfds = input_tfds.map(
        lambda sample_years, sample_lats, sample_lons, sample_files: tf.py_function(
            data_gen.get_input_data,
            [sample_years, sample_lats, sample_lons, sample_files],
            Tout=tf.float64,
        )
    )
    output_tfds = output_tfds.map(
        lambda sample_years, sample_lats, sample_lons, sample_files: tf.py_function(
            data_gen.get_output_data,
            [sample_years, sample_lats, sample_lons, sample_files],
            Tout=tf.float64,
        )
    )

    tfds_all = tf.data.Dataset.zip((input_tfds, output_tfds))

    return tfds_all


def get_tags(settings):
    if settings["mode"] == "training":
        tags_train, tags_val = get_training_tags(settings)
    elif settings["mode"] == "inference":
        tags_train, tags_val = get_inference_tags(settings)
    else:
        raise NotImplementedError("no such mode.")

    return tags_train, tags_val


def get_training_tags(settings):
    rng = np.random.default_rng(settings["rng_seed"])

    (
        lat_s_bound,
        lat_n_bound,
        lon_w_bound,
        lon_e_bound,
    ) = read_landsat.get_landsat_bounds(settings, region=settings["training_region"])
    print(lat_s_bound, lat_n_bound, lon_w_bound, lon_e_bound)
    print("\n")

    with rasterio.open(DEFAULT_MASK_FILENAME) as buffer_mask:
        sample_lats = []
        sample_lons = []
        sample_years = []

        for latfile in np.arange(
            lat_s_bound + settings["tile_len_deg"],
            lat_n_bound + settings["tile_len_deg"],
            settings["tile_len_deg"],
        ):
            for lonfile in np.arange(
                lon_w_bound, lon_e_bound, settings["tile_len_deg"]
            ):
                # check that landsat file even exists
                landsat_filenames = read_landsat.get_input_filename(
                    settings["training_years"],
                    np.ones(len(settings["training_years"])) * latfile,
                    np.ones(len(settings["training_years"])) * lonfile,
                    settings,
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
                    latfile - settings["tile_len_deg"],
                    latfile,
                    lonfile,
                    lonfile + settings["tile_len_deg"],
                )
                ilat_s, ilat_n, ilon_w, ilon_e = methods.get_tile_indices(
                    buffer_mask, tile_bounds
                )
                ilat_s, ilat_n, ilon_w, ilon_e = methods.trim_hfi_region(
                    (ilat_s, ilat_n, ilon_w, ilon_e),
                    buffer_mask,
                    region=settings["training_region"],
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

                subsample_lons, subsample_lats = buffer_mask.xy(ilat_grid, ilon_grid, offset="ul")
                subsample_lons, subsample_lats = np.asarray(subsample_lons), np.asarray(
                    subsample_lats
                )

                # Remove all possibilities of edges and corners
                edge_width = (
                    (np.ceil((settings["scene_width"] - 1) / 2) + 1)
                    * settings["landsat_to_hfi_ratio"]
                    * settings["landsat_pixel_to_deg"]
                )

                i_nonedge = np.flatnonzero(
                    np.logical_and(
                        np.abs(
                            np.abs(subsample_lons)
                            - np.rint(np.abs(subsample_lons) / settings["tile_len_deg"])
                            * settings["tile_len_deg"]
                        )
                        > edge_width,
                        np.abs(
                            np.abs(subsample_lats)
                            - np.rint(np.abs(subsample_lats) / settings["tile_len_deg"])
                            * settings["tile_len_deg"]
                        )
                        > edge_width,
                    )
                )
                subsample_lats = subsample_lats[i_nonedge]
                subsample_lons = subsample_lons[i_nonedge]

                if len(subsample_lats) < settings["batch_size"]:
                    continue

                nbatches = int(np.ceil(np.sum(settings["nbatches"]) * frac_land))
                nsamples = int(settings["batch_size"] * nbatches)

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
                        settings["training_years"], size=nbatches, replace=True
                    ),
                    settings["batch_size"],
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
    nbatches = int(len(sample_lats) // settings["batch_size"])
    p = settings["nbatches"][0] / np.sum(settings["nbatches"])
    train_bool = np.repeat(
        np.random.choice([0, 1], size=nbatches, replace=True, p=[1.0 - p, p]),
        settings["batch_size"],
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
        tagyear_train, taglat_train, taglon_train, settings
    )
    tagfile_val = read_landsat.get_input_filename(
        tagyear_val, taglat_val, taglon_val, settings
    )

    # Put into nice packages
    tags_train = (tagyear_train, taglat_train, taglon_train, tagfile_train)
    tags_val = (tagyear_val, taglat_val, taglon_val, tagfile_val)

    # PRINT META DATA
    print(
        f"\ntotal training samples = {len(tagyear_train)}, total validation samples = {len(tagyear_val)}\n"
    )

    return tags_train, tags_val


def get_inference_tags(settings):
    with rasterio.open(DEFAULT_MASK_FILENAME) as buffer_mask:
        ilat_s, ilat_n, ilon_w, ilon_e = methods.get_tile_indices(
            buffer_mask, settings["tile"]
        )
        ilat_s, ilat_n, ilon_w, ilon_e = methods.trim_hfi_region(
            (ilat_s, ilat_n, ilon_w, ilon_e),
            buffer_mask,
            region=settings["inference_region"],
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
        np.ones(shape=sample_lats.shape) * settings["inference_years"][0],
        dtype=int,
    )
    tagfile_inf = read_landsat.get_input_filename(
        tagyear_inf, sample_lats, sample_lons, settings
    )

    # PRINT SIZES
    n_inference = tagyear_inf.shape
    print(f"{n_inference = }")
    assert len(n_inference) > 0, "you have no data to predict."

    # Put into a nice package
    tags = (tagyear_inf, sample_lats, sample_lons, tagfile_inf)
    return tags, None


class data_generator:
    # THIS CODE ASSUMES THAT GET_INPUT_DATA() IS CALLED PRIOR TO GET_OUTPUT_DATA
    # EVERY ITERATION BY TENSORFLOW

    # init method or constructor
    def __init__(self, settings, tags_dict):
        self.settings = settings
        self.tags_dict = tags_dict
        self.rng = np.random.default_rng(settings["rng_seed"])

    def get_input_data(self, sample_years, sample_lats, sample_lons, sample_files):
        if self.settings["mode"] == "training":
            # grab random tags associated with the filenames for training only
            tile_key = sample_files[0].numpy().decode("utf8")
            sample_years, sample_lats, sample_lons, sample_files = self.tags_dict[
                tile_key
            ]

            i = self.rng.choice(
                np.arange(0, len(sample_years)),
                self.settings["batch_size"],
                replace=False,
            )
            sample_years, sample_lats, sample_lons, sample_files = (
                sample_years[i],
                sample_lats[i],
                sample_lons[i],
                sample_files[i],
            )

            filename = LANDSAT_DIRECTORY + sample_files[0] + ".tif"

        else:
            sample_years = sample_years.numpy()
            sample_lats = sample_lats.numpy()
            sample_lons = sample_lons.numpy()
            sample_files = sample_files

            filename = (
                LANDSAT_DIRECTORY + sample_files[0].numpy().decode("utf8") + ".tif"
            )

        self.current_tags = (sample_years, sample_lats, sample_lons, sample_files)

        # read landsat file
        try:
            assert all(x == sample_files[0] for x in sample_files), print(sample_files)
        except:
            print(sample_files)
            raise AssertionError("something is wrong. could just be the GPU though.")

        batch_input = np.zeros(
            (
                len(sample_years),
                self.settings["scene_width_landsat"],
                self.settings["scene_width_landsat"],
                len(self.settings["channels"]),
            )
        )
        if not os.path.isfile(filename):
            if self.settings["mode"] == "training":
                raise ValueError("No such input Landsat file: " + filename)
            elif self.settings["mode"] == "inference":
                return tf.convert_to_tensor(batch_input)
            else:
                raise NotImplementedError("no such mode.")

        # intialize tif neighborhood dictionary and loop through samples to get the data
        tif_dict = {}
        tif_dict = read_landsat.fill_tif_dict(
            "central",
            sample_years[0],
            sample_lats[0],
            sample_lons[0],
            tif_dict,
            self.settings,
        )

        for isample in np.arange(0, len(sample_years)):
            try:
                sample_out, tif_dict = read_landsat.read_input_data(
                    self.settings,
                    tif_dict,
                    sample_years[isample],
                    sample_lons[isample],
                    sample_lats[isample],
                    self.settings["channels"],
                    self.settings["scene_width"],
                )
            except:
                sample_out = 0.0  # when we create a generator, make this nan so that the output is set to nan if possible.
            batch_input[isample, :, :, :] = sample_out

        # close tifs in the dictionary
        for key in tif_dict.keys():
            try:
                tif_dict[key].close()
            except:
                pass

        # convert to tensor
        dat = tf.convert_to_tensor(batch_input)

        return dat

    def get_output_data(self, sample_years, sample_lats, sample_lons, sample_files):
        sample_years, sample_lats, sample_lons, sample_files = self.current_tags

        # Get HFI file
        assert all(x == sample_years[0] for x in sample_years)
        filename = DATA_DIRECTORY + "hii_" + str(sample_years[0]) + "-01-01_uint8.tif"

        batch_output = np.zeros((len(sample_years), 1))

        if not os.path.isfile(filename):
            return tf.convert_to_tensor(batch_output * 0.0)

        with rasterio.open(filename) as output_tiff:
            for isample in np.arange(0, len(sample_years)):
                batch_output[isample] = read_output_data(
                    self, output_tiff, sample_lons[isample], sample_lats[isample]
                )

        # convert to tensor
        dat = tf.convert_to_tensor(batch_output)

        return dat


def read_output_data(self, tiff, sample_lon, sample_lat):
    ilat, ilon = tiff.index(sample_lon, sample_lat)
    window = Window(ilon, ilat, 1, 1)

    output_mask = (
        tiff.read_masks(1, window=window) // 255.0
    )  # convert to 0/1, with 0 = no data
    sample_output = output_mask * tiff.read(1, window=window)

    if len(sample_output) == 0:
        sample_output = 0.0

    # this is where we can force the network to predict zeros or ones
    if self.settings["mode"] == "training":
        if sample_output == 0.0:
            sample_output = self.settings["kluge_value_for_zero"]

    return sample_output
