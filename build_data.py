"""Build the data.

Classes
---------
data_generator

Functions
---------
build_tf_dataset(settings, sample_years, sample_lats, sample_lons, batch_size)
permute_shuffle_sample_list(settings, sample_years, sample_lats, sample_lons)
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

DATA_DIRECTORY = "data/"
LANDSAT_DIRECTORY = "data/landsat_export_1x1/"
LANDSAT_TO_HFI_RATIO = 38


def build_tf_dataset(settings, sample_years, sample_lats, sample_lons, batch_size, shuffle=True):

    # make data generator
    data_gen = data_generator(settings)

    # create tf datasets
    input_tfds = tf.data.Dataset.from_tensor_slices((sample_years, sample_lats, sample_lons))
    output_tfds = tf.data.Dataset.from_tensor_slices((sample_years, sample_lats, sample_lons))

    # batch the data together so the iterator loops through batches instead of samples
    if shuffle:
        input_tfds = input_tfds.batch(batch_size).shuffle(buffer_size=int(len(sample_years) / batch_size), reshuffle_each_iteration=True, seed=settings["rng_seed"])
        output_tfds = output_tfds.batch(batch_size).shuffle(buffer_size=int(len(sample_years) / batch_size), reshuffle_each_iteration=True, seed=settings["rng_seed"])
    else:
        input_tfds = input_tfds.batch(batch_size)
        output_tfds = output_tfds.batch(batch_size)

    # use the mapping function to map sample tags to the data generator functions
    input_tfds = input_tfds.map(lambda sample_years, sample_lats, sample_lons:
                                tf.py_function(data_gen.get_input_data, [sample_years, sample_lats, sample_lons], Tout=tf.float64))
    output_tfds = output_tfds.map(lambda sample_years, sample_lats, sample_lons:
                                  tf.py_function(data_gen.get_output_data, [sample_years, sample_lats, sample_lons], Tout=tf.float64))

    tfds_all = tf.data.Dataset.zip((input_tfds, output_tfds))

    return tfds_all


def permute_shuffle_sample_list(settings,
                                sample_years, sample_lats, sample_lons,
                                sample_weights=None):

    if sample_weights is None:
        sample_weights = np.ones(sample_years.shape)
    sample_weights = sample_weights / np.sum(sample_weights)

    nsamples = np.sum(settings["nbatches"]) * settings["batch_size"]
    rng = np.random.default_rng(settings["rng_seed"])

    iloc = rng.choice(np.arange(0, sample_lats.shape[0]), size=nsamples, replace=False, p=sample_weights)
    sample_lats = sample_lats[iloc]
    sample_lons = sample_lons[iloc]

    # make it so that every batch has the same year throughout for loading files
    sample_years = np.repeat(np.random.choice(sample_years, size=np.sum(settings["nbatches"]), replace=True),
                             settings["batch_size"])

    return sample_years, sample_lats, sample_lons


def make_sample_list(settings, evaluate_all=False):

    # GET THE LATITUDE AND LONGITUDE LOCATION LISTS
    filename = DATA_DIRECTORY + "hfi2010_merisINT.epsg4326.tif"  # example file to get bounds and example hfi frequencies

    with rasterio.open(filename) as output_tiff:
        ilat0, ilon0 = output_tiff.index(settings["latlon_bounds"][2],
                                         settings["latlon_bounds"][0])
        ilat1, ilon1 = output_tiff.index(settings["latlon_bounds"][3],
                                         settings["latlon_bounds"][1])

        ilat_grid, ilon_grid = np.meshgrid(np.arange(ilat0, ilat1 + 1), np.arange(ilon0, ilon1 + 1), indexing="ij")
        print("output region shape = " + str(ilon_grid.shape))

        sample_lons, sample_lats = output_tiff.xy(np.ndarray.flatten(ilat_grid), np.ndarray.flatten(ilon_grid))
        sample_lons, sample_lats = np.asarray(sample_lons), np.asarray(sample_lats)

        # load HFI data
        window = Window.from_slices((ilat0, ilat1 + 1), (ilon0, ilon1 + 1))
        hfi = output_tiff.read(1, window=window)
        hfi = np.ndarray.flatten(hfi)

        # remove water locations for training (as determined by the 2010 map)
        hfi = np.where(hfi == 255, 0., hfi)
        iland = np.where(hfi != 255)[0]
        hfi, sample_lats, sample_lons = hfi[iland], sample_lats[iland], sample_lons[iland]

    if evaluate_all:
        sample_lons, sample_lats = output_tiff.xy(np.ndarray.flatten(ilat_grid, order="C"), np.ndarray.flatten(ilon_grid, order="C"))
        sample_lons, sample_lats = np.asarray(sample_lons), np.asarray(sample_lats)
        tagyear_test = np.asarray(np.ones(shape=sample_lats.shape) * settings["testing_year"], dtype=int)

        # PRINT SIZES
        ntest = tagyear_test.shape
        print(f"{ntest = }")

        return tagyear_test, sample_lats, sample_lons

    else:
        # subsample by half-decile (since original HFI data is 0-50) to account for class imbalance
        sample_weights = compute_sample_weight(class_weight="balanced", y=np.round(hfi / 5) * 5)

        # SHUFFLE TOGETHER and BATCH BY EQUAL YEAR
        # <TO DO> will need to batch by equal tile at a later date
        sample_years = np.asarray(settings["training_years"], dtype=int)
        sample_years, sample_lats, sample_lons = permute_shuffle_sample_list(settings,
                                                                             sample_years,
                                                                             sample_lats,
                                                                             sample_lons,
                                                                             sample_weights)
        # CHECK IF UNIQUE (this should always be true)
        __, counts = np.unique([sample_years, sample_lats, sample_lons], axis=1, return_counts=True)
        assert np.sum(counts > 1) == 0.

        # SPLIT INTO TRAINING AND VALIDATION
        ntrain = settings["nbatches"][0] * settings["batch_size"]
        tagyear_train, taglat_train, taglon_train = sample_years[:ntrain], sample_lats[:ntrain], sample_lons[:ntrain]
        assert len(tagyear_train) == ntrain

        nval = settings["nbatches"][1] * settings["batch_size"]
        tagyear_val, taglat_val, taglon_val = sample_years[ntrain:ntrain + nval], sample_lats[ntrain:ntrain + nval], sample_lons[ntrain:ntrain + nval]
        assert len(tagyear_val) == nval

        # PRINT SIZES
        print(f"{ntrain = }, {nval = }")

        return tagyear_train, taglat_train, taglon_train, tagyear_val, taglat_val, taglon_val


class data_generator:

    # init method or constructor
    def __init__(self, settings):
        self.settings = settings

    def get_input_data(self, years, sample_lats, sample_lons):

        assert np.sum(years - years[0]) == 0
        year = years[0]  # all years are the same for each batch

        channels = self.settings["channels"]
        scene_width = self.settings["scene_width"]

        # read landsat file
        filename = LANDSAT_DIRECTORY + str(year.numpy()) + "_" + self.settings["tilename"] + ".tif"
        filename_mask = LANDSAT_DIRECTORY + str(year.numpy()) + "_" + self.settings["tilename"] + "_mask.tif"

        batch_input = np.zeros((len(years), scene_width * LANDSAT_TO_HFI_RATIO, scene_width * LANDSAT_TO_HFI_RATIO, len(channels)))
        with rasterio.open(filename) as input_tiff:
            with rasterio.open(filename_mask) as input_mask:
                for isample in np.arange(0, len(years)):
                    ilat, ilon = input_tiff.index(sample_lons[isample], sample_lats[isample])

                    ilat0, ilat1 = ilat[0] - LANDSAT_TO_HFI_RATIO * 2 + 1, ilat[0] + LANDSAT_TO_HFI_RATIO
                    ilon0, ilon1 = ilon[0] - LANDSAT_TO_HFI_RATIO, ilon[0] + LANDSAT_TO_HFI_RATIO * 2 - 1

                    window = Window.from_slices((ilat0, ilat1 + 1), (ilon0, ilon1 + 1))
                    batch_input[isample, :, :, :] = np.transpose(input_mask.read(1, window=window) * input_tiff.read(channels, window=window) / 255.,
                                                                 axes=(1, 2, 0))

        # convert to tensor
        dat = tf.convert_to_tensor(batch_input)

        return dat

    def get_output_data(self, years, sample_lats, sample_lons):

        assert np.sum(years - years[0]) == 0
        year = years[0]  # all years are the same for each batch

        # read HFI file
        filename = DATA_DIRECTORY + "hfi" + str(year.numpy()) + "_merisINT.epsg4326.tif"
        if not os.path.isfile("filename"):
            # print("** Loading 2010 HFI values for labels. This is not compatible with training! **")
            filename = DATA_DIRECTORY + "hfi" + str(2010) + "_merisINT.epsg4326.tif"

        batch_output = np.zeros((len(years), 1))
        with rasterio.open(filename) as output_tiff:
            for isample in np.arange(0, len(years)):

                ilat, ilon = output_tiff.index(sample_lons[isample], sample_lats[isample])
                window = Window(ilon[0], ilat[0], 1, 1)

                output_mask = output_tiff.read_masks(1, window=window) // 255.  # convert to 0/1, with 0 = no data
                batch_output[isample] = output_mask * output_tiff.read(1, window=window) / 50.

                # this is where we can force the network to predict zeros
                if batch_output[isample] == 0.:
                    batch_output[isample] = self.settings["kluge_value_for_zero"]

        # convert to tensor
        dat = tf.convert_to_tensor(batch_output)

        return dat
