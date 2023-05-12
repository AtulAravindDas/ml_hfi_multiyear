"""Build the data.

Classes
---------
data_generator

Functions
---------
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
from methods import permute_shuffle_sample_list

DATA_DIRECTORY = "data/"
LANDSAT_DIRECTORY = "data/landsat_export_1x1/"
PREDICTIONS_DIRECTORY = "predictions/"


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


def make_sample_list(settings, evaluate_all=False):

    # GET THE LATITUDE AND LONGITUDE LOCATION LISTS
    filename = DATA_DIRECTORY + "hii_2020-01-01_uint8.tif"
    filename_mask = DATA_DIRECTORY + "hii_coastal_buffer_mask.tif"

    with rasterio.open(filename) as output_tiff:
        with rasterio.open(filename_mask) as buffer_mask:
            ilat0, ilon0 = buffer_mask.index(settings["latlon_bounds"][2],
                                             settings["latlon_bounds"][0])
            ilat1, ilon1 = buffer_mask.index(settings["latlon_bounds"][3],
                                             settings["latlon_bounds"][1])

            ilat_grid, ilon_grid = np.meshgrid(np.arange(ilat0, ilat1 + 1), np.arange(ilon0, ilon1 + 1), indexing="ij")
            print("output region shape = " + str(ilon_grid.shape))

            sample_lons, sample_lats = buffer_mask.xy(np.ndarray.flatten(ilat_grid), np.ndarray.flatten(ilon_grid))
            sample_lons, sample_lats = np.asarray(sample_lons), np.asarray(sample_lats)

            # load HFI data
            window = Window.from_slices((ilat0, ilat1 + 1), (ilon0, ilon1 + 1))
            hfi_mask = buffer_mask.read(1, window=window)
            hfi = output_tiff.read(1, window=window)

            hfi = np.ndarray.flatten(hfi)
            hfi_mask = np.ndarray.flatten(hfi_mask)

            # remove oceanic locations for training
            hfi = np.where(hfi == 255., 0., hfi)
            iland = np.where(hfi_mask == 1)[0]
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
            # subsample by decile to account for class imbalance
            if settings["subsample"]:
                sampling_weights = compute_sample_weight(class_weight="balanced", y=np.round(hfi / 10) * 10)
            else:
                sampling_weights = np.ones(np.shape(hfi))

            denseweight_dist = methods.get_denseweight_dist(settings, hfi)

            # SHUFFLE TOGETHER and BATCH BY EQUAL YEAR
            # <TO DO> will need to batch by equal tile at a later date
            sample_years = np.asarray(settings["training_years"], dtype=int)
            (sample_years,
             sample_lats,
             sample_lons,
             ) = permute_shuffle_sample_list(settings, sample_years, sample_lats, sample_lons, sampling_weights)

            # CHECK IF UNIQUE (this should always be true)
            # __, counts = np.unique([sample_years, sample_lats, sample_lons], axis=1, return_counts=True)
            # assert np.sum(counts > 1) == 0.

            # SPLIT INTO TRAINING AND VALIDATION
            ntrain = settings["nbatches"][0] * settings["batch_size"]
            tagyear_train, taglat_train, taglon_train = sample_years[:ntrain], sample_lats[:ntrain], sample_lons[:ntrain]
            assert len(tagyear_train) == ntrain

            nval = settings["nbatches"][1] * settings["batch_size"]
            tagyear_val, taglat_val, taglon_val = sample_years[ntrain:ntrain + nval], sample_lats[ntrain:ntrain + nval], sample_lons[ntrain:ntrain + nval]
            assert len(tagyear_val) == nval

            # PRINT SIZES
            print(f"{ntrain = }, {nval = }")

            return tagyear_train, taglat_train, taglon_train, tagyear_val, taglat_val, taglon_val, denseweight_dist


def save_predictions_tif(settings, hfi_predict, predictions_filename):

    # GET TIFF META DATA
    labels_filename = DATA_DIRECTORY + "hii_" + str(settings["testing_year"]) + "-01-01_uint8.tif"
    filename_mask = DATA_DIRECTORY + "hii_coastal_buffer_mask.tif"

    with rasterio.open(filename_mask) as buffer_mask:
        ilat0, ilon0 = buffer_mask.index(settings["latlon_bounds"][2], settings["latlon_bounds"][0])
        ilat1, ilon1 = buffer_mask.index(settings["latlon_bounds"][3], settings["latlon_bounds"][1])
        lon0, lat0 = buffer_mask.xy(ilat0, ilon0)
        lon1, lat1 = buffer_mask.xy(ilat1, ilon1)

        window = Window.from_slices((ilat0, ilat1 + 1), (ilon0, ilon1 + 1))
        hfi_mask = buffer_mask.read(1, window=window)

    if os.path.isfile(labels_filename):
        with rasterio.open(labels_filename) as labels_tiff:
            ilat0, ilon0 = labels_tiff.index(settings["latlon_bounds"][2], settings["latlon_bounds"][0])
            ilat1, ilon1 = labels_tiff.index(settings["latlon_bounds"][3], settings["latlon_bounds"][1])
            window = Window.from_slices((ilat0, ilat1 + 1), (ilon0, ilon1 + 1))
            hfi_labels = labels_tiff.read(1, window=window)
    else:
        hfi_labels = np.zeros(np.shape(hfi_predict)) * np.nan

    # SAVE THE TIFF
    width = ilon1 - ilon0 + 1
    height = ilat1 - ilat0 + 1
    res_lat = ((lat1 - lat0)) / (height - 1)
    res_lon = ((lon1 - lon0)) / (width - 1)

    hfi_labels = np.reshape(hfi_labels, (height, width), order="C")
    hfi_predict = np.reshape(hfi_predict, (height, width), order="C")
    hfi_predict = np.asarray(np.round(hfi_predict), dtype="uint8")
    hfi_predict = np.where(hfi_mask == 1, hfi_predict, 255)  # remove ocean and turn to nan

    meta_data = {}
    meta_data["nodata"] = 255
    meta_data["width"] = width
    meta_data["height"] = height
    meta_data["driver"] = 'GTiff'
    meta_data["count"] = 1
    meta_data["crs"] = rasterio.CRS.from_epsg(4326)
    meta_data["dtype"] = hfi_predict.dtype
    meta_data["transform"] = Affine.translation(lon0, lat0) * Affine.scale(res_lon, res_lat)

    with rasterio.open(PREDICTIONS_DIRECTORY + predictions_filename + ".tif", "w", **meta_data) as dst:
        dst.write(hfi_predict, 1)
        dst.set_band_description(1, 'mlHFI prediction')

    return hfi_predict, hfi_labels, lat0, lat1, lon0, lon1


class data_generator:

    # init method or constructor
    def __init__(self, settings):
        self.settings = settings

    def get_input_data(self, years, sample_lats, sample_lons):

        assert np.sum(years - years[0]) == 0

        channels = self.settings["channels"]
        scene_width = self.settings["scene_width"]
        rng = np.random.default_rng()

        # read landsat file
        batch_input = np.zeros((len(years), scene_width, scene_width, len(channels)))

        filename = LANDSAT_DIRECTORY + "landsat_" + self.settings["tilename"] + "_" + str(years[0].numpy()) + ".tif"

        with rasterio.open(filename) as input_tiff:

            for isample in np.arange(0, len(years)):
                ilat, ilon = input_tiff.index(sample_lons[isample], sample_lats[isample])

                # WRONG
                # ilat0, ilat1 = ilat[0] - scene_width / 3 * 2 + 1, ilat[0] + scene_width / 3
                # ilon0, ilon1 = ilon[0] - scene_width / 3, ilon[0] + scene_width / 3 * 2 - 1

                # CORRECT - MAYBE
                # ilat0, ilat1 = ilat[0] - scene_width / 3, ilat[0] + scene_width / 3 * 2 - 1
                # ilon0, ilon1 = ilon[0] - scene_width / 3, ilon[0] + scene_width / 3 * 2 - 1

                ilat0, ilat1 = ilat[0] - scene_width / 3 * 2, ilat[0] + scene_width / 3 - 1
                ilon0, ilon1 = ilon[0] - scene_width / 3 * 2, ilon[0] + scene_width / 3 - 1

                window = Window.from_slices((ilat0, ilat1 + 1), (ilon0, ilon1 + 1))
                input_scene = np.transpose(input_tiff.read(channels, window=window), axes=(1, 2, 0))

                # add noise to de-noise
                if self.settings["training"]:
                    random_noise = rng.integers(-self.settings["input_noise"], self.settings["input_noise"] + 1, size=1)
                    batch_input[isample, :, :, :] = (input_scene + random_noise)
                else:
                    batch_input[isample, :, :, :] = input_scene

        # convert to tensor
        dat = tf.convert_to_tensor(batch_input)

        return dat

    def get_output_data(self, years, sample_lats, sample_lons):

        assert np.sum(years - years[0]) == 0

        batch_output = np.zeros((len(years), 1))

        # Get HFI file
        filename = DATA_DIRECTORY + "hii_" + str(years[0].numpy()) + "-01-01_uint8.tif"
        if not os.path.isfile("filename"):
            # print("** Loading 2020 HFI values for labels. This is not compatible with training! **")
            filename = DATA_DIRECTORY + "hii_2020-01-01_uint8.tif"

        with rasterio.open(filename) as output_tiff:
            for isample in np.arange(0, len(years)):

                ilat, ilon = output_tiff.index(sample_lons[isample], sample_lats[isample])
                window = Window(ilon[0], ilat[0], 1, 1)

                output_mask = output_tiff.read_masks(1, window=window) // 255.  # convert to 0/1, with 0 = no data
                batch_output[isample] = output_mask * output_tiff.read(1, window=window)

                # this is where we can force the network to predict zeros
                if self.settings["training"]:
                    if batch_output[isample] == 0.:
                        batch_output[isample] = self.settings["kluge_value_for_zero"]

                # batch_output[isample] = read_output_data(self, output_tiff, sample_lons[isample], sample_lats[isample])

        # convert to tensor
        dat = tf.convert_to_tensor(batch_output)

        return dat


def read_output_data(self, tiff, sample_lon, sample_lat):

    ilat, ilon = tiff.index(sample_lon, sample_lat)
    window = Window(ilon[0], ilat[0], 1, 1)

    output_mask = tiff.read_masks(1, window=window) // 255.  # convert to 0/1, with 0 = no data
    sample_output = output_mask * tiff.read(1, window=window)

    # this is where we can force the network to predict zeros
    if self.settings["training"]:
        if sample_output == 0.:
            sample_output = self.settings["kluge_value_for_zero"]

    return sample_output
