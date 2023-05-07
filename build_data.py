"""Build the data.

Classes
---------
data_generator

Functions
---------
permute_shuffle_sample_list(settings, sample_years, sample_lats, sample_lons)
make_sample_list(settings,)
data_generator.get_x_data(self, years, sample_lats, sample_lons)
data_generator.get_y_data(self, years, sample_lats, sample_lons)
"""
import rasterio
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_sample_weight

DATA_DIRECTORY = "data/"
LANDSAT_TO_HFI_RATIO = 38


def build_tf_dataset(settings, sample_years, sample_lats, sample_lons, batch_size):

    # make data generator
    data_gen = data_generator(settings)

    # create tf datasets
    x_tfds = tf.data.Dataset.from_tensor_slices((sample_years, sample_lats, sample_lons))
    y_tfds = tf.data.Dataset.from_tensor_slices((sample_years, sample_lats, sample_lons))

    # batch the data together so the iterator loops through batches instead of samples
    x_tfds = x_tfds.batch(batch_size).shuffle(buffer_size=int(len(sample_years) / batch_size), reshuffle_each_iteration=False, seed=settings["rng_seed"])
    y_tfds = y_tfds.batch(batch_size).shuffle(buffer_size=int(len(sample_years) / batch_size), reshuffle_each_iteration=False, seed=settings["rng_seed"])

    # use the mapping function to map sample tags to the data generator functions
    x_tfds = x_tfds.map(lambda sample_years, sample_lats, sample_lons:
                        tf.py_function(data_gen.get_x_data, [sample_years, sample_lats, sample_lons],
                                       Tout=tf.float64))
    y_tfds = y_tfds.map(lambda sample_years, sample_lats, sample_lons:
                        tf.py_function(data_gen.get_y_data, [sample_years, sample_lats, sample_lons],
                                       Tout=tf.float64))

    tfds_all = tf.data.Dataset.zip((x_tfds, y_tfds))

    return tfds_all


def permute_shuffle_sample_list(settings, nbatches, batch_size,
                                sample_years, sample_lats, sample_lons,
                                sample_weights=None):

    if sample_weights is None:
        sample_weights = np.ones(sample_years.shape)
    sample_weights = sample_weights / np.sum(sample_weights)

    nsamples = nbatches * batch_size
    rng = np.random.default_rng(settings["rng_seed"])

    iloc = rng.choice(np.arange(0, sample_lats.shape[0]), size=nsamples, replace=False, p=sample_weights)
    sample_lats = sample_lats[iloc]
    sample_lons = sample_lons[iloc]

    # make it so that every batch has the same year throughout for loading files
    sample_years = np.repeat(np.random.choice(sample_years, size=nbatches, replace=True),
                             batch_size)

    return sample_years, sample_lats, sample_lons


def make_sample_list(settings):

    # GET THE LATITUDE AND LONGITUDE LOCATION LISTS
    filename = DATA_DIRECTORY + "hfi" + str(2013) + "_merisINT.epsg4326.tif"  # example file to get bounds

    row_grid, col_grid = np.meshgrid(np.arange(settings["training_ibounds"][0], settings["training_ibounds"][1]),
                                     np.arange(settings["training_ibounds"][2], settings["training_ibounds"][3]),
                                     indexing="xy"
                                     )
    print("output region shape = " + str(row_grid.shape))

    output_tiff = rasterio.open(filename)
    sample_lons, sample_lats = output_tiff.xy(np.ndarray.flatten(row_grid), np.ndarray.flatten(col_grid))
    sample_lats, sample_lons = np.asarray(sample_lats), np.asarray(sample_lons)

    # remove water locations for training (as determined by the 2013 map)
    # we could have read the mask instead using >> output_mask = output_tiff.read_masks(1)
    hfi = output_tiff.read(1)[settings["training_ibounds"][0]:settings["training_ibounds"][1], settings["training_ibounds"][2]:settings["training_ibounds"][3]]
    hfi = np.ndarray.flatten(hfi)
    iland = np.where(hfi != 255)[0]
    hfi, sample_lats, sample_lons = hfi[iland], sample_lats[iland], sample_lons[iland]

    # subsample by half-decile (since data out of 50) to account for class imbalance
    sample_weights = compute_sample_weight(class_weight="balanced", y=np.round(hfi / 5) * 5)

    # SHUFFLE TOGETHER and BATCH BY EQUAL YEAR
    sample_years = np.asarray(settings["training_years"], dtype=int)

    sample_years, sample_lats, sample_lons = permute_shuffle_sample_list(settings,
                                                                         np.sum(settings["nbatches"]),
                                                                         settings["batch_size"],
                                                                         sample_years,
                                                                         sample_lats,
                                                                         sample_lons,
                                                                         sample_weights)
    # CHECK IF UNIQUE
    unq, counts = np.unique([sample_years, sample_lats, sample_lons], axis=1, return_counts=True)
    percent_repeated = np.round(100. * np.sum(counts > 1) / len(sample_years), 2)
    print(f"percent repeated samples in training and validation = {percent_repeated}%")

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

    def get_x_data(self, years, sample_lats, sample_lons):

        assert np.sum(years - years[0]) == 0
        year = years[0]  # all years are the same for each batch

        channels = self.settings["channels"]
        scene_width = self.settings["scene_width"]

        # read landsat file
        filename = DATA_DIRECTORY + "/landsat_export_1x1/" + str(year.numpy()) + "_06S_106E.tif"
        filename_mask = DATA_DIRECTORY + "/landsat_export_1x1/" + str(year.numpy()) + "_06S_106E_mask.tif"

        batch_x = np.zeros((len(years), scene_width * LANDSAT_TO_HFI_RATIO, scene_width * LANDSAT_TO_HFI_RATIO, len(channels)))
        with rasterio.open(filename) as input_tiff:
            with rasterio.open(filename_mask) as input_mask:
                for isample in np.arange(0, len(years)):
                    row, col = input_tiff.index(sample_lons[isample], sample_lats[isample])

                    row0, row1 = row[0] - LANDSAT_TO_HFI_RATIO * ((scene_width - 1) // 2 + 1), row[0] + LANDSAT_TO_HFI_RATIO * (scene_width - 1) // 2
                    col0, col1 = col[0] - LANDSAT_TO_HFI_RATIO * (scene_width - 1) // 2, col[0] + LANDSAT_TO_HFI_RATIO * ((scene_width - 1) // 2 + 1)

                    batch_x[isample, :, :, :] = np.transpose(input_mask.read((1))[row0:row1, col0:col1] * input_tiff.read(channels)[:, row0:row1, col0:col1] / 255., axes=(1, 2, 0))

        # convert to tensor
        dat = tf.convert_to_tensor(batch_x)

        return dat

    def get_y_data(self, years, sample_lats, sample_lons):

        assert np.sum(years - years[0]) == 0
        year = years[0]  # all years are the same for each batch

        # read HFI file
        filename = DATA_DIRECTORY + "hfi" + str(year.numpy()) + "_merisINT.epsg4326.tif"

        batch_y = np.zeros((len(years), 1))
        with rasterio.open(filename) as output_tiff:
            output_mask = output_tiff.read_masks(1) // 255

            for isample in np.arange(0, len(years)):
                row, col = output_tiff.index(sample_lons[isample], sample_lats[isample])
                batch_y[isample, :] = output_mask[row, col] * output_tiff.read(1)[row, col] / 50.

        # convert to tensor
        dat = tf.convert_to_tensor(batch_y)

        return dat
