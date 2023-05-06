"""Build the data.

Functions
---------
scheduler(epoch, lr)
train_model(settings, model)
"""
import rasterio
import numpy as np
import tensorflow as tf

DATA_DIRECTORY = "data/"
LANDSAT_TO_HFI_RATIO = 38


def get_shuffled_batch(settings,):

    filename = DATA_DIRECTORY + "hfp" + str(2013) + "_merisINT.epsg4326.tif"

    output_tiff = rasterio.open(filename)
    print(np.shape(output_tiff))

    # row_grid, col_grid = np.meshgrid(np.arange(0,np.shape(output_tiff)[1]),
    #                                  np.arange(0,np.shape(output_tiff)[0]),
    #                                  indexing="xy"
    #                                  )
    row_grid, col_grid = np.meshgrid(np.arange(settings["training_ibounds"][0], settings["training_ibounds"][1]),
                                     np.arange(settings["training_ibounds"][2], settings["training_ibounds"][3]),
                                     indexing="xy"
                                     )

    sample_lons, sample_lats = output_tiff.xy(np.ndarray.flatten(row_grid), np.ndarray.flatten(col_grid))
    sample_years = np.ones(np.shape(sample_lons)) * 2013
    sample_years, sample_lats, sample_lons = np.asarray(sample_years, dtype=int), np.asarray(sample_lats), np.asarray(sample_lons)

    i_locs = np.arange(0, len(sample_lons))
    rng = np.random.default_rng(settings["rng_seed"])
    rng.shuffle(i_locs)

    nsamples = np.sum(settings["n_batches"]) * settings["batch_size"]
    sample_years = sample_years[i_locs[:nsamples]]
    sample_lats = sample_lats[i_locs[:nsamples]]
    sample_lons = sample_lons[i_locs[:nsamples]]

    return sample_years, sample_lats, sample_lons


class data_generator:

    # init method or constructor
    def __init__(self, settings):
        self.settings = settings

    def get_x_data(self, years, sample_lats, sample_lons):

        year = years[0]  # all simulations are the same for each batch based on how we've set it up
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

        year = years[0]  # all simulations are the same for each batch based on how we've set it up

        # read HFI file
        filename = DATA_DIRECTORY + "hfp" + str(year.numpy()) + "_merisINT.epsg4326.tif"

        batch_y = np.zeros((len(years), 1))
        with rasterio.open(filename) as output_tiff:
            output_mask = output_tiff.read_masks(1) // 255

            for isample in np.arange(0, len(years)):
                row, col = output_tiff.index(sample_lons[isample], sample_lats[isample])
                batch_y[isample, :] = output_mask[row, col] * output_tiff.read(1)[row, col] / 50.

        # convert to tensor
        dat = tf.convert_to_tensor(batch_y)

        return dat
