"""Methods for doing analysis and processing data

Functions
---------
get_sample_weights(settings, data)
permute_shuffle_sample_list(settings, sample_years, sample_lats, sample_lons)
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy
import tensorflow as tf
import methods
import rasterio

__author__ = "Elizabeth A. Barnes and Randal J. Barnes"
__date__ = "11 May 2023"


DEFAULT_FILENAME = "hii_2020-01-01_uint8.tif"


def get_directories():
    dir_dict = {
        "data_dir": "data/",
        "landsat_dir": "data/landsat_export_1x1/",
        "predictions_dir": "predictions/",
        "figures_dir": "figures/",
        "save_model_dir": "saved_models/",
        "mosaics_dir": "mosaics/",
    }
    return dir_dict


def remove_nodata(x, y=None):

    assert np.shape(x) == np.shape(y)

    if y is None:
        index = np.where(x != 255)[0]
        return x[index]
    else:
        index = np.where(x != 255)[0]
        x, y = x[index], y[index]

        index = np.where(y != 255)[0]
        x, y = x[index], y[index]

        return x, y


def get_tile_indices(hfi_tif, tile):
    # tile == lat_s, lat_n, lon_w, lon_e

    ilat_n, ilon_w = hfi_tif.index(tile[2], tile[1])
    ilat_s, ilon_e = hfi_tif.index(tile[3], tile[0])

    indices = ilat_s, ilat_n, ilon_w, ilon_e

    return trim_hfi_region(indices, hfi_tif, tile)


def trim_hfi_region(indices, hfi_tif, region):

    ilat_s, ilat_n, ilon_w, ilon_e = indices

    lat_indices = np.arange(ilat_n, ilat_s + 1)
    lon_indices = np.arange(ilon_w, ilon_e + 1)

    lons, __ = hfi_tif.xy(np.zeros(lon_indices.shape), lon_indices)
    __, lats = hfi_tif.xy(lat_indices, np.zeros(lat_indices.shape))
    lons, lats = np.asarray(lons), np.asarray(lats)

    # get within tile bounds
    ilat_indices = np.where((lats > region[0]) & (lats <= region[1]))[0]
    ilon_indices = np.where((lons >= region[2]) & (lons < region[3]))[0]

    # grab indices that are still in-play
    ilat_s = np.max(lat_indices[ilat_indices])
    ilat_n = np.min(lat_indices[ilat_indices])
    ilon_w = np.min(lon_indices[ilon_indices])
    ilon_e = np.max(lon_indices[ilon_indices])

    return ilat_s, ilat_n, ilon_w, ilon_e


def get_denseweight_dist(settings, data):
    # see Steininger et al. (2021)
    # https://link.springer.com/article/10.1007/s10994-021-06023-5

    epsilon = 0.001
    alpha = settings["sample_weights_alpha"]
    x_values = np.arange(0, 101, 1)

    hist = np.histogram(data, bins=x_values)
    hist_dist = scipy.stats.rv_histogram(hist, density=True)
    hist_dist = hist_dist.pdf(x_values)

    denseweight_dist = np.maximum(1. - alpha * hist_dist, epsilon)
    denseweight_dist = denseweight_dist / np.mean(denseweight_dist)

    return denseweight_dist


def get_denseweights(settings, tags):
    with rasterio.open(get_directories()["data_dir"] + DEFAULT_FILENAME) as tif:
        sample_lats = tags[1]
        sample_lons = tags[2]
        data = tif.sample([*zip(sample_lons, sample_lats)], indexes=1)
        data = np.ndarray.flatten(np.asarray(list(data)))

        return get_denseweight_dist(settings, data)


def dw_calculator(denseweight_dist, data):

    # scaled_data = tf.cast(tf.math.round(100. * data), dtype=tf.int32)
    scaled_data = tf.cast(tf.math.round(data), dtype=tf.int32)
    return tf.gather(denseweight_dist, scaled_data)


class DenseWeight_Loss(tf.keras.losses.Loss):
    def __init__(self, denseweight_dist, power=2):
        super().__init__()
        self.denseweight_dist = denseweight_dist
        self.power = power

    def call(self, y_true, y_pred):

        # loss = tf.math.squared_difference(y_true, y_pred)

        loss = tf.math.difference(y_true, y_pred)
        loss = tf.math.pow(loss, self.power)

        weights = dw_calculator(self.denseweight_dist, y_true)
        loss = tf.multiply(loss, weights)

        loss = tf.reduce_mean(loss)

        return tf.sqrt(loss)
