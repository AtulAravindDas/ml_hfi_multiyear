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
        "save_model_dir": "saved_models/"
    }
    return dir_dict


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
    def __init__(self, denseweight_dist):
        super().__init__()
        self.denseweight_dist = denseweight_dist

    def call(self, y_true, y_pred):

        loss = tf.math.squared_difference(y_true, y_pred)

        weights = dw_calculator(self.denseweight_dist, y_true)
        loss = tf.multiply(loss, weights)

        loss = tf.reduce_mean(loss)

        return tf.sqrt(loss)


# def permute_shuffle_sample_list(settings,
#                                 sample_years, sample_lats, sample_lons,
#                                 sampling_weights=None):

#     if sampling_weights is None:
#         sampling_weights = np.ones(sample_years.shape)
#     sampling_weights = sampling_weights / np.sum(sampling_weights)

#     nsamples = np.sum(settings["nbatches"]) * settings["batch_size"]
#     rng = np.random.default_rng(settings["rng_seed"])

#     iloc = rng.choice(np.arange(0, sample_lats.shape[0]), size=nsamples, replace=False, p=sampling_weights)
#     sample_lats = sample_lats[iloc]
#     sample_lons = sample_lons[iloc]

#     # sample_years = rng.choice(sample_years, size=nsamples, replace=True)  # THIS IS MUCH SLOWER
#     # make it so that every batch has the same year throughout for loading files
#     sample_years = np.repeat(np.random.choice(sample_years, size=np.sum(settings["nbatches"]), replace=True),
#                              settings["batch_size"])

#     return sample_years, sample_lats, sample_lons
