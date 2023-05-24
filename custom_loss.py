"""Loss functions for training the network.

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


DEFAULT_FILENAME = "hii_2015-01-01_uint8.tif"


def get_denseweight_dist(settings, data):
    # see Steininger et al. (2021)
    # https://link.springer.com/article/10.1007/s10994-021-06023-5

    epsilon = 0.001
    alpha = settings["sample_weights_alpha"]
    x_values = np.arange(0, 101, 1)

    hist = np.histogram(data, bins=x_values)
    hist_dist = scipy.stats.rv_histogram(hist, density=True)
    hist_dist = hist_dist.pdf(x_values)

    denseweight_dist = np.maximum(1.0 - alpha * hist_dist, epsilon)
    denseweight_dist = denseweight_dist / np.mean(denseweight_dist)

    if "extra_denseweight_high" in settings.keys():
        high_index = settings["extra_denseweight_high"][0]
        denseweight_dist[high_index:] = (
            denseweight_dist[high_index:] * settings["extra_denseweight_high"][1]
        )

    return denseweight_dist


def get_denseweights(settings, tags):
    with rasterio.open(methods.get_directories()["data_dir"] + DEFAULT_FILENAME) as tif:
        sample_lats = tags[1]
        sample_lons = tags[2]
        data = tif.sample([*zip(sample_lons, sample_lats)], indexes=1)
        data = np.ndarray.flatten(np.asarray(list(data)))

        return get_denseweight_dist(settings, data)


def apply_denseweights(denseweight_dist, data):
    # scaled_data = tf.cast(tf.math.round(100. * data), dtype=tf.int32)
    scaled_data = tf.cast(tf.math.round(data), dtype=tf.int32)
    return tf.gather(denseweight_dist, scaled_data)


class DenseWeightMSE_Loss(tf.keras.losses.Loss):
    def __init__(self, denseweight_dist):
        super().__init__()
        self.denseweight_dist = denseweight_dist

    def call(self, y_true, y_pred):
        loss = tf.math.squared_difference(y_true, y_pred)
        weights = apply_denseweights(self.denseweight_dist, y_true)

        loss = tf.multiply(loss, weights)
        loss = tf.reduce_mean(loss)

        return tf.sqrt(loss)


class DenseDualWeightMSE_Loss(tf.keras.losses.Loss):
    # NOTE: to also focus on zero, one could subtract 50 from the predictions
    # and truth and then use this loss as currently written

    # got this idea from:
    # R. Lagerquist, D. Turner, I. Ebert-Uphoff, J. Stewart, and V. Hagerty,
    # “Using deep learning to emulate and accelerate a radiative-transfer model,”
    # Journal of Atmospheric and Oceanic Technology, vol. conditionally accepted, 2021.

    def __init__(self, denseweight_dist, params):
        super().__init__()
        self.denseweight_dist = denseweight_dist
        self.gamma_weight = params[0]
        self.offset = params[1]
        self.offset_width = params[2]

    def call(self, y_true, y_pred):
        loss = tf.math.squared_difference(y_true, y_pred)

        weights = apply_denseweights(self.denseweight_dist, y_true)
        loss = tf.multiply(loss, weights)

        yt = tf.math.maximum(tf.math.abs(y_true - self.offset), self.offset_width)
        yp = tf.math.maximum(tf.math.abs(y_pred - self.offset), self.offset_width)

        weights = (tf.math.maximum(yt, yp)) ** self.gamma_weight
        loss = tf.multiply(loss, weights)

        return tf.sqrt(tf.reduce_mean(loss))
