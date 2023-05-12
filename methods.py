"""Methods for doing analysis and processing data

Functions
---------
get_sample_weights(settings, data)
permute_shuffle_sample_list(settings, sample_years, sample_lats, sample_lons)
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy

__author__ = "Elizabeth A. Barnes and Randal J. Barnes"
__date__ = "11 May 2023"


def get_sample_weights(settings, data):

    alpha = settings["sample_weights_alpha"]
    epsilon = .0001

    hist = np.histogram(data, bins=100)
    hist_dist = scipy.stats.rv_histogram(hist, density=True)

    weights = np.maximum(1 - alpha * hist_dist.pdf(data), epsilon)
    weights = weights / np.mean(weights)

    # plt.figure()
    # x_values = np.arange(0, 1.01, .01)
    # plot_weights = np.maximum(1 - alpha * hist_dist.pdf(x_values), epsilon)
    # plot_weights = plot_weights / np.mean(plot_weights)
    # n, bins, __ = plt.hist(data, np.arange(0, 1.01, .01), density=True)
    # plt.plot(x_values, plot_weights)
    # plt.ylim(0, None)

    return weights


def permute_shuffle_sample_list(settings,
                                sample_years, sample_lats, sample_lons,
                                sampling_weights=None):

    if sampling_weights is None:
        sampling_weights = np.ones(sample_years.shape)
    sampling_weights = sampling_weights / np.sum(sampling_weights)

    nsamples = np.sum(settings["nbatches"]) * settings["batch_size"]
    rng = np.random.default_rng(settings["rng_seed"])

    iloc = rng.choice(np.arange(0, sample_lats.shape[0]), size=nsamples, replace=False, p=sampling_weights)
    sample_lats = sample_lats[iloc]
    sample_lons = sample_lons[iloc]

    # sample_years = rng.choice(sample_years, size=nsamples, replace=True)  # THIS IS MUCH SLOWER
    # make it so that every batch has the same year throughout for loading files
    sample_years = np.repeat(np.random.choice(sample_years, size=np.sum(settings["nbatches"]), replace=True),
                             settings["batch_size"])

    return sample_years, sample_lats, sample_lons
