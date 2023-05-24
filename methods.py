"""Methods for doing analysis and processing data

Functions
---------
get_sample_weights(settings, data)
permute_shuffle_sample_list(settings, sample_years, sample_lats, sample_lons)
"""
import numpy as np
import matplotlib.pyplot as plt
import rasterio

__author__ = "Elizabeth A. Barnes and Randal J. Barnes"
__date__ = "11 May 2023"


def get_directories():
    dir_dict = {
        "data_dir": "data/",
        "landsat_dir": "data/landsat_export_1x1/",
        "predictions_dir": "predictions/",
        "figures_dir": "figures/",
        "save_model_dir": "saved_models/",
        "mosaics_dir": "mosaics/",
        "shapefiles_dir": "data/shapefiles/",
    }
    return dir_dict


def remove_nodata(x, y=None):

    if y is None:
        index = np.where(x != 255)[0]
        return x[index]
    else:
        assert np.shape(x) == np.shape(y)

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

    lons, __ = hfi_tif.xy(np.zeros(lon_indices.shape), lon_indices, offset="ul")
    __, lats = hfi_tif.xy(lat_indices, np.zeros(lat_indices.shape), offset="ul")
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
