"""Methods for doing analysis and processing data

Functions
---------
get_sample_weights(settings, data)
permute_shuffle_sample_list(settings, sample_years, sample_lats, sample_lons)
"""
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.windows import Window

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


def remove_nodata(x, y=None, nodata=255):
    if y is None:
        index = np.where(x != nodata)[0]
        return x[index]
    else:
        assert np.shape(x) == np.shape(y)

        index = np.where(x != nodata)[0]
        x, y = x[index], y[index]

        index = np.where(y != nodata)[0]
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


def mask_shapefile_region(hfi_tif, mask_tif, shp_dict, country_names, showplot=False):

    # trim the grids to be the same
    lon_w, lat_n = hfi_tif.xy(0, 0, offset="ul")
    lon_e, lat_s = hfi_tif.xy(hfi_tif.height - 1, hfi_tif.width - 1, offset="ul")

    ilat_s, ilat_n, ilon_w, ilon_e = get_tile_indices(
        mask_tif, (lat_s, lat_n, lon_w, lon_e)
    )
    ilat_s, ilon_e = (
        ilat_s + 1,
        ilon_e + 1,
    )  # since the input region bounds were inclusive

    # get shapefile values
    window = Window.from_slices((ilat_n, ilat_s + 1), (ilon_w, ilon_e + 1))
    shp_mask = mask_tif.read(1, window=window)

    # get hfi values
    hfi = np.asarray(hfi_tif.read(1), dtype="float")

    # mask the hfi
    country_codes = shp_dict.loc[shp_dict["ADMIN"].isin(country_names)].index.to_numpy()
    masked_hfi = np.where(np.isin(shp_mask, country_codes), hfi, 255)

    if showplot:
        plt.figure()
        plt.imshow(masked_hfi)
        plt.show()

    return masked_hfi
