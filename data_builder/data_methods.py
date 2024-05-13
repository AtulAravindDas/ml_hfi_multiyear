"""
This module provides functions and classes for loading and processing data.

Classes
---------
None


Functions
---------
remove_nodata(x, y=None, nodata=255)
    Remove nodata values from the input arrays.

get_lats_lons_list(region, tile_len_deg)
    Get a list of latitude and longitude values based on the given region and tile length.

get_region_bounds(region, tile_len_deg)
    Get the bounding box coordinates of the given region.

get_tile_indices(hfi_tif, tile)
    Get the indices of the tile within the HFI raster.

trim_hfi_region(indices, hfi_tif, region)
    Trim the HFI region based on the given indices and region boundaries.

mask_shapefile_region(hfi_tif, mask_tif, shp_dict, country_names)
    Mask the HFI raster based on a shapefile region.


"""

import numpy as np
from rasterio.windows import Window
from data_builder import read_landsat

__author__ = "Elizabeth A. Barnes and Randal J. Barnes"
__date__ = "24 April 2024"


def remove_nodata(x, y=None, nodata=255):
    """
    Remove nodata values from the input arrays.

    Parameters:
    - x (ndarray): Input array.
    - y (ndarray, optional): Second input array. Default is None.
    - nodata (int, optional): Nodata value to be removed. Default is 255.

    Returns:
    - If y is None, returns the filtered x array.
    - If y is not None, returns the filtered x and y arrays.
    """
    if y is None:
        index = np.where(x != nodata)[0]
        return x[index]
    else:
        assert np.shape(x) == np.shape(y), f"{np.shape(x)} != {np.shape(y)}"

        index = np.where(x != nodata)[0]
        x, y = x[index], y[index]

        index = np.where(y != nodata)[0]
        x, y = x[index], y[index]

        return x, y


def get_lats_lons_list(region, tile_len_deg):
    """
    Get a list of latitude and longitude values based on the given region and tile length.

    Parameters:
        region (list or dict): The region of interest. If it's a list, it should contain the
            coordinates of the bounding box in the order [lat_s, lat_n, lon_w, lon_e]. If it's
            a dict, it should contain the keys 'lats' and 'lons' which represent the latitude
            and longitude values respectively.
        tile_len_deg (float): The length of each tile in degrees.

        Example Usage in Config:
        "training_region": {"tilelats": [[0], [10], [30], [40], [50]],
                            "tilelons": [[0, 10], [0, 10, -30], [0, 50, 20], [0, 10], [0]]},
        "inference_region": [0, 20, 0, 20],

    Returns:
        dict: A dictionary containing the latitude and longitude lists.

    Raises:
        NotImplementedError: If the region type is not supported.

    """

    if isinstance(region, list):

        (lat_s, lat_n, lon_w, lon_e) = read_landsat.get_landsat_bounds(
            region=region, tile_len_deg=tile_len_deg
        )

        lats_list = [
            np.arange(lat_s + tile_len_deg, lat_n + tile_len_deg, tile_len_deg),
        ]
        lons_list = [np.arange(lon_w, lon_e, tile_len_deg)]

    elif isinstance(region, dict):
        lats_list = region["tilelats"]
        lons_list = region["tilelons"]

        if len(lats_list) != len(lons_list):

            assert len(lons_list) == 1 or len(lats_list) == 1

            if len(lons_list) == 1:
                lons_list = np.repeat(lons_list, len(lats_list), axis=0)

            if len(lats_list) == 1:
                lats_list = np.repeat(lats_list, len(lons_list), axis=0)

    else:
        raise NotImplementedError

    return {"lats": lats_list, "lons": lons_list}


def get_region_bounds(region, tile_len_deg):
    """
    Get the bounding box coordinates of the given region.

    Parameters:
        region (list or dict): The region of interest. If it's a list, it should contain the
            coordinates of the bounding box in the order [lat_s, lat_n, lon_w, lon_e]. If it's
            a dict, it should contain the keys 'lats' and 'lons' which represent the latitude
            and longitude values respectively.
        tile_len_deg (float): The length of each tile in degrees.

    Returns:
        tuple: A tuple containing the bounding box coordinates (lat_s, lat_n, lon_w, lon_e).

    Raises:
        NotImplementedError: If the region type is not supported.
    """
    lats_lons_dict = get_lats_lons_list(region, tile_len_deg)

    lat_s = np.min(lats_lons_dict["lats"]) - tile_len_deg
    lat_n = np.max(lats_lons_dict["lats"])

    lon_w = np.min(lats_lons_dict["lons"])
    lon_e = np.max(lats_lons_dict["lons"]) + tile_len_deg

    return lat_s, lat_n, lon_w, lon_e


def get_tile_indices(hfi_tif, tile):
    """
    Get the indices of the tile within the HFI raster.

    Parameters:
    - hfi_tif (rasterio.DatasetReader): HFI raster dataset.
    - tile (tuple): Tuple containing the tile boundaries (lat_s, lat_n, lon_w, lon_e).

    Returns:
    - Tuple containing the indices of the tile within the HFI raster.
    """
    ilat_n, ilon_w = hfi_tif.index(tile[2], tile[1])
    ilat_s, ilon_e = hfi_tif.index(tile[3], tile[0])

    indices = ilat_s, ilat_n, ilon_w, ilon_e

    return trim_hfi_region(indices, hfi_tif, tile)


def trim_hfi_region(indices, hfi_tif, region):
    """
    Trim the HFI region based on the given indices and region boundaries.

    Parameters:
    - indices (tuple): Tuple containing the indices of the HFI region to be trimmed.
    - hfi_tif (rasterio.DatasetReader): HFI raster dataset.
    - region (tuple): Tuple containing the region boundaries (lat_s, lat_n, lon_w, lon_e).

    Returns:
    - Tuple containing the trimmed indices of the HFI region.
    """
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


def mask_shapefile_region(hfi_tif, mask_tif, shp_dict, country_names):
    """
    Mask the HFI raster based on a shapefile region.

    Parameters:
    - hfi_tif (rasterio.DatasetReader): HFI raster dataset.
    - mask_tif (rasterio.DatasetReader): Mask raster dataset.
    - shp_dict (pandas.DataFrame): DataFrame containing shapefile information.
    - country_names (list): List of country names to be masked.

    Returns:
    - Masked HFI raster array.
    """
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

    return masked_hfi
