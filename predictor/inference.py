"""

Functions:
----------
create_mosaic(filenames_list: list) -> Tuple[numpy.ndarray, rasterio.Affine]:
    This function creates a mosaic from a list of filenames. It opens each file, merges them into a mosaic,
    and then closes each file. The function returns the mosaic and the transformation associated with it.

predict(config, model, tags) -> numpy.ndarray:
    Generate predictions for ml-HFI.

make_predictions(config, model, tags, dataloader) -> Tuple[numpy.ndarray, numpy.ndarray, Tuple[float, float, float, float]]:
    Generate predictions for ml-HFI.

save_predictions_tif(hfi_predict, predictions_filename, trans=None, latlon_bounds=None, nodata=255) -> dict:
    Save the predictions as a GeoTIFF file.

"""

import os
import numpy as np
import rasterio
from rasterio import merge
from rasterio.windows import Window
from rasterio.transform import Affine
from data_builder import data_methods
import gc
import utils.utils as utils
import torch


def create_mosaic(filenames_list):
    """
    This function creates a mosaic from a list of filenames. It opens each file, merges them into a mosaic,
    and then closes each file. The function returns the mosaic and the transformation associated with it.

    Parameters:
    filenames_list (list): A list of filenames to be merged into a mosaic.

    Returns:
    numpy.ndarray: The merged mosaic.
    rasterio.Affine: The transformation associated with the mosaic.
    """

    src_files_to_mosaic = []
    for fp in filenames_list:
        src = rasterio.open(fp)
        src_files_to_mosaic.append(src)

    # merge the tif files
    mosaic, out_trans = merge.merge(src_files_to_mosaic)

    for src in src_files_to_mosaic:
        src.close()

    return mosaic[0, :, :], out_trans


def make_predictions(config, model, tags, dataloader):
    """
    Generate predictions for ml-HFI.

    Args:
        config (dict): A dictionary containing the config for prediction.
        model: The trained model used for prediction.
        tags: The tags used for prediction.
        dataloader: The dataloader used for prediction.

    Returns:
        tuple: A tuple containing the predicted HFI data, the labels (if available), and the latitude and longitude bounds.

    Raises:
        FileNotFoundError: If the labels file is not found.

    """

    device = utils.prepare_device(config["device"], config["device_id"])
    with torch.inference_mode():
        hfi_predict = model.predict(
            dataloader=dataloader,
            device=device
        )

    # GET TIFF META DATA
    DATA_DIR = utils.get_directories(config["machine"])["data_dir"]
    labels_filename = (
        DATA_DIR + "hii_" + str(config["data"]["inference_years"][0]) + "-01-01_uint8.tif"
    )

    default_filepaths = utils.get_default_filepaths()
    filename_mask = default_filepaths["mask_filepath"]

    lat_s, lat_n, lon_w, lon_e = (
        np.min(tags[1]),
        np.max(tags[1]),
        np.min(tags[2]),
        np.max(tags[2]),
    )

    with rasterio.open(filename_mask) as buffer_mask:
        ilat_s, ilat_n, ilon_w, ilon_e = data_methods.get_tile_indices(
            buffer_mask, (lat_s, lat_n, lon_w, lon_e)
        )
        ilat_s, ilon_e = (
            ilat_s + 1,
            ilon_e + 1,
        )  # since the input region bounds were inclusive

        lon_w, lat_n = buffer_mask.xy(ilat_n, ilon_w, offset="ul")
        lon_e, lat_s = buffer_mask.xy(ilat_s, ilon_e, offset="ul")

        window = Window.from_slices((ilat_n, ilat_s + 1), (ilon_w, ilon_e + 1))
        hfi_mask = buffer_mask.read(1, window=window)

    # DEFINE THE BOUNDS OF THE PREDICTED DATA
    latlon_bounds = (lat_s, lat_n, lon_w, lon_e)
    width = hfi_mask.shape[1]
    height = hfi_mask.shape[0]

    # RESHAPE FROM VECTOR TO GRID
    hfi_predict = np.reshape(hfi_predict, (height, width), order="C")
    hfi_predict = np.asarray(np.round(hfi_predict), dtype="uint8")
    hfi_predict = np.where(
        hfi_mask == 1, hfi_predict, 255
    )  # remove ocean and turn to nan

    # GET LABELS IF THEY EXIST
    if os.path.isfile(labels_filename):
        with rasterio.open(labels_filename) as labels_tiff:
            window = Window.from_slices((ilat_n, ilat_s + 1), (ilon_w, ilon_e + 1))
            hfi_labels = labels_tiff.read(1, window=window)
    else:
        hfi_labels = np.zeros(np.shape(hfi_predict)) * np.nan

    return hfi_predict, hfi_labels, latlon_bounds


def save_predictions_tif(
    hfi_predict, predictions_filename, trans=None, latlon_bounds=None, nodata=255
):
    """
    Save the predictions as a GeoTIFF file.

    Args:
        hfi_predict (numpy.ndarray): The predicted values.
        predictions_filename (str): The filename to save the GeoTIFF file.
        trans (affine.Affine, optional): The affine transformation matrix. If not provided, it will be calculated based on the latlon_bounds.
        latlon_bounds (tuple, optional): The latitude and longitude bounds of the data in the format (lat_s, lat_n, lon_w, lon_e).
        nodata (int, optional): The nodata value to be set in the GeoTIFF file.

    Returns:
        dict: A dictionary containing the metadata of the saved GeoTIFF file.
    """
    width = hfi_predict.shape[1]
    height = hfi_predict.shape[0]

    if trans is None:
        lat_s, lat_n, lon_w, lon_e = latlon_bounds
        res_lat = (lat_s - lat_n) / (height - 1)
        res_lon = (lon_e - lon_w) / (width - 1)
        trans = Affine.translation(lon_w, lat_n) * Affine.scale(res_lon, res_lat)

    # SAVE THE TIFF
    meta_data = {}
    meta_data["nodata"] = nodata
    meta_data["width"] = width
    meta_data["height"] = height
    meta_data["driver"] = "GTiff"
    meta_data["count"] = 1
    meta_data["crs"] = rasterio.CRS.from_epsg(4326)
    meta_data["dtype"] = hfi_predict.dtype
    meta_data["transform"] = trans
    meta_data["compress"] = "lzw"

    with rasterio.open(predictions_filename, "w", **meta_data) as dst:
        dst.write(hfi_predict, 1)
        dst.set_band_description(1, "mlHFI prediction")

    return meta_data
