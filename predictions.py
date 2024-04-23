"""
predictions.py

This module contains functions for creating a mosaic from a list of filenames.

Functions:
----------
create_mosaic(filenames_list: list) -> Tuple[numpy.ndarray, rasterio.Affine]:
predict(config, model, tags) -> numpy.ndarray:
make_predictions(config, model, tags) -> Tuple[numpy.ndarray, numpy.ndarray, Tuple[float, float, float, float]]:

"""

import os
import numpy as np
import rasterio
from rasterio import merge
from rasterio.windows import Window
from rasterio.transform import Affine
import methods
import gc
# import data_loader.build_tags as build_tags


# Get the directory paths from the methods module
directory_paths = methods.get_directories()
DATA_DIRECTORY = directory_paths["data_dir"]
LANDSAT_DIRECTORY = directory_paths["landsat_dir"]
PREDICTIONS_DIRECTORY = directory_paths["predictions_dir"]


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
        # print(src.meta)

    # merge the tif files
    mosaic, out_trans = merge.merge(src_files_to_mosaic)

    # NOTE: alternative if we want every mosaic to cover the entire globe.
    # This takes up more space though.
    # with rasterio.open(DATA_DIRECTORY + "hii_coastal_buffer_mask.tif") as buffer_tif:
    #     bb = buffer_tif.bounds
    #     mosaic, out_trans = merge.merge(src_files_to_mosaic,
    #                                     bounds=bb,
    #                                     )

    for src in src_files_to_mosaic:
        src.close()

    return mosaic[0, :, :], out_trans


def predict(config, model, tags):
    """
    Make predictions using the given config, model, and tags.

    Args:
        config (dict): A dictionary containing the config for prediction.
        model: The trained model used for prediction.
        tags: The tags used for prediction.

    Returns:
        numpy.ndarray: An array containing the predicted values.

    """
    # Make predictions with tf.dataset - SLOW
    # hfi_predict = model.predict(tfds, verbose=1)
    # gc.collect()

    # Make predictions with data generator only and custom loop
    chunk_size = config["inference_chunksize"]

    tags_dict = {}
    for filename in np.unique(tags[-1]):
        isample = [index for (index, item) in enumerate(tags[-1]) if item == filename]
        tags_dict[filename] = np.asarray(isample)

    data_gen = build_tags.data_generator(config, tags, tags_dict)

    hfi_predict = np.zeros((tags[0].shape[0], 1))
    for i in np.arange(0, tags[0].shape[0], chunk_size):
        index_end = np.min([i + chunk_size, tags[0].shape[0]])
        x_input = data_gen.get_data(np.arange(i, index_end), input_only=True)
        hfi_predict[i:index_end] = model.predict(
            x_input, batch_size=config["batch_size"], verbose=0
        )
        gc.collect()

    return hfi_predict


def make_predictions(config, model, tags):
    """
    Generate predictions for ml-HFI.

    Args:
        config (dict): A dictionary containing the config for prediction.
        model: The trained model used for prediction.
        tags: The tags used for prediction.

    Returns:
        tuple: A tuple containing the predicted HFI data, the labels (if available), and the latitude and longitude bounds.

    Raises:
        FileNotFoundError: If the labels file is not found.

    """

    hfi_predict = predict(config, model, tags)

    # GET TIFF META DATA
    labels_filename = (
        DATA_DIRECTORY
        + "hii_"
        + str(config["inference_years"][0])
        + "-01-01_uint8.tif"
    )
    filename_mask = DATA_DIRECTORY + "hii_coastal_buffer_mask.tif"

    lat_s, lat_n, lon_w, lon_e = (
        np.min(tags[1]),
        np.max(tags[1]),
        np.min(tags[2]),
        np.max(tags[2]),
    )

    with rasterio.open(filename_mask) as buffer_mask:
        ilat_s, ilat_n, ilon_w, ilon_e = methods.get_tile_indices(
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
