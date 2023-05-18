import os
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.transform import Affine
import methods


directory_paths = methods.get_directories()
DATA_DIRECTORY = directory_paths["data_dir"]
LANDSAT_DIRECTORY = directory_paths["landsat_dir"]
PREDICTIONS_DIRECTORY = directory_paths["predictions_dir"]


def save_predictions_tif(settings, hfi_predict, predictions_filename):
    # GET TIFF META DATA
    labels_filename = (
        DATA_DIRECTORY + "hii_" + str(settings["testing_years"][0]) + "-01-01_uint8.tif"
    )
    filename_mask = DATA_DIRECTORY + "hii_coastal_buffer_mask.tif"

    # BUG: this code will not run
    with rasterio.open(filename_mask) as buffer_mask:
        ilat0, ilon0 = buffer_mask.index(
            settings["tile"][2], settings["tile"][1]
        )
        ilat1, ilon1 = buffer_mask.index(
            settings["tile"][3], settings["tile"][0]
        )
        lon0, lat0 = buffer_mask.xy(ilat0, ilon0)
        lon1, lat1 = buffer_mask.xy(ilat1, ilon1)

        window = Window.from_slices((ilat0, ilat1 + 1), (ilon0, ilon1 + 1))
        hfi_mask = buffer_mask.read(1, window=window)

    if os.path.isfile(labels_filename):
        with rasterio.open(labels_filename) as labels_tiff:
            ilat0, ilon0 = labels_tiff.index(
                settings["tile"][2], settings["tile"][1]
            )
            ilat1, ilon1 = labels_tiff.index(
                settings["tile"][3], settings["tile"][0]
            )
            window = Window.from_slices((ilat0, ilat1 + 1), (ilon0, ilon1 + 1))
            hfi_labels = labels_tiff.read(1, window=window)
    else:
        hfi_labels = np.zeros(np.shape(hfi_predict)) * np.nan

    # SAVE THE TIFF
    width = ilon1 - ilon0 + 1
    height = ilat1 - ilat0 + 1
    res_lat = ((lat1 - lat0)) / (height - 1)
    res_lon = ((lon1 - lon0)) / (width - 1)

    hfi_labels = np.reshape(hfi_labels, (height, width), order="C")
    hfi_predict = np.reshape(hfi_predict, (height, width), order="C")
    hfi_predict = np.asarray(np.round(hfi_predict), dtype="uint8")
    hfi_predict = np.where(
        hfi_mask == 1, hfi_predict, 255
    )  # remove ocean and turn to nan

    meta_data = {}
    meta_data["nodata"] = 255
    meta_data["width"] = width
    meta_data["height"] = height
    meta_data["driver"] = "GTiff"
    meta_data["count"] = 1
    meta_data["crs"] = rasterio.CRS.from_epsg(4326)
    meta_data["dtype"] = hfi_predict.dtype
    meta_data["transform"] = Affine.translation(lon0, lat0) * Affine.scale(
        res_lon, res_lat
    )

    with rasterio.open(
        PREDICTIONS_DIRECTORY + predictions_filename + ".tif", "w", **meta_data
    ) as dst:
        dst.write(hfi_predict, 1)
        dst.set_band_description(1, "mlHFI prediction")

    return hfi_predict, hfi_labels, lat0, lat1, lon0, lon1
