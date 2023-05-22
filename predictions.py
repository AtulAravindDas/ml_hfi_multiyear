import os
import numpy as np
import rasterio
from rasterio import merge
from rasterio.windows import Window
from rasterio.transform import Affine
import methods
import gc


directory_paths = methods.get_directories()
DATA_DIRECTORY = directory_paths["data_dir"]
LANDSAT_DIRECTORY = directory_paths["landsat_dir"]
PREDICTIONS_DIRECTORY = directory_paths["predictions_dir"]


def create_mosaic(filenames_list):
    src_files_to_mosaic = []
    for fp in filenames_list:
        src = rasterio.open(PREDICTIONS_DIRECTORY + fp)
        src_files_to_mosaic.append(src)
        # print(src.meta)

    mosaic, out_trans = merge.merge(src_files_to_mosaic)

    for src in src_files_to_mosaic:
        src.close()

    return mosaic[0, :, :], out_trans


def make_predictions(settings, model, tfds, tags):

    hfi_predict = model.predict(tfds, verbose=1)
    gc.collect()

    # GET TIFF META DATA
    labels_filename = (
        DATA_DIRECTORY + "hii_" + str(settings["inference_years"][0]) + "-01-01_uint8.tif"
    )
    filename_mask = DATA_DIRECTORY + "hii_coastal_buffer_mask.tif"

    lat_s, lat_n, lon_w, lon_e = (np.min(tags[1]), np.max(tags[1]), np.min(tags[2]), np.max(tags[2]))

    with rasterio.open(filename_mask) as buffer_mask:

        ilat_s, ilat_n, ilon_w, ilon_e = methods.get_tile_indices(buffer_mask, (lat_s, lat_n, lon_w, lon_e))
        ilat_s, ilon_e = ilat_s + 1, ilon_e + 1  # since the input region bounds were inclusive

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
    hfi_predict = np.where(hfi_mask == 1, hfi_predict, 255)  # remove ocean and turn to nan

    # GET LABELS IF THEY EXIST
    if os.path.isfile(labels_filename):
        with rasterio.open(labels_filename) as labels_tiff:
            window = Window.from_slices((ilat_n, ilat_s + 1), (ilon_w, ilon_e + 1))
            hfi_labels = labels_tiff.read(1, window=window)
    else:
        hfi_labels = np.zeros(np.shape(hfi_predict)) * np.nan

    return hfi_predict, hfi_labels, latlon_bounds


def save_predictions_tif(hfi_predict, predictions_filename, trans=None, latlon_bounds=None):

    width = hfi_predict.shape[1]
    height = hfi_predict.shape[0]

    if trans is None:
        lat_s, lat_n, lon_w, lon_e = latlon_bounds
        res_lat = (lat_s - lat_n) / (height - 1)
        res_lon = (lon_e - lon_w) / (width - 1)
        trans = Affine.translation(lon_w, lat_n) * Affine.scale(res_lon, res_lat)

    # SAVE THE TIFF
    meta_data = {}
    meta_data["nodata"] = 255
    meta_data["width"] = width
    meta_data["height"] = height
    meta_data["driver"] = "GTiff"
    meta_data["count"] = 1
    meta_data["crs"] = rasterio.CRS.from_epsg(4326)
    meta_data["dtype"] = hfi_predict.dtype
    meta_data["transform"] = trans
    meta_data["compress"] = "lzw"

    with rasterio.open(
        predictions_filename, "w", **meta_data
    ) as dst:
        dst.write(hfi_predict, 1)
        dst.set_band_description(1, "mlHFI prediction")

    return meta_data
