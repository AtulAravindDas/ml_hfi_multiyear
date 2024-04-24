"""Data loader modules.

Classes
---------
CustomData(torch.utils.data.Dataset)

"""

import os
from torch.utils.data import Dataset
import torch
import numpy as np
import pickle
import rasterio
from rasterio.windows import Window

from data_builder import read_landsat
from utils import utils


directory_paths = utils.get_directories()
SAVE_MODEL_DIRECTORY = directory_paths["save_model_dir"]
DATA_DIRECTORY = directory_paths["data_dir"]
LANDSAT_DIRECTORY = directory_paths["landsat_dir"]
PREDICTIONS_DIRECTORY = directory_paths["predictions_dir"]


class CustomData(torch.utils.data.Dataset):
    """
    Custom dataset for data loading.
    """

    def __init__(self, config, tags):

        tags_dict = {}
        for filename in np.unique(tags[-1]):
            isample = [
                index for (index, item) in enumerate(tags[-1]) if item == filename
            ]
            tags_dict[filename] = np.asarray(isample)

        self.tags = tags
        self.tags_dict = tags_dict
        self.config = config
        self.rng = np.random.default_rng(config["rng_seed"])

        self.sample_years = self.tags[0]
        self.sample_lats = self.tags[1]
        self.sample_lons = self.tags[2]
        self.sample_files = self.tags[3]

    def __len__(self):
        # return len(self.tags[0])
        return np.ceil(len(self.tags[0]) / self.config["batch_size"]).astype(int)

    def __getitem__(self, id_batch):

        # get consecutive indices based on starting index of batch_id
        # idx = np.arange(idx, idx + self.config["batch_size"])
        idx = np.arange(
            id_batch * self.config["batch_size"],
            (id_batch + 1) * self.config["batch_size"],
        )

        if self.config["mode"] == "training":
            tile_key = self.sample_files[idx[0]]

            i = self.rng.choice(
                self.tags_dict[tile_key],
                self.config["batch_size"],
                replace=False,
            )

            sample_years = self.sample_years[i]
            sample_lats = self.sample_lats[i]
            sample_lons = self.sample_lons[i]
            sample_files = np.asarray(self.sample_files)[i]

        elif self.config["mode"] == "inference":
            try:
                sample_years = self.sample_years[idx]
                sample_lats = self.sample_lats[idx]
                sample_lons = self.sample_lons[idx]
                sample_files = np.asarray(self.sample_files)[idx]
            except:
                idx = np.where(idx < len(self.sample_years))[0]
                sample_years = self.sample_years[idx]
                sample_lats = self.sample_lats[idx]
                sample_lons = self.sample_lons[idx]
                sample_files = np.asarray(self.sample_files)[idx]

        else:
            raise NotImplementedError("no such mode.")

        assert all(
            x == sample_files[0] for x in sample_files
        ), f"these must all be the same files: {sample_files}"

        # GET INPUT DATA
        input_data = self.get_input_data(
            sample_years, sample_lats, sample_lons, sample_files
        )
        # swap dimensions to make (batch, channels, width, height)
        input_data = np.swapaxes(input_data, 1, 3)

        # GET OUTPUT DATA
        output_data = self.get_output_data(
            sample_years, sample_lats, sample_lons, sample_files
        )

        return (
            torch.tensor(input_data, dtype=torch.float32),
            torch.tensor(output_data, dtype=torch.float32),
        )

    def get_input_data(self, sample_years, sample_lats, sample_lons, sample_files):
        batch_input = np.zeros(
            (
                len(sample_years),
                self.config["scene_width_landsat"],
                self.config["scene_width_landsat"],
                len(self.config["channels"]),
            )
        )

        # read landsat file
        filename = LANDSAT_DIRECTORY + sample_files[0] + ".tif"

        if not os.path.isfile(filename):
            if self.config["mode"] == "training":
                raise ValueError("No such input Landsat file: " + filename)
            elif self.config["mode"] == "inference":
                # return torch.tensor(batch_input, dtype=torch.float32)
                return batch_input
            else:
                raise NotImplementedError("no such mode.")

        # intialize tif neighborhood dictionary and loop through samples to get the data
        tif_dict = {}
        tif_dict, flag = read_landsat.fill_tif_dict(
            "central",
            sample_years[0],
            sample_lats[0],
            sample_lons[0],
            tif_dict,
            self.config,
        )

        for isample in np.arange(0, len(sample_years)):

            sample_out, tif_dict, usecase = read_landsat.read_input_data(
                self.config,
                tif_dict,
                sample_years[isample],
                sample_lons[isample],
                sample_lats[isample],
                self.config["channels"],
                self.config["scene_width"],
            )

            batch_input[isample, :, :, :] = sample_out

        # close tifs in the dictionary
        for key in tif_dict.keys():
            if isinstance(tif_dict[key], rasterio.io.DatasetReader):
                tif_dict[key].close()

        # convert to tensor
        # dat = torch.tensor(batch_input, dtype=torch.float32)

        return batch_input

    def get_output_data(self, sample_years, sample_lats, sample_lons, sample_files):
        # Get HFI file
        assert all(
            x == sample_years[0] for x in sample_years
        ), f"these must all be the same years: {sample_years}"

        batch_output = np.zeros((len(sample_years), 1))

        filename = DATA_DIRECTORY + "hii_" + str(sample_years[0]) + "-01-01_uint8.tif"
        if not os.path.isfile(filename):
            return batch_output * 0.0

        with rasterio.open(filename) as output_tiff:
            for isample in np.arange(0, len(sample_years)):
                batch_output[isample] = read_output_data(
                    self.config,
                    output_tiff,
                    sample_lons[isample],
                    sample_lats[isample],
                )

        return batch_output


def read_output_data(config, tiff, sample_lon, sample_lat):
    ilat, ilon = tiff.index(sample_lon, sample_lat)
    window = Window(ilon, ilat, 1, 1)

    output_mask = (
        tiff.read_masks(1, window=window) // 255.0
    )  # convert to 0/1, with 0 = no data
    sample_output = output_mask * tiff.read(1, window=window)

    if len(sample_output) == 0:
        sample_output = 0.0

    # TODO: We are trying to make this obsolete, so delete later.
    # this is where we can force the network to predict zeros or ones
    # if config["mode"] == "training":
    #     if sample_output == 0.0:
    #         sample_output = config["kluge_value_for_zero"]

    return sample_output
