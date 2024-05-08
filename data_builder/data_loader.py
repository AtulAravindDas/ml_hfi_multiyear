"""
Data loader modules.

Functions
---------
read_output_data(config, tiff, sample_lon, sample_lat)

Classes
---------
CustomData(torch.utils.data.Dataset)
    A class representing a custom data.

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


class CustomData(torch.utils.data.Dataset):
    """
    Custom dataset for data loading.
    """

    def __init__(self, config, tags, batch_size=32):
        """
        Initialize the CustomData dataset.

        Args:
            config (dict): Configuration parameters for the dataset.
            tags (list): List of tags for the dataset.

        Returns:
            None
        """
        self.data_dir = utils.get_directories(config["machine"])["data_dir"]
        self.landsat_dir = utils.get_directories(config["machine"])["landsat_dir"]

        tags_dict = {}
        for filename in np.unique(tags[-1]):
            isample = [
                index for (index, item) in enumerate(tags[-1]) if item == filename
            ]
            tags_dict[filename] = np.asarray(isample)

        self.tags = tags
        self.tags_dict = tags_dict
        self.config = config
        self.batch_size = batch_size
        self.rng = np.random.default_rng(config["seed"] + 55)

        self.sample_years = self.tags[0]
        self.sample_lats = self.tags[1]
        self.sample_lons = self.tags[2]
        self.sample_files = np.asarray(self.tags[3])

    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
            int: The length of the dataset.
        """
        return np.ceil(len(self.tags[0]) / self.batch_size).astype(int)

    def __getitem__(self, id_batch):
        """
        Get an item from the dataset.

        Args:
            id_batch (int): The index of the batch.

        Returns:
            tuple: A tuple containing the input data and the output data.
        """
        idx = np.arange(
            id_batch * self.batch_size,
            (id_batch + 1) * self.batch_size,
        )

        if self.config["mode"] == "training":
            try:
                tile_key = self.sample_files[self.rng.choice(idx, 1, replace=False)[0]]
            except:
                idx = np.where(idx < len(self.sample_files))[0]
                tile_key = self.sample_files[self.rng.choice(idx, 1, replace=False)[0]]

            i = self.rng.choice(
                self.tags_dict[tile_key],
                self.batch_size,
                replace=True,
            )

            sample_years = self.sample_years[i]
            sample_lats = self.sample_lats[i]
            sample_lons = self.sample_lons[i]
            sample_files = self.sample_files[i]

        elif self.config["mode"] == "inference":

            # quicklook option to only grab every "quicklook_skiplen" index for inference
            if self.config["inference"]["quicklook"]:
                idx = np.arange(
                    id_batch * self.batch_size,
                    (id_batch + 1) * self.batch_size,
                    self.config["inference"]["quicklook_skiplen"],
                )

            try:
                sample_years = self.sample_years[idx]
                sample_lats = self.sample_lats[idx]
                sample_lons = self.sample_lons[idx]
                sample_files = self.sample_files[idx]
            except:
                idx = np.where(idx < len(self.sample_years))[0]
                sample_years = self.sample_years[idx]
                sample_lats = self.sample_lats[idx]
                sample_lons = self.sample_lons[idx]
                sample_files = self.sample_files[idx]

        else:
            raise NotImplementedError("no such mode.")

        assert all(
            x == sample_files[0] for x in sample_files
        ), f"these must all be the same files: {sample_files}"

        input_data = self.get_input_data(
            sample_years, sample_lats, sample_lons, sample_files
        )
        input_data = np.swapaxes(input_data, 1, 3)

        output_data = self.get_output_data(
            sample_years, sample_lats, sample_lons, sample_files
        )

        return (
            torch.tensor(input_data, dtype=torch.float32),
            torch.tensor(output_data, dtype=torch.float32),
        )

    def get_input_data(self, sample_years, sample_lats, sample_lons, sample_files):
        """
        Get the input data for a batch.

        Args:
            sample_years (list): List of sample years.
            sample_lats (list): List of sample latitudes.
            sample_lons (list): List of sample longitudes.
            sample_files (list): List of sample files.

        Returns:
            np.ndarray: The input data for the batch.
        """
        batch_input = np.zeros(
            (
                len(sample_years),
                self.config["scene_width_landsat"],
                self.config["scene_width_landsat"],
                len(self.config["data"]["channels"]),
            )
        )

        filename = self.landsat_dir + sample_files[0] + ".tif"

        if not os.path.isfile(filename):
            if self.config["mode"] == "training":
                raise ValueError("No such input Landsat file: " + filename)
            elif self.config["mode"] == "inference":
                return batch_input
            else:
                raise NotImplementedError("no such mode.")

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

            sample_input, tif_dict, usecase = read_landsat.read_input_data(
                self.config,
                tif_dict,
                sample_years[isample],
                sample_lons[isample],
                sample_lats[isample],
                self.config["data"]["channels"],
                self.config["data"]["scene_width"],
                rng=self.rng,
            )

            batch_input[isample, :, :, :] = sample_input

        for key in tif_dict.keys():
            if isinstance(tif_dict[key], rasterio.io.DatasetReader):
                tif_dict[key].close()

        return batch_input

    def get_output_data(self, sample_years, sample_lats, sample_lons, sample_files):
        """
        Get the output data for a batch.

        Args:
            sample_years (list): List of sample years.
            sample_lats (list): List of sample latitudes.
            sample_lons (list): List of sample longitudes.
            sample_files (list): List of sample files.

        Returns:
            np.ndarray: The output data for the batch.
        """
        assert all(
            x == sample_years[0] for x in sample_years
        ), f"these must all be the same years: {sample_years}"

        batch_output = np.zeros((len(sample_years), 1))

        filename = self.data_dir + "hii_" + str(sample_years[0]) + "-01-01_uint8.tif"
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
    """
    Read the output data from a TIFF file.

    Args:
        config (dict): Configuration parameters.
        tiff (rasterio.io.DatasetReader): The TIFF file.
        sample_lon (float): The sample longitude.
        sample_lat (float): The sample latitude.

    Returns:
        float: The output data value.
    """
    ilat, ilon = tiff.index(sample_lon, sample_lat)
    window = Window(ilon, ilat, 1, 1)

    output_mask = (
        tiff.read_masks(1, window=window) // 255.0
    )  # convert to 0/1, with 0 = no data
    sample_output = output_mask * tiff.read(1, window=window)

    if len(sample_output) == 0:
        sample_output = 0.0

    # NOTE: moved this code to the loss function where it belongs
    # Force the network to predict zeros or ones
    # if config["mode"] == "training":
    #     if config["data"].get("kluge_value_for_zero", None) is not None:
    #         if sample_output == 0.0:
    #             sample_output = config["data"]["kluge_value_for_zero"]

    return sample_output
