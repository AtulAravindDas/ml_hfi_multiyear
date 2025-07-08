"""Read the landsat data.

This module provides functions to read Landsat data and extract specific regions of interest.

Classes
---------
None


Functions
---------
fill_tif_dict(name, sample_year, sample_lat, sample_lon, tif_dict, config)
    Fill the tif_dict with Landsat data based on the given name and coordinates.
read_tif(tif, channels, window)
    Read the specified channels from the given tif file within the specified window.
get_landsat_bounds(region, tile_len_deg)
    Get the latitude and longitude bounds for the given region based on the tile length.
get_input_filename(years, lats, lons, config)
    Get the input filenames based on the years, latitudes, longitudes, and configuration.
read_input_data(config, tif_dict, sample_year, sample_lon, sample_lat, channels, scene_width, rng=np.random.default_rng(42))
    Read the input data from the Landsat files based on the given parameters.

"""
import re
import os
import numpy as np
import rasterio
from rasterio.windows import from_bounds, Window
from rasterio.transform import from_bounds as make_transform
from utils import utils


def fill_tif_dict(name, sample_year, sample_lat, sample_lon, tif_dict, config):
    """
    Fills a dictionary with information about a TIFF file based on the given parameters.

    Args:
        name (str): The name of the TIFF file.
        sample_year (int): The year of the sample.
        sample_lat (float): The latitude of the sample.
        sample_lon (float): The longitude of the sample.
        tif_dict (dict): The dictionary to be filled with TIFF file information.
        config (dict): Configuration parameters.

    Returns:
        tuple: A tuple containing the updated tif_dict and a flag indicating if the file exists.
    """
    if name == "central":
        filename = get_input_filename(
            (sample_year,), (sample_lat,), (sample_lon,), config
        )
    elif name == "northwest":
        filename = get_input_filename(
            (sample_year,),
            (sample_lat + config["tile_len_deg"],),
            (sample_lon - config["tile_len_deg"],),
            config,
        )
    elif name == "north":
        filename = get_input_filename(
            (sample_year,),
            (sample_lat + config["tile_len_deg"],),
            (sample_lon,),
            config,
        )
    elif name == "northeast":
        filename = get_input_filename(
            (sample_year,),
            (sample_lat + config["tile_len_deg"],),
            (sample_lon + config["tile_len_deg"],),
            config,
        )
    elif name == "east":
        filename = get_input_filename(
            (sample_year,),
            (sample_lat,),
            (sample_lon + config["tile_len_deg"],),
            config,
        )
    elif name == "southeast":
        filename = get_input_filename(
            (sample_year,),
            (sample_lat - config["tile_len_deg"],),
            (sample_lon + config["tile_len_deg"],),
            config,
        )
    elif name == "south":
        filename = get_input_filename(
            (sample_year,),
            (sample_lat - config["tile_len_deg"],),
            (sample_lon,),
            config,
        )
    elif name == "southwest":
        filename = get_input_filename(
            (sample_year,),
            (sample_lat - config["tile_len_deg"],),
            (sample_lon - config["tile_len_deg"],),
            config,
        )
    elif name == "west":
        filename = get_input_filename(
            (sample_year,),
            (sample_lat,),
            (sample_lon - config["tile_len_deg"],),
            config,
        )
    else:
        raise NotImplementedError("no such name.")

    # check if the file exists, if not, throw flag
    directory_paths = utils.get_directories(config["machine"])
    tif_filename = directory_paths["landsat_dir"] + filename[0] + ".tif"
    if os.path.isfile(tif_filename) is False:
        # print(tif_filename)
        return tif_dict, True

    # since the file exists, set dictionary values about the tif file
    tif_dict[name] = rasterio.open(tif_filename)
    tif_dict[name + "_height"] = tif_dict[name].height
    tif_dict[name + "_width"] = tif_dict[name].width

    return tif_dict, False


def read_tif(tif, channels, window):
    """
    Read specific channels from a TIFF file and return the data as a NumPy array.

    Parameters:
    tif (object): The TIFF file object.
    channels (list): The list of channel indices to read. Indexed from 1.
    window (tuple): The window coordinates (x, y, width, height) to read from.

    Returns:
    numpy.ndarray: The data from the specified channels and window.

    """
    return np.transpose(tif.read(channels, window=window), axes=(1, 2, 0))


def extract_lat_lon_from_filename(tif_path):
    """
    Extract the latitude and longitude from a filename like 'landsat_-20lat_-70lon_2024.tif'.

    Returns:
        (lat, lon): Tuple of floats.
    """
    filename = os.path.basename(tif_path)
    match = re.search(r'(-?\d+)[lL]at_(-?\d+)[lL]on', filename)
    if match:
        lat = int(match.group(1))
        lon = int(match.group(2))
        return lat, lon
    else:
        raise ValueError(f"Could not extract lat/lon from: {filename}")


def read_tif_checking_fixed(tif, channels, window, tile_len):
    lat_max, lon_min = extract_lat_lon_from_filename(tif.name)
    lat_min = lat_max - tile_len
    lon_max = lon_min + tile_len
    

    transform=tif.transform
    if transform.e > 0:
        # Bad transform: bottom-up (latitude increases down the image)
        left, bottom, right, top = tif.bounds
        fixed_transform = make_transform(left, top, right, bottom, tif.width, tif.height)  
        fixed_transform
    else:
        fixed_transform=transform


    # Build the full tile window from lat/lon bounds
    full_window = from_bounds(lon_min, lat_min, lon_max, lat_max, transform=fixed_transform)

    # Check for 0–360 wraparound
    if abs(full_window.col_off) > tif.width:
        lon_min = (lon_min + 360) % 360
        lon_max = lon_min + tile_len
        full_window = from_bounds(lon_min, lat_min, lon_max, lat_max, transform=fixed_transform)


    # Clip window to be within full_window
    clipped_col_off = max(window.col_off, full_window.col_off)
    clipped_row_off = max(window.row_off, full_window.row_off)

    # Compute how many pixels to get without going over the full_window bounds
    col_end = min(window.col_off + window.width, full_window.col_off + full_window.width)
    row_end = min(window.row_off + window.height, full_window.row_off + full_window.height)

    # Adjust width/height if too small — shift origin backward if needed to preserve size
    adjusted_width = window.width
    adjusted_height = window.height

    # Check if we're short on width
    if (col_end - clipped_col_off) < window.width:
        clipped_col_off = col_end - window.width
        clipped_col_off = max(clipped_col_off, full_window.col_off)

    # Check if we're short on height
    if (row_end - clipped_row_off) < window.height:
        clipped_row_off = row_end - window.height
        clipped_row_off = max(clipped_row_off, full_window.row_off)

    # Final window — guaranteed to have the right size
    final_window = Window(
        col_off=clipped_col_off,
        row_off=clipped_row_off,
        width=window.width,
        height=window.height
    )

    return np.transpose(tif.read(channels, window=final_window), axes=(1, 2, 0))


def get_landsat_bounds(region, tile_len_deg):
    """
    Calculate the bounds of a Landsat tile based on a given region and tile length.

    Parameters:
    region (tuple): A tuple containing the latitude and longitude bounds of the region.
    tile_len_deg (float): The length of each tile in degrees.

    Returns:
    tuple: A tuple containing the southern, northern, western, and eastern bounds of the Landsat tile.
    """

    lat_s = np.floor(region[0] / tile_len_deg) * tile_len_deg
    lat_n = np.ceil(region[1] / tile_len_deg) * tile_len_deg
    lon_w = np.floor(region[2] / tile_len_deg) * tile_len_deg
    lon_e = np.ceil(region[3] / tile_len_deg) * tile_len_deg

    return lat_s, lat_n, lon_w, lon_e


def get_input_filename(years, lats, lons, config):
    """
    Generates a list of input filenames based on the given years, latitudes, longitudes, and configuration.

    Args:
        years (list): A list of years.
        lats (list): A list of latitudes.
        lons (list): A list of longitudes.
        config (dict): A dictionary containing configuration parameters.

    Returns:
        list: A list of input filenames.

    """
    filenames = []
    for isample in range(len(years)):
        file_year = int(years[isample])
        file_lat = int(
            np.ceil(lats[isample] / config["tile_len_deg"]) * config["tile_len_deg"]
        )
        file_lon = int(
            np.floor(lons[isample] / config["tile_len_deg"]) * config["tile_len_deg"]
        )
        filenames.append(f"landsat_{file_lat}lat_{file_lon}lon_{file_year}")

    return filenames


def read_input_data(
    config,
    tif_dict,
    sample_year,
    sample_lon,
    sample_lat,
    channels,
    scene_width,
    rng=np.random.default_rng(42),
):
    """
    Reads input data based on the given configuration and parameters.

    Args:
        config (dict): Configuration parameters.
        tif_dict (dict): Dictionary containing TIFF files.
        sample_year (int): Year of the sample.
        sample_lon (float): Longitude of the sample.
        sample_lat (float): Latitude of the sample.
        channels (list): List of channels to read from the TIFF files.
        scene_width (int): Width of the scene.
        rng (numpy.random.Generator, optional): Random number generator. Defaults to np.random.default_rng(42).

    Returns:
        numpy.ndarray: Input data.

    Raises:
        NotImplementedError: If the use case is not implemented.

    """

    # if the tif file is on the dateline there its longitudes are 0-360
    # this is a fix for that without prescribing the tile name
    ilat, ilon = tif_dict["central"].index(sample_lon, sample_lat)
    
    if ilon < -tif_dict["central_width"]:
        sample_lon = 360.0 + sample_lon
        ilat, ilon = tif_dict["central"].index(sample_lon, sample_lat)

    scene_width_2 = int((scene_width - 1) / 2)
    ilat_n = int(ilat - config["landsat_to_hfi_ratio"] * scene_width_2)
    ilat_s = int(ilat + config["landsat_to_hfi_ratio"] * (scene_width_2 + 1) - 1)
    ilon_w = int(ilon - config["landsat_to_hfi_ratio"] * scene_width_2)
    ilon_e = int(ilon + config["landsat_to_hfi_ratio"] * (scene_width_2 + 1) - 1)

    # determine the usecase
    if (
        ilat_n >= 0
        and ilon_w >= 0
        and ilat_s < tif_dict["central_height"]
        and ilon_e < tif_dict["central_width"]
    ):
        usecase = "usecase_central"
    elif ilat_n < 0 and ilon_w < 0:
        usecase = "usecase_northwest"
    elif ilat_n < 0 and ilon_w >= 0 and ilon_e < tif_dict["central_width"]:
        usecase = "usecase_north"
    elif ilat_n < 0 and ilon_e >= tif_dict["central_width"]:
        usecase = "usecase_northeast"
    elif (
        ilat_n >= 0
        and ilat_s < tif_dict["central_height"]
        and ilon_e >= tif_dict["central_width"]
    ):
        usecase = "usecase_east"
    elif ilat_s >= tif_dict["central_height"] and ilon_e >= tif_dict["central_width"]:
        usecase = "usecase_southeast"
    elif (
        ilon_w >= 0
        and ilat_s >= tif_dict["central_height"]
        and ilon_e < tif_dict["central_width"]
    ):
        usecase = "usecase_south"
    elif ilon_w < 0 and ilat_s >= tif_dict["central_height"]:
        usecase = "usecase_southwest"
    elif ilat_n >= 0 and ilon_w < 0 and ilat_s < tif_dict["central_height"]:
        usecase = "usecase_west"
    else:
        raise NotImplementedError("no such use case")

    # the speed of this code assumes that training samples will never be on edges or corners
    if config["mode"] == "training":
        assert usecase == "usecase_central", (
            usecase,
            sample_year,
            sample_lat,
            sample_lon,
            ilat,
            ilon,
            ilat_s,
            ilat_n,
            ilon_e,
            ilon_w,
        )

    # USECASE 0 - central only
    if usecase == "usecase_central":
        central_output = read_tif_checking_fixed( 
            tif_dict["central"],
            channels,
            window=Window.from_slices((ilat_n, ilat_s + 1), (ilon_w, ilon_e + 1)),
            tile_len=config["tile_len_deg"], 
        )
        sample_input = central_output

    # USECASE 1 - northwest corner
    elif usecase == "usecase_northwest":
        if tif_dict.get("north") is None:
            tif_dict, flag_north = fill_tif_dict(
                "north", sample_year, sample_lat, sample_lon, tif_dict, config
            )
        else:
            flag_north = False

        if tif_dict.get("northwest") is None:
            tif_dict, flag_northwest = fill_tif_dict(
                "northwest", sample_year, sample_lat, sample_lon, tif_dict, config
            )
        else:
            flag_northwest = False

        if tif_dict.get("west") is None:
            tif_dict, flag_west = fill_tif_dict(
                "west", sample_year, sample_lat, sample_lon, tif_dict, config
            )
        else:
            flag_west = False

        if any([flag_north, flag_northwest, flag_west]):
            sample_input = 0.0
        else:
            central_output = read_tif_checking_fixed( 
                tif_dict["central"],
                channels,
                window=Window.from_slices((0, ilat_s + 1), (0, ilon_e + 1)),
                tile_len=config["tile_len_deg"], 
            )
            west_output = read_tif_checking_fixed( 
                tif_dict["west"],
                channels,
                window=Window.from_slices(
                    (0, ilat_s + 1),
                    (tif_dict["west_width"] + ilon_w, tif_dict["west_width"]),
                ),
                tile_len=config["tile_len_deg"], 
            )
            north_output = read_tif_checking_fixed( 
                tif_dict["north"],
                channels,
                window=Window.from_slices(
                    (tif_dict["north_height"] + ilat_n, tif_dict["north_height"]),
                    (0, ilon_e + 1),
                ),
                tile_len=config["tile_len_deg"], 
            )
            northwest_output = read_tif_checking_fixed( 
                tif_dict["northwest"],
                channels,
                window=Window.from_slices(
                    (
                        tif_dict["northwest_height"] + ilat_n,
                        tif_dict["northwest_height"],
                    ),
                    (tif_dict["northwest_width"] + ilon_w, tif_dict["northwest_width"]),
                ),
                tile_len=config["tile_len_deg"], 
            )
            sample_input = np.vstack(
                (
                    np.hstack((northwest_output, north_output)),
                    np.hstack((west_output, central_output)),
                )
            )
            

    # USECASE 2 - north edge
    elif usecase == "usecase_north":
        if tif_dict.get("north") is None:
            tif_dict, flag_north = fill_tif_dict(
                "north", sample_year, sample_lat, sample_lon, tif_dict, config
            )
        else:
            flag_north = False

        if flag_north:
            sample_input = 0.0
        else:

            central_output = read_tif_checking_fixed( 
                tif_dict["central"],
                channels,
                window=Window.from_slices((0, ilat_s + 1), (ilon_w, ilon_e + 1)),
                tile_len=config["tile_len_deg"],
            )
            north_output = read_tif_checking_fixed( 
                tif_dict["north"],
                channels,
                window=Window.from_slices(
                    (tif_dict["north_height"] + ilat_n, tif_dict["north_height"]),
                    (ilon_w, ilon_e + 1),
                ),
                tile_len=config["tile_len_deg"],
            )
            
            sample_input = np.vstack((north_output, central_output)) 
        

    # USECASE 3 - northeast corner
    elif usecase == "usecase_northeast":
        if tif_dict.get("north") is None:
            tif_dict, flag_north = fill_tif_dict(
                "north", sample_year, sample_lat, sample_lon, tif_dict, config
            )
        else:
            flag_north = False

        if tif_dict.get("northeast") is None:
            tif_dict, flag_northeast = fill_tif_dict(
                "northeast", sample_year, sample_lat, sample_lon, tif_dict, config
            )
        else:
            flag_northeast = False

        if tif_dict.get("east") is None:
            tif_dict, flag_east = fill_tif_dict(
                "east", sample_year, sample_lat, sample_lon, tif_dict, config
            )
        else:
            flag_east = False

        if any([flag_north, flag_northeast, flag_east]):
            sample_input = 0.0
        else:
            central_output = read_tif_checking_fixed( 
                tif_dict["central"],
                channels,
                window=Window.from_slices(
                    (0, ilat_s + 1), (ilon_w, tif_dict["central_width"])
                ),
                tile_len=config["tile_len_deg"], 
            )
            east_output = read_tif_checking_fixed( 
                tif_dict["east"],
                channels,
                window=Window.from_slices(
                    (0, ilat_s + 1), (0, ilon_e - tif_dict["central_width"] + 1)
                ),
                tile_len=config["tile_len_deg"], 
            )
            north_output = read_tif_checking_fixed( 
                tif_dict["north"],
                channels,
                window=Window.from_slices(
                    (tif_dict["north_height"] + ilat_n, tif_dict["north_height"]),
                    (ilon_w, tif_dict["north_width"]),
                ),
                tile_len=config["tile_len_deg"], 
            )
            northeast_output = read_tif_checking_fixed( 
                tif_dict["northeast"],
                channels,
                window=Window.from_slices(
                    (tif_dict["north_height"] + ilat_n, tif_dict["north_height"]),
                    (0, ilon_e - tif_dict["central_width"] + 1),
                ),
                tile_len=config["tile_len_deg"], 
            )

            sample_input = np.vstack(
                (
                    np.hstack((north_output, northeast_output)),
                    np.hstack((central_output, east_output)),
                )
            )

    # USECASE 4 - east edge
    elif usecase == "usecase_east":
        if tif_dict.get("east") is None:
            tif_dict, flag_east = fill_tif_dict(
                "east", sample_year, sample_lat, sample_lon, tif_dict, config
            )
        else:
            flag_east = False

        if flag_east:
            sample_input = 0.0
        else:
            central_output = read_tif_checking_fixed( 
                tif_dict["central"],
                channels,
                window=Window.from_slices(
                    (ilat_n, ilat_s + 1), (ilon_w, tif_dict["central_width"])
                ),
                tile_len=config["tile_len_deg"], 
            )
            east_output = read_tif_checking_fixed( 
                tif_dict["east"],
                channels,
                window=Window.from_slices(
                    (ilat_n, ilat_s + 1), (0, ilon_e - tif_dict["central_width"] + 1)
                ),
                tile_len=config["tile_len_deg"], 
            )

            sample_input = np.hstack((central_output, east_output))

    # USECASE 5 - southeast corner
    elif usecase == "usecase_southeast":
        if tif_dict.get("south") is None:
            tif_dict, flag_south = fill_tif_dict(
                "south", sample_year, sample_lat, sample_lon, tif_dict, config
            )
        else:
            flag_south = False

        if tif_dict.get("southeast") is None:
            tif_dict, flag_southeast = fill_tif_dict(
                "southeast", sample_year, sample_lat, sample_lon, tif_dict, config
            )
        else:
            flag_southeast = False

        if tif_dict.get("east") is None:
            tif_dict, flag_east = fill_tif_dict(
                "east", sample_year, sample_lat, sample_lon, tif_dict, config
            )
        else:
            flag_east = False

        if any([flag_south, flag_southeast, flag_east]):
            sample_input = 0.0
        else:
            central_output = read_tif_checking_fixed( 
                tif_dict["central"],
                channels,
                window=Window.from_slices(
                    (ilat_n, tif_dict["central_height"]),
                    (ilon_w, tif_dict["central_width"]),
                ),
                tile_len=config["tile_len_deg"], 
            )
            east_output = read_tif_checking_fixed( 
                tif_dict["east"],
                channels,
                window=Window.from_slices(
                    (ilat_n, tif_dict["central_height"]),
                    (0, ilon_e - tif_dict["central_width"] + 1),
                ),
                tile_len=config["tile_len_deg"], 
            )
            south_output = read_tif_checking_fixed( 
                tif_dict["south"],
                channels,
                window=Window.from_slices(
                    (0, ilat_s - tif_dict["central_height"] + 1),
                    (ilon_w, tif_dict["south_width"]),
                ),
                tile_len=config["tile_len_deg"], 
            )
            southeast_output = read_tif_checking_fixed( 
                tif_dict["southeast"],
                channels,
                window=Window.from_slices(
                    (0, ilat_s - tif_dict["central_height"] + 1),
                    (0, ilon_e - tif_dict["central_width"] + 1),
                ),
                tile_len=config["tile_len_deg"], 
            )

            sample_input = np.vstack(
                (
                    np.hstack((central_output, east_output)),
                    np.hstack((south_output, southeast_output)),
                )
            )

    # USECASE 6 - south edge
    elif usecase == "usecase_south":
        if tif_dict.get("south") is None:
            tif_dict, flag_south = fill_tif_dict(
                "south", sample_year, sample_lat, sample_lon, tif_dict, config
            )
        else:
            flag_south = False

        if flag_south:
            sample_input = 0.0
        else:
            central_output = read_tif_checking_fixed( 
                tif_dict["central"],
                channels,
                window=Window.from_slices(
                    (ilat_n, tif_dict["central_height"]), (ilon_w, ilon_e + 1)
                ),
                tile_len=config["tile_len_deg"], 
            )
            south_output = read_tif_checking_fixed( 
                tif_dict["south"],
                channels,
                window=Window.from_slices(
                    (0, ilat_s - tif_dict["central_height"] + 1), (ilon_w, ilon_e + 1)
                ),
                tile_len=config["tile_len_deg"], 
            )
            print("")
            sample_input = np.vstack((central_output, south_output))

    # USECASE 7 - southwest corner
    elif usecase == "usecase_southwest":
        if tif_dict.get("south") is None:
            tif_dict, flag_south = fill_tif_dict(
                "south", sample_year, sample_lat, sample_lon, tif_dict, config
            )
        else:
            flag_south = False

        if tif_dict.get("southwest") is None:
            tif_dict, flag_southwest = fill_tif_dict(
                "southwest", sample_year, sample_lat, sample_lon, tif_dict, config
            )
        else:
            flag_southwest = False

        if tif_dict.get("west") is None:
            tif_dict, flag_west = fill_tif_dict(
                "west", sample_year, sample_lat, sample_lon, tif_dict, config
            )
        else:
            flag_west = False

        if any([flag_south, flag_southwest, flag_west]):
            sample_input = 0.0
        else:
            central_output = read_tif_checking_fixed( 
                tif_dict["central"],
                channels,
                window=Window.from_slices(
                    (ilat_n, tif_dict["central_height"]), (0, ilon_e + 1)
                ),
                tile_len=config["tile_len_deg"], 
            )
            west_output = read_tif_checking_fixed( 
                tif_dict["west"],
                channels,
                window=Window.from_slices(
                    (ilat_n, tif_dict["west_height"]),
                    (tif_dict["west_width"] + ilon_w, tif_dict["west_width"]),
                ),
                tile_len=config["tile_len_deg"], 
            )
            south_output = read_tif_checking_fixed( 
                tif_dict["south"],
                channels,
                window=Window.from_slices(
                    (0, ilat_s - tif_dict["central_height"] + 1), (0, ilon_e + 1)
                ),
                tile_len=config["tile_len_deg"], 
            )
            southwest_output = read_tif_checking_fixed( 
                tif_dict["southwest"],
                channels,
                window=Window.from_slices(
                    (0, ilat_s - tif_dict["central_height"] + 1),
                    (tif_dict["west_width"] + ilon_w, tif_dict["west_width"]),
                ),
                tile_len=config["tile_len_deg"], 
            )

            sample_input = np.vstack(
                (
                    np.hstack((west_output, central_output)),
                    np.hstack((southwest_output, south_output)),
                )
            )

    # USECASE 8 - west edge
    elif usecase == "usecase_west":
        if tif_dict.get("west") is None:
            tif_dict, flag_west = fill_tif_dict(
                "west", sample_year, sample_lat, sample_lon, tif_dict, config
            )
        else:
            flag_west = False

        if flag_west:
            sample_input = 0.0
        else:
            
            central_output = read_tif_checking_fixed( 
                tif_dict["central"],
                channels,
                window=Window.from_slices((ilat_n, ilat_s + 1), (0, ilon_e + 1)),
                tile_len=config["tile_len_deg"], 
            )
            west_output = read_tif_checking_fixed( 
                tif_dict["west"],
                channels,
                window=Window.from_slices(
                    (ilat_n, ilat_s + 1),
                    (tif_dict["west_width"] + ilon_w, tif_dict["west_width"]),
                ),
                tile_len=config["tile_len_deg"], 
            )
 
            sample_input = np.hstack((west_output, central_output))
            

    else:
        raise NotImplementedError("such a case does not exist. something is wrong.")

    if isinstance(sample_input, float):
        return 0.0, tif_dict, usecase
    else:
        assert (
            sample_input.shape[0] == config["scene_width_landsat"]
        ), f"{sample_input.shape[0]=}, {usecase=}"
        assert (
            sample_input.shape[1] == config["scene_width_landsat"]
        ), f"{sample_input.shape[1]=}, {usecase=}"

        # add noise to de-noise
        if config["mode"] == "training":
            random_noise = rng.integers(
                -config["architecture"]["input_noise"],
                config["architecture"]["input_noise"] + 1,
                size=sample_input.shape[-1],
            )
            sample_input = sample_input + random_noise

            # Make negative values zero
            # On the one hand, this makes sense because in reality there will not be negative values.
            # On the other hand, this might be a problem for the model to learn the relative relationships.
            # Will comment it out for now.
            # sample_input[sample_input < 0] = 0

        return sample_input, tif_dict, usecase
