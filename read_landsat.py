"""Read the landsat data.

Functions
---------
fill_tif_dict(name, filename, dict)
read_input_data(self, tif_dict, sample_lon, sample_lat, channels, scene_width)

"""

import numpy as np
import rasterio
from rasterio.windows import Window
import methods

directory_paths = methods.get_directories()
LANDSAT_DIRECTORY = directory_paths["landsat_dir"]


def fill_tif_dict(name, sample_year, sample_lat, sample_lon, tif_dict, settings):

    if name == "central":
        filename = get_input_filename((sample_year,), (sample_lat,), (sample_lon,), settings)
    elif name == "northwest":
        filename = get_input_filename((sample_year,), (sample_lat + settings["tile_len_deg"],), (sample_lon - settings["tile_len_deg"],), settings)
    elif name == "north":
        filename = get_input_filename((sample_year,), (sample_lat + settings["tile_len_deg"],), (sample_lon,), settings)
    elif name == "northeast":
        filename = get_input_filename((sample_year,), (sample_lat + settings["tile_len_deg"],), (sample_lon + settings["tile_len_deg"],), settings)
    elif name == "east":
        filename = get_input_filename((sample_year,), (sample_lat,), (sample_lon + settings["tile_len_deg"],), settings)
    elif name == "southeast":
        filename = get_input_filename((sample_year,), (sample_lat - settings["tile_len_deg"],), (sample_lon + settings["tile_len_deg"],), settings)
    elif name == "south":
        filename = get_input_filename((sample_year,), (sample_lat - settings["tile_len_deg"],), (sample_lon,), settings)
    elif name == "southwest":
        filename = get_input_filename((sample_year,), (sample_lat - settings["tile_len_deg"],), (sample_lon - settings["tile_len_deg"],), settings)
    elif name == "west":
        filename = get_input_filename((sample_year,), (sample_lat,), (sample_lon - settings["tile_len_deg"],), settings)
    else:
        raise NotImplementedError("no such name.")

    try:
        tif_dict[name] = rasterio.open(LANDSAT_DIRECTORY + filename[0] + ".tif")
        tif_dict[name + "_height"] = tif_dict[name].height
        tif_dict[name + "_width"] = tif_dict[name].width
    except:
        pass

    return tif_dict


def read_tif(tif, channels, window):
    return np.transpose(tif.read(channels, window=window), axes=(1, 2, 0))


def get_landsat_bounds(settings, region):
    lat_s = np.floor(region[0] / settings["tile_len_deg"]) * settings["tile_len_deg"]
    lat_n = np.ceil(region[1] / settings["tile_len_deg"]) * settings["tile_len_deg"]
    lon_w = np.floor(region[2] / settings["tile_len_deg"]) * settings["tile_len_deg"]
    lon_e = np.ceil(region[3] / settings["tile_len_deg"]) * settings["tile_len_deg"]

    return lat_s, lat_n, lon_w, lon_e


def get_input_filename(years, lats, lons, settings):

    filenames = []
    for isample in range(len(years)):
        file_year = int(years[isample])
        file_lat = int(np.ceil(lats[isample] / settings["tile_len_deg"]) * settings["tile_len_deg"])
        file_lon = int(np.floor(lons[isample] / settings["tile_len_deg"]) * settings["tile_len_deg"])
        filenames.append(f"landsat_{file_lat}lat_{file_lon}lon_{file_year}")

    return filenames


def read_input_data(settings, tif_dict, sample_year, sample_lon, sample_lat, channels, scene_width):

    # sample_lon = sample_lon.numpy()
    # sample_lat = sample_lat.numpy()

    ilat, ilon = tif_dict["central"].index(sample_lon, sample_lat)
    ilat_n, ilat_s = ilat - scene_width / 3, ilat + scene_width / 3 * 2 - 1
    ilon_w, ilon_e = ilon - scene_width / 3, ilon + scene_width / 3 * 2 - 1

    # determine the usecase
    if ilat_n >= 0 and ilon_w >= 0 and ilat_s < tif_dict["central_height"] and ilon_e < tif_dict["central_width"]:
        usecase = "usecase_central"
    elif ilat_n < 0 and ilon_w < 0:
        usecase = "usecase_northwest"
    elif ilat_n < 0 and ilon_w >= 0 and ilon_e < tif_dict["central_width"]:
        usecase = "usecase_north"
    elif ilat_n < 0 and ilon_w >= tif_dict["central_width"]:
        usecase = "usecase_northeast"
    elif ilat_n >= 0 and ilat_s < tif_dict["central_height"] and ilon_w >= tif_dict["central_width"]:
        usecase = "usecase_east"
    elif ilat_s >= tif_dict["central_height"] and ilon_w >= tif_dict["central_width"]:
        usecase = "usecase_southeast"
    elif ilon_w >= 0 and ilat_s >= tif_dict["central_height"] and ilon_w < tif_dict["central_width"]:
        usecase = "usecase_south"
    elif ilon_w < 0 and ilat_s >= tif_dict["central_height"]:
        usecase = "usecase_southwest"
    elif ilat_n >= 0 and ilon_w < 0 and ilat_s < tif_dict["central_height"]:
        usecase = "usecase_west"
    else:
        raise NotImplementedError("no such use case")

    # the speed of this code assumes that training samples will never be on edges or corners
    if settings["mode"] == "training":
        assert usecase == "usecase_central"

    # USECASE 0 - central only
    if usecase == "usecase_central":
        central_output = read_tif(tif_dict["central"], channels,
                                  window=Window.from_slices((ilat_n, ilat_s + 1), (ilon_w, ilon_e + 1)))
        sample_output = central_output

    # USECASE 1 - northwest corner
    elif usecase == "usecase_northwest":
        if tif_dict.get("north") is None:
            tif_dict = fill_tif_dict("north", sample_year, sample_lat, sample_lon, tif_dict, settings)
        if tif_dict.get("northwest") is None:
            tif_dict = fill_tif_dict("northwest", sample_year, sample_lat, sample_lon, tif_dict, settings)
        if tif_dict.get("west") is None:
            tif_dict = fill_tif_dict("west", sample_year, sample_lat, sample_lon, tif_dict, settings)

        central_output = read_tif(tif_dict["central"], channels,
                                  window=Window.from_slices((0, ilat_s + 1), (0, ilon_e + 1)))
        west_output = read_tif(tif_dict["west"], channels,
                               window=Window.from_slices((0, ilat_s + 1), (tif_dict["west_width"] + ilon_w, tif_dict["west_width"])))
        north_output = read_tif(tif_dict["north"], channels,
                                window=Window.from_slices((tif_dict["north_height"] + ilat_n, tif_dict["north_height"]), (0, ilon_e + 1)))
        northwest_output = read_tif(tif_dict["northwest"], channels,
                                    window=Window.from_slices((tif_dict["northwest_height"] + ilat_n, tif_dict["northwest_height"]), (tif_dict["northwest_width"] + ilon_e, tif_dict["northwest_width"])))

        sample_output = np.vstack((np.hstack((northwest_output, north_output)),
                                   np.hstack((west_output, central_output))))

    # USECASE 2 - north edge
    elif usecase == "usecase_north":
        if tif_dict.get("north") is None:
            tif_dict = fill_tif_dict("north", sample_year, sample_lat, sample_lon, tif_dict, settings)

        central_output = read_tif(tif_dict["central"], channels,
                                  window=Window.from_slices((0, ilat_s + 1), (ilon_w, ilon_e + 1)))
        north_output = read_tif(tif_dict["north"], channels,
                                window=Window.from_slices((tif_dict["north_height"] + ilat_n, tif_dict["north_height"]), (ilon_w, ilon_e + 1)))

        sample_output = np.vstack((north_output, central_output))

    # USECASE 3 - northeast corner
    elif usecase == "usecase_northeast":
        if tif_dict.get("north") is None:
            tif_dict = fill_tif_dict("north", sample_year, sample_lat, sample_lon, tif_dict, settings)
        if tif_dict.get("northeast") is None:
            tif_dict = fill_tif_dict("northeast", sample_year, sample_lat, sample_lon, tif_dict, settings)
        if tif_dict.get("east") is None:
            tif_dict = fill_tif_dict("east", sample_year, sample_lat, sample_lon, tif_dict, settings)

        central_output = read_tif(tif_dict["central"], channels,
                                  window=Window.from_slices((0, ilat_s + 1), (ilon_w, tif_dict["central_width"])))
        east_output = read_tif(tif_dict["east"], channels,
                               window=Window.from_slices((0, ilat_s + 1), (0, ilon_e - tif_dict["central_width"] + 1)))
        north_output = read_tif(tif_dict["north"], channels,
                                window=Window.from_slices((tif_dict["north_height"] + ilat_n, tif_dict["north_height"]), (ilon_w, tif_dict["north_width"])))
        northeast_output = read_tif(tif_dict["northeast"], channels,
                                    window=Window.from_slices((tif_dict["north_height"] + ilat_n, tif_dict["north_height"]), (0, ilon_e - tif_dict["central_width"] + 1)))

        sample_output = np.vstack((np.hstack((north_output, northeast_output)),
                                   np.hstack((central_output, east_output))))

    # USECASE 4 - east edge
    elif usecase == "usecase_east":
        if tif_dict.get("east") is None:
            tif_dict = fill_tif_dict("east", sample_year, sample_lat, sample_lon, tif_dict, settings)

        central_output = read_tif(tif_dict["central"], channels,
                                  window=Window.from_slices((ilat_n, ilat_s + 1), (ilon_w, tif_dict["central_width"])))
        east_output = read_tif(tif_dict["east"], channels,
                               window=Window.from_slices((ilat_n, ilat_s + 1), (0, ilon_e - tif_dict["central_width"] + 1)))

        sample_output = np.hstack((central_output, east_output))

    # USECASE 5 - southeast corner
    elif usecase == "usecase_southeast":
        if tif_dict.get("south") is None:
            tif_dict = fill_tif_dict("south", sample_year, sample_lat, sample_lon, tif_dict, settings)
        if tif_dict.get("southeast") is None:
            tif_dict = fill_tif_dict("southeast", sample_year, sample_lat, sample_lon, tif_dict, settings)
        if tif_dict.get("east") is None:
            tif_dict = fill_tif_dict("east", sample_year, sample_lat, sample_lon, tif_dict, settings)

        central_output = read_tif(tif_dict["central"], channels,
                                  window=Window.from_slices((ilat_n, tif_dict["central_height"]), (ilon_w, tif_dict["central_width"])))
        east_output = read_tif(tif_dict["east"], channels,
                               window=Window.from_slices((ilat_n, tif_dict["central_height"]), (0, ilon_e - tif_dict["central_width"] + 1)))
        south_output = read_tif(tif_dict["south"], channels,
                                window=Window.from_slices((0, ilat_s - tif_dict["central_height"] + 1), (ilon_w, tif_dict["south_width"])))
        southeast_output = read_tif(tif_dict["southeast"], channels,
                                    window=Window.from_slices((0, ilat_s - tif_dict["central_height"] + 1), (0, ilon_e - tif_dict["central_width"] + 1)))

        sample_output = np.vstack((np.hstack((central_output, east_output)),
                                   np.hstack((south_output, southeast_output))))

    # USECASE 6 - south edge
    elif usecase == "usecase_south":
        if tif_dict.get("south") is None:
            tif_dict = fill_tif_dict("south", sample_year, sample_lat, sample_lon, tif_dict, settings)

        central_output = read_tif(tif_dict["central"], channels,
                                  window=Window.from_slices((ilat_n, tif_dict["central_height"]), (ilon_w, ilon_e + 1)))
        south_output = read_tif(tif_dict["south"], channels,
                                window=Window.from_slices((0, ilat_s - tif_dict["central_height"] + 1), (ilon_w, ilon_e + 1)))

        sample_output = np.vstack((central_output, south_output))

    # USECASE 7 - southwest corner
    elif usecase == "usecase_southwest":
        if tif_dict.get("south") is None:
            tif_dict = fill_tif_dict("south", sample_year, sample_lat, sample_lon, tif_dict, settings)
        if tif_dict.get("southwest") is None:
            tif_dict = fill_tif_dict("southwest", sample_year, sample_lat, sample_lon, tif_dict, settings)
        if tif_dict.get("west") is None:
            tif_dict = fill_tif_dict("west", sample_year, sample_lat, sample_lon, tif_dict, settings)

        central_output = read_tif(tif_dict["central"], channels,
                                  window=Window.from_slices((ilat_n, tif_dict["central_height"]), (0, ilon_e + 1)))
        west_output = read_tif(tif_dict["west"], channels,
                               window=Window.from_slices((ilat_n, tif_dict["west_height"]), (tif_dict["west_width"] + ilon_w, tif_dict["west_width"])))
        south_output = read_tif(tif_dict["south"], channels,
                                window=Window.from_slices((0, ilat_s - tif_dict["central_height"] + 1), (0, ilon_e + 1)))
        southwest_output = read_tif(tif_dict["southwest"], channels,
                                    window=Window.from_slices((0, ilat_s - tif_dict["central_height"] + 1), (tif_dict["west_width"] + ilon_w, tif_dict["west_width"])))

        sample_output = np.vstack((np.hstack((west_output, central_output)),
                                   np.hstack((southwest_output, south_output))))

    # USECASE 8 - west edge
    elif usecase == "usecase_west":
        if tif_dict.get("west") is None:
            tif_dict = fill_tif_dict("west", sample_year, sample_lat, sample_lon, tif_dict, settings)

        central_output = read_tif(tif_dict["central"], channels,
                                  window=Window.from_slices((ilat_n, ilat_s + 1), (0, ilon_e + 1)))
        west_output = read_tif(tif_dict["west"], channels,
                               window=Window.from_slices((ilat_n, ilat_s + 1), (tif_dict["west_width"] + ilon_w, tif_dict["west_width"])))

        sample_output = np.hstack((west_output, central_output))
    else:
        raise NotImplementedError("such a case does not exist. something is wrong.")

    # print(usecase, sample_lat, sample_lon, ilat_s, ilat_s, ilon_w, ilon_e,)
    assert sample_output.shape[0] == scene_width, f"{sample_output.shape[0] = }, {usecase = }"
    assert sample_output.shape[1] == scene_width, f"{sample_output.shape[1] = }, {usecase = }"

    # add noise to de-noise
    rng = np.random.default_rng()
    if settings["mode"] == "training":
        random_noise = rng.integers(
            -settings["input_noise"], settings["input_noise"] + 1, size=1
        )
        sample_output = sample_output + random_noise

    return sample_output, tif_dict
