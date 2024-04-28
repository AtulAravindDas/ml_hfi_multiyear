import os
import utils

dirs = utils.get_directories(config["machine"])

for key in dirs:

    mkdir_folder = dirs[key]
    os.system("mkdir " + mkdir_folder)

    if key == "data_dir":
        print(key)
        print("   --> add hfi labels here (and other data in subdirecties).")

    elif key == "landsat_dir":
        print(key)
        print("   --> add landsat data here.")

    elif key == "shapefiles_dir":
        print(key)
        print("   --> add shapefile data (.pkl, .tif).")
