# Multi-year High-Resolution ml-HFI

This project provides a deep-learning estimate of the Human Footprint Index (HFI or HII), trained using Landsat imagery. The resolution of the HFI is approximately 300m, based on the labeled dataset of 2nd generation human footprint created by [0].

## Requirements

This project is implemented using Python 3.12.0 and PyTorch 2.1.2. The performance can be significantly improved by using GPUs, assuming the Landsat data is quickly readable.

## Setting Up the Python Environment

You can set up the Python environment for this project by running the following commands:

```bash
conda create --name env-torch-mlhfi python=3.12.0
conda activate env-torch-mlhfi
conda install numpy scipy pandas matplotlib palettable flake8 jupyterlab black jupyterlab_code_formatter xarray scikit-learn datashader netCDF4 cartopy pytorch torchvision
conda install -c conda-forge dask
pip install ipython-autotime cmocean cmasher cmaps torchinfo rasterio rioxarray seaborn vit-pytorch
```

## Getting Started

Follow these steps to get started with the project:

1. Run `init.py` to create the necessary directories.
2. Download the required data and place it into the appropriate directories. You will need at least the following files:
    - Landsat tiles
    - 2015-2020 HII labels
    - `shapefile_mosaic.tif`
    - `shapefile_dataframe.pkl` (Note: You won't need the rest of the shapefiles unless you want to recreate the shapefile masks)
3. Configure `config_[###].py` as per your requirements.
4. Run the major scripts in the following order:
    - `driver.py`
    - `oracle.py`
    - `assesor.py`

Please refer to the individual script files for more detailed instructions.

## Configuration File
The configuration file (config_###.json) documentation. It is still a work in progress...

- `expname` (str): The name of the experiment.
- `machine` (str: "falco", "riviera", "blackforest"): The name or identifier of the machine where the experiment is run.
- `seed` (int): The seed used for random number generation.
- `device` (str: "cpu", "gpu"): The device on which the computations are performed. This could be 'cpu', 'gpu'.

- `data`: This section contains all the configuration related to building the data.
    - `tags_loadname` (str: default="null"): The name of the experiment from which to load the tags if it already exists.
    - `training_years` (list): The years used for training. Must have associated HII labels.
    - `training_region` (list or dict): The geographical region to grab training tiles. There are two options:
        - (list) Coordinates of the bounding box in the order [lat_s, lat_n, lon_w, lon_e].
        - (dict) Keys 'lats' and 'lons' which represent the latitude and longitude values respectively.
        {"tilelats": [[0], [10], [30], [40], [50]],
        "tilelons": [[0, 10], [0, 10, -30], [0, 50, 20], [0, 10], [0]]
        }
    - `inference_years` (list): The years used for inference.
    - `inference_region` (list or dict): The geographical region to grab inference tiles. There are two options:
        - (list) Coordinates of the bounding box in the order [lat_s, lat_n, lon_w, lon_e].
        - (dict) Keys 'lats' and 'lons' which represent the latitude and longitude values respectively.
        {"tilelats": [[0], [10], [30], [40], [50]],
        "tilelons": [[0, 10], [0, 10, -30], [0, 50, 20], [0, 10], [0]]
        }
    - `channels` (list): Landsat channels to use in training. Indexing starts at 1, not zero.
    - `scene_width` (int): Width and height in HII pixels of each multi-channel input image. Image is assumed to be square.

    <!---
    -->


## Helpful commands
* Gets rid of Apple Icons that mess up github: ``find . -name "Icon?" -print0 | xargs -0 rm -rf``
* Restarts VSCode Server: ``rm -rf ~/.vscode-server``


## Credits
***
This work is a collaborative effort between Dr. Bryam Orihuela Pinto, Dr. Patrick Keys, Dr. Frances Davenport, Dr. Randal Barnes and Dr. Elizabeth Barnes.

### References
* [0] HII References: https://wcshumanfootprint.org/
* [1] HII Labels: https://wcshumanfootprint.org/data-access

### License
This project is licensed under an MIT license.

MIT © [Elizabeth A. Barnes](https://github.com/eabarnes1010)




