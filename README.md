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




