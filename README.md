# Multi-year High-Resolution ml-HFI
***
Deep-learning estimate of the human footprint index (HFI or HII; also known as the Human Impact Index as trained here) throughout the past decade(s) as estimated from Landsat imagery.

This work relies entirely on the efforts of [0] to create a labeled dataset of 2nd generation human footprint [1]. This puts the resolution of the HFI at ~300m.

## Tensorflow Code
***
This code was written in python 3.12.0 and pytorch 2.1.2.post3 and can be significantly sped-up using GPUs.

### Python Environment - Pytorch
The following python environment was used to implement this code.
```
conda create --name env-torch-mlhfi python=3.12.0
conda activate env-torch-mlhfi
conda install numpy scipy pandas matplotlib palettable flake8 jupyterlab black jupyterlab_code_formatter xarray scikit-learn datashader netCDF4 cartopy pytorch
torchvision
conda install -c conda-forge dask
pip install ipython-autotime cmocean cmasher cmaps torchinfo rasterio rioxarray
conda install torchvision
pip install seaborn
```

## Get Started
* Run ``init.py`` to create necessary directories.
* Fill required data into directories (e.g. google drive links).
    * landsat tiles
    * 2015-2020 HII labels
    * at minimum you will need ``shapefile_mosaic.tif`` and ``shapefile_dataframe.pkl`` (_you will not need the rest of the shapefiles unless you want to recreate the shapefile masks_)
* Set ``config_[###].py`` as desired.
* Order to run major scripts.
  * _driver.ipynb
  * _oracle.ipynb
  * _assesor.ipynb

## Credits
***
This work is a collaborative effort between Dr. Bryam Orihuela Pinto, Dr. Patrick Keys, Dr. Frances Davenport, Dr. Randal Barnes and Dr. Elizabeth Barnes.

### References
* [0] HII References: https://wcshumanfootprint.org/
* [1] HII Labels: https://wcshumanfootprint.org/data-access

### License
This project is licensed under an MIT license.

MIT © [Elizabeth A. Barnes](https://github.com/eabarnes1010)




