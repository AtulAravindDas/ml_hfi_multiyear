# Multi-year ml-HFI
***
Deep-learing estimate of the human footprint index (HFI or HII; also known as the Human Impact Index as trained here) throughout the past decade(s) as estimated from Landsat imagery.

This work relies entirely on the efforts of [0] to create a labeled dataset of 2nd generation human footprint [1]. This puts the resolution of the HFI at ~300m.

## Tensorflow Code
***
This code was written in python 3.10.10 and tensorflow 2.10 and can be significantly sped-up using GPUs.

### Python Environment
The following python environment was used to implement this code.
```
- conda create --name env-mlhfi python=3.10.10
- conda activate env-mlhfi
- conda install -c apple -c conda-forge -c nodefaults tensorflow-deps
- python -m pip install tensorflow-macos==2.10.0
- python -m pip install tensorflow-metal==0.6.0
- conda install numpy scipy matplotlib xarray scikit-learn datashader jupyterlab
- pip install silence-tensorflow palettable rasterio rioxarray
- conda install -c conda-forge dask netCDF4
- conda install -c conda-forge cartopy
- conda install palettable seaborn
```

## Get Started
* Run ``init.py`` to create necessary directories.
* Fill required data into directories (e.g. google drive links).
    * landsat tiles
    * 2015-2020 HII labels
    * at minimum you will need ``shapefile_mosaic.tif`` and ``shapefile_dataframe.pkl`` (_you will not need the rest of the shapefiles unless you want to recreate the shapefile masks_)
* Set ``experiment_settings.py`` as desired.
* Order to run major scripts.
  * _driver.ipynb
  * _oracle.ipynb
  * _analysis.ipynb
  * _assesor.ipynb

## Credits
***
This work is a collaborative effort between Dr. Patrick Keys, Dr. Frances Davenport, Dr. Randal Barnes and Dr. Elizabeth Barnes.

### References
* [0] HII References: https://wcshumanfootprint.org/
* [1] HII Labels: https://wcshumanfootprint.org/data-access

### License
This project is licensed under an MIT license.

MIT © [Elizabeth A. Barnes](https://github.com/eabarnes1010)




