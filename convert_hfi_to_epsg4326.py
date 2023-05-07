# This code needs to be only run once, and it has been.

import xarray as xr
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
dst_crs = 'EPSG:4326'


data_directory = "data/"

for year in (2000, 2005, 2010, 2013):

    print('year = ' + str(year))

    fileName = 'hfi' + str(year) + '_merisINT.tif'

    with rasterio.open(data_directory + fileName) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })

        with rasterio.open(data_directory + fileName[:-3] + 'epsg4326.tif', 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest)