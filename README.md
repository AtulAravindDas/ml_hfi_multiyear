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

## General Training Approach

A list of tile filenames defines the dataset that is iterated over. For each epoch, a batch of pixels is selected from each tile in the list of tile filenames. Increasing the number of batches to sample from each tile just involves repeating filenames in the tile filenames list. This is set by the parameter `n_repeat_tile`. The list of filenames is shuffled at the end of each epoch.

This same approach is used for the validation data. That is, a list of tile filenames are iterated over, grabbing a batch from each tile. By default, the validation tiles are only sampled over one time.

## General Inference Approach

Predictions are made, in a consecutive fashion, for every pixel in the tile. Since this can take a long time there is an additional option to take a `quicklook` at the data. This means only make predictions at equal intervals of pixels.

## Configuration File

The configuration file (config_[###].json) documentation. It is still a work in progress...

- `expname` (str): Name of the experiment.
- `machine` (str: "falco", "riviera", "blackforest"): Name or identifier of the machine where the experiment is run.
- `seed` (int): Seed used for random number generation.
- `device` (str: "cpu", "gpu"): Device on which the computations are performed. This could be 'cpu', 'gpu'.

- `data`: Configuration for building the data.
  - `tags_loadname` (str: default="null"): The name of the experiment from which to load the tags if it already exists.
  - `training_years` (list): Years used for training. Must have associated HII labels.
  - `training_region` (list or dict): The geographical region to grab training tiles. There are two options:
    - (list) Coordinates of the bounding box in the order [lat_s, lat_n, lon_w, lon_e].
    - (dict) Keys 'lats' and 'lons' which represent the latitude and longitude values respectively.
        {"tilelats": [[0], [10], [30], [40], [50]],
        "tilelons": [[0, 10], [0, 10, -30], [0, 50, 20], [0, 10], [0]]
        }
  - `inference_years` (list): Years used for inference.
  - `inference_region` (list or dict): Geographical region from which to grab inference tiles. There are two options:
    - (list) Coordinates of the bounding box in the order [lat_s, lat_n, lon_w, lon_e].
    - (dict) Keys 'lats' and 'lons' which represent the latitude and longitude values respectively.
        {"tilelats": [[0], [10], [30], [40], [50]],
        "tilelons": [[0, 10], [0, 10, -30], [0, 50, 20], [0, 10], [0]]
        }
  - `channels` (list): Landsat channels to use in training. Indexing starts at 1, not zero.
  - `scene_width` (int): Width and height in HII pixels of each multi-channel input image. Image is assumed to be square.

  - `percentage_sampling` (float: (0, 1.0]): percentage of each tile to sample when creating the training set
  - `val_frac` (float: (0, 1.0)): fraction of the sampled training set to be set aside for validation
  - `min_binfrac_for_tile` (float: (0, 1.0)): minimum fraction of the requested samples per bin to keep the bin; if a bin does not have at least this fraction of the requested samples the entire tile is thrown out of the training set (unless in the upper two bins, in which case this is ignored)
  - `min_samples_in_tile` (int): minimum number of samples in a tile, otherwise the tile is thrown out
  - `oversample_rare_bins` (boolean): whether to oversample bins without the requested number of samples
  - `oversample_rate` (float): rate at which to oversample the rare bins; a value of 1.0 means no oversampling, a value of 10 means oversample ten times the number of samples.

- `architecture`: Configuration for building the model architecture.
  - `type` (str: "cnn", "resnet", "vit"): each type has different config keys to set.
    - "cnn": `filters`, `cnn_activation`, `kernel_size`, `skip_channels`, `dense_in`, `dense_units`, `dense_activations`, `dropout`, `final_in`, `final_activation`
    - "resnet": `resnet_pretrained`, `resnet_trainable`, `resnet_drop_layer`, `skip_channels`, `dense_in`, `dense_units`, `dense_activations`, `dropout`, `final_in`, `final_activation`
    - "vit": `n_conv_layers`, `kernel_size`
  - `input_noise` (int): Maximum amount of noise to add to each landsat image during training. Noise is added according to a uniform distribution from [-input_noise, input_noise]

- `trainer`: Configuration for training the model.
  - `resume training` (boolean): whether to load the saved model and continue training, or start with a clean model
  - `num_workers` (int): number of workers to spin-up
  - `pin_memory` (boolean): whether to pin memory

  - `batch_size` (int): batch size for training data
  - `n_repeat_tile` (int): number of times to sample data from each tile per epoch
  - `max_batches` (int): maximum number of batches per epoch. Set to a large value for standard training so it will not be invoked
  - `max_epochs` (int): maximum number of epochs to train. Note that this, as well as the early stopping option below, will determine how many epochs are trained.

  - `val_batch_size` (int): batch size for validation data
  - `val_n_repeat_tile` (int): number of times to sample data from each tile during validation inference at the end of an epoch
  - `val_max_batches` (int): maxmim number of batches during validation inference. Set to a large value for standard training so it will not be invoked

  - `loss`: configuration for custom loss function
    - `type` (str: default = "WeightedSMSELoss"): name of custom loss function
    - `args`:
      - `zero_weighting` (float: default="null"): how much extra to weight the loss for values near zero
      - `zero_threshold` (float: default=0.0): threshold to define what a value near zero is. All values below this threshold will be weighted.
      - `one_weighting` (float: default="null"): how much extra to weight the loss for values near one
      - `one_threshold` (float: default=0.9): threshold to define what a value near one is. All values above this threshold will be weighted.
      - `kluge_value_for_zero` (float: default=0.0): extra error to add to values that are exactly zero as in Keys et al. (2020) to pull values to zero. This does not use zero_threshold above, but only applies to pixels with labels that are exactly zero.

  - `early_stopping`: configuration for early stopping class
    - `args`:
      - `patience` (int): how long to wait for loss to drop until training is stopped
      - `min_delta` (float): minimum delta loss must reduce

- `optimizer`: Refer to the [PyTorch optimizer docs](https://pytorch.org/docs/stable/optim.html) for more details.
  - `type`: type of optimizer to use
  - `args`: args specific to the optimizer

- `lr_scheduler`: Refer to the [PyTorch learning rate scheduler docs](https://pytorch.org/docs/stable/optim.html#module-torch.optim.lr_scheduler) for more details.
  - `type`: type of learning rate scheduler to use
  - `args`: args specific to the scheduler

- `metrics` (list): custom metrics to evaluate during training

- `inference`: Configuration for inference.
  - `num_workers` (int): number of workers to spin-up
  - `pin_memory` (boolean): whether to pin memory
  - `batch_size` (int): batch size for model.predict()
  - `quicklook` (boolean): whether to skip pixels to speed-up inference
  - `quicklook_skiplen` (int): number of pixels to skip

## Helpful commands

- Gets rid of Apple Icons that mess up github: ``find . -name "Icon?" -print0 | xargs -0 rm -rf``
- Restarts VSCode Server: ``rm -rf ~/.vscode-server``

## Credits

***
This work is a collaborative effort between Dr. Bryam Orihuela Pinto, Dr. Patrick Keys, Dr. Frances Davenport, Dr. Randal Barnes and Dr. Elizabeth Barnes.

### References

- [0] HII References: <https://wcshumanfootprint.org/>
- [1] HII Labels: <https://wcshumanfootprint.org/data-access>

### License

This project is licensed under an MIT license.

MIT © [Elizabeth A. Barnes](https://github.com/eabarnes1010)
