import os
import numpy as np
import argparse
import torch
import rasterio
import ast

from utils import utils
from data_builder import build_tags
from data_builder import data_loader
from predictor import inference
from data_builder import read_landsat, data_methods

REWRITE = True


def main():
    """
    This script performs inference using a pre-trained model on Landsat data.
    It loads the model, processes Landsat tiles, makes predictions, and saves the results as TIFF files.
    The predictions are then tiled together to create a mosaic image.

    Usage on CUDA: CUDA_VISIBLE_DEVICES=1, python oracle.py <expname> <gpu_id=0>
    Usage on MPS: python oracle.py <expname>

    Arguments:
        expname (str): Experiment name to specify the config file, e.g. exp101
        gpu_id (int; OPTIONAL): GPU device ID (number)
        inference_region ("list or dict";OPTIONAL): if provided, override inference region from config file
        inference_years ("list" ;OPTIONAL): if provided, override inference year(s) from config file

    Example:
        python oracle.py exp101 0 "[10,20,30,50]" "[2019,2010]"
    """

    # print(f"python version = {sys.version}")
    # print(f"numpy version = {np.__version__}")
    # print(f"pytorch version = {torch.__version__}")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "expname", help="experiment name to specify the config file, e.g. exp101"
    )
    parser.add_argument(
        "gpu_id",
        help="GPU device ID (number), e.g. 1 [default=0]",
        nargs="?",
        default="0",
    )

    # Add optional arguments
    parser.add_argument(
        "inference_region",
        type=str,
        nargs="?",
        default=None,
        help='Region for inference (list or dict - in "quotes")',
    )
    parser.add_argument(
        "inference_years",
        type=str,
        nargs="?",
        default=None,
        help='Years for inference (list) - in "quotes"',
    )

    args = parser.parse_args()

    # SET config
    config = utils.get_config(args.expname)
    config["mode"] = "inference"
    config["device_id"] = args.gpu_id

    # Convert string arguments to appropriate types using ast.literal_eval, if provided
    if args.inference_region is not None:
        try:
            config["data"]["inference_region"] = ast.literal_eval(args.inference_region)
        except (ValueError, SyntaxError) as e:
            raise ValueError(f"Invalid input for inference_region: {e}")

    if args.inference_years is not None:
        try:
            config["data"]["inference_years"] = ast.literal_eval(args.inference_years)
        except (ValueError, SyntaxError) as e:
            raise ValueError(f"Invalid input for inference_years: {e}")

    print(args.expname)

    # GET THE DATA
    lats_lons_dict = data_methods.get_lats_lons_list(
        region=config["data"]["inference_region"], tile_len_deg=config["tile_len_deg"]
    )
    print(
        f"\nInference Tile Bounds: \n"
        f"  {lats_lons_dict['lats']=} \n"
        f"  {lats_lons_dict['lons']=} "
    )

    # load the model
    model = utils.load_model(config, clean=False)

    for year in config["data"]["inference_years"]:
        print(" --- " + str(year) + "---")
        config["data"]["inference_years"] = (year,)
        filenames_list = []

        for lats_list, lons_list in zip(lats_lons_dict["lats"], lats_lons_dict["lons"]):

            for latfile in lats_list:
                for lonfile in lons_list:

                    # check if landsat tile exists, if so, get it.
                    config["tile"] = (
                        latfile - config["tile_len_deg"],
                        latfile,
                        lonfile,
                        lonfile + config["tile_len_deg"],
                    )
                    landsat_file = read_landsat.get_input_filename(
                        config["data"]["inference_years"],
                        (latfile,),
                        (lonfile,),
                        config,
                    )

                    if (
                        os.path.isfile(
                            utils.get_directories(config["machine"])["landsat_dir"]
                            + landsat_file[0]
                            + ".tif"
                        )
                        is False
                    ):
                        continue

                    # check if prediction file already exists
                    predictions_filename = utils.get_predictions_filename(
                        config, landsat_file[0]
                    )
                    filenames_list.append(predictions_filename)
                    if os.path.isfile(predictions_filename) and REWRITE is False:
                        print("prediction file already exists: ", predictions_filename)
                        continue
                    print(landsat_file[0])

                    # GET THE SAMPLE TAGS
                    tags, __ = build_tags.get_tags(config)
                    if len(tags.years) == 0:
                        # all water, so skip
                        continue

                    # MAKE PREDICTIONS and SAVE AS TIFF
                    ds_inf = data_loader.CustomData(
                        config,
                        tags.years,
                        tags.lats,
                        tags.lons,
                        tags.files,
                        tags.dict,
                        config["inference"]["batch_size"],
                    )
                    inf_loader = torch.utils.data.DataLoader(
                        ds_inf,
                        batch_size=None,
                        batch_sampler=None,
                        shuffle=False,
                        drop_last=False,
                        pin_memory=config["inference"]["pin_memory"],
                        num_workers=config["inference"]["num_workers"],
                    )
                    hfi_predict, _, latlon_bounds = inference.make_predictions(
                        config, model, tags, inf_loader
                    )

                    # save the predictions
                    _ = inference.save_predictions_tif(
                        hfi_predict,
                        predictions_filename,
                        latlon_bounds=latlon_bounds,
                    )

                    # close tif files
                    for key in ds_inf.tif_dict.keys():
                        if isinstance(ds_inf.tif_dict[key], rasterio.io.DatasetReader):
                            ds_inf.tif_dict[key].close()
                    ds_inf.tif_dict = {}

                    filenames_list.append(predictions_filename)

        # TILE THE PREDICTIONS TOGETHER
        model_name = utils.get_model_name(config["expname"], config["seed"])
        mosaic_filename = (
            utils.get_directories(config["machine"])["mosaics_dir"]
            + model_name
            + "_"
            + str(config["data"]["inference_years"][0])
            + "_mlhfi_mosaic.tif"
        )
        mosaic, mosaic_trans = inference.create_mosaic(filenames_list)
        _ = inference.save_predictions_tif(mosaic, mosaic_filename, trans=mosaic_trans)
        print("mosaic saved.")


if __name__ == "__main__":
    main()
