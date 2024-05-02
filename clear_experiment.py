import os
import argparse
from utils import utils


def main():
    """
    Clear the all existing directories and files for a specific experiment + seed.

    Parameters
    ----------
    expname (str): Experiment name to specify the config file, e.g. exp101

    Returns
    -------
    None

    Example
    -------
        python clear_experiment.py <exp101> <--dry-run>
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "expname", help="experiment name to specify the config file, e.g. exp101"
    )
    parser.add_argument(
        "-d", "--dry-run", action="store_true", help="perform a dry run without deleting any files"
    )
    args = parser.parse_args()
    config = utils.get_config(args.expname)

    model_name = utils.get_model_name(config["expname"], config["seed"])
    directory_paths = utils.get_directories(config["machine"])

    for key in ("save_model_dir", "predictions_dir"):
        dir = directory_paths[key] + config["expname"] + "/" + model_name + "/"
        if os.path.exists(dir):
            print("rm -r ", dir)
            if not args.dry_run:
                os.system("rm -r " + dir)

        dir = directory_paths[key] + config["expname"] + "/"
        if os.path.exists(dir):
            if not args.dry_run:
                os.system("rm " + dir + "Icon?")
                os.system("rm " + dir + ".DS_Store")

            if not os.listdir(dir):
                print("rm -r ", dir)
                if not args.dry_run:
                    os.system("rm -r " + dir)

    for key in ("figures_dir", "mosaics_dir"):
        dir = directory_paths[key]
        if os.path.exists(dir):
            print("rm " + dir + model_name + "_*")
            if not args.dry_run:
                os.system("rm " + dir + model_name + "_*")


if __name__ == "__main__":
    main()
