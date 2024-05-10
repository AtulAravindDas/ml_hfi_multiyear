import argparse
import warnings

import matplotlib.pyplot as plt  # type: ignore
import torch
import torchinfo
import numpy as np

import trainer.losses as loss_module
import trainer.metrics as metrics_module
from data_builder import build_tags, data_loader
from trainer.trainer import Trainer
from utils import utils
from visualizer import plots

warnings.filterwarnings("ignore")
torch.set_warn_always(False)

# TODO: figure out what an assertion error was thrown for not use_case central for exp003 data


def main():
    """
    Main function for training and evaluating a model.

    Usage on CUDA: CUDA_VISIBLE_DEVICES=1, python driver.py <expname> <gpu_id=0>
    Usage on MPS: python oracle.py <expname>

    Args:
        expname (str): Experiment name to specify the config file, e.g. exp101.
        gpu_id (str; OPTIONAL): GPU device ID (number), e.g. 1 [default=0].

    Returns:
        None
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
    args = parser.parse_args()

    # GET config
    config = utils.get_config(args.expname)
    config["mode"] = "training"
    config["device_id"] = args.gpu_id

    print("Running on Machine: " + config["machine"])

    # SET RANDOM SEEDS
    utils.set_random_seeds(config["seed"])

    # LOAD THE DATA
    print("Building the data.")
    tags_train, tags_val = build_tags.get_tags(config)

    # training data
    ds_train = data_loader.CustomData(
        config,
        tags_train.years,
        tags_train.lats,
        tags_train.lons,
        tags_train.files,
        tags_train.dict,
        config["trainer"]["batch_size"],
    )
    train_loader = torch.utils.data.DataLoader(
        ds_train,
        batch_size=None,
        batch_sampler=None,
        shuffle=True,
        drop_last=False,
        num_workers=config["trainer"]["num_workers"],
        pin_memory=config["trainer"]["pin_memory"],
        persistent_workers=False,
    )

    # validation data
    ds_val = data_loader.CustomData(
        config,
        tags_val.years,
        tags_val.lats,
        tags_val.lons,
        tags_val.files,
        tags_val.dict,
        config["trainer"]["val_batch_size"],
    )
    val_loader = torch.utils.data.DataLoader(
        ds_val,
        batch_size=None,
        batch_sampler=None,
        shuffle=True,
        drop_last=False,
        num_workers=config["trainer"]["num_workers"],
        pin_memory=config["trainer"]["pin_memory"],
        persistent_workers=False,
    )

    # BUILD THE MODEL
    verbose = True
    model = utils.load_model(config, clean=not config["trainer"]["resume_training"])

    if verbose:

        # print the model summary
        input_shape = (
            config["trainer"]["batch_size"],
            len(config["data"]["channels"]),
            config["scene_width_landsat"],
            config["scene_width_landsat"],
        )

        torchinfo.summary(
            model,
            input_shape,
            verbose=1,
            col_names=("input_size", "output_size", "num_params"),
        )

    # Setup the optimizer, loss function, and metrics
    optimizer = getattr(torch.optim, config["optimizer"]["type"])(
        model.parameters(), **config["optimizer"]["args"]
    )

    loss = getattr(loss_module, config["trainer"]["loss"]["type"])(
        **config["trainer"]["loss"]["args"]
    )

    metric_funcs = [getattr(metrics_module, met) for met in config["metrics"]]

    # BUILD THE TRAINER
    device = utils.prepare_device(config["device"], config["device_id"])
    trainer = Trainer(
        model,
        loss,
        metric_funcs,
        optimizer,
        max_epochs=config["trainer"]["max_epochs"],
        data_loader=train_loader,
        validation_data_loader=val_loader,
        device=device,
        config=config,
    )

    # FIT THE MODEL
    print("Training the model...")
    model.to(device)
    trainer.fit()
    model.eval()

    # Save the Pytorch Model
    utils.save_torch_model(model, config)

    # Plot the training loss curves
    plt.figure(figsize=(20, 4))
    for i, m in enumerate(("loss", *config["metrics"])):
        plt.subplot(1, 4, i + 1)
        plt.plot(trainer.log.history["epoch"], trainer.log.history[m], label=m)
        plt.plot(
            trainer.log.history["epoch"],
            trainer.log.history["val_" + m],
            label="val_" + m,
        )
        plt.axvline(
            x=trainer.early_stopper.best_epoch,
            linestyle="--",
            color="k",
            linewidth=0.75,
        )
        plt.title(m)
        plt.legend()
        plt.ylim(0, 15.0)
    plt.tight_layout()

    plots.savefig(config, "training_loss_curves")
    plt.close()

    print("Completed.")


if __name__ == "__main__":
    main()
