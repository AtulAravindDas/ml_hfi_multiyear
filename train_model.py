"""Train the CNNs.

Functions
---------
scheduler(epoch, lr)
train_model(settings, model)
"""
import os
import time
import numpy as np
import silence_tensorflow.auto
import tensorflow as tf
import random
import methods

__author__ = "Elizabeth A. Barnes and Randal J. Barnes"
__version__ = "05 May 2023"


directory_paths = methods.get_directories()
SAVE_MODEL_DIRECTORY = directory_paths["save_model_dir"]


def scheduler(epoch, learning_rate):
    if epoch < 10.0:
        return learning_rate
    else:
        return learning_rate * tf.math.exp(-0.1)


def train_model(settings, model, tfds_train, tfds_val, denseweight_dist):
    # SET RANDOM SEEDS AGAIN FOR MODEL TRAINING
    np.random.seed(settings["rng_seed"])
    random.seed(settings["rng_seed"])
    tf.random.set_seed(settings["rng_seed"])

    # DEFINE ALL CALLBACKS

    # early stopping
    earlystopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=settings["patience"],
        verbose=1,
        mode="auto",
        restore_best_weights=True,
    )

    # learning rate scheduler
    learning_rate_callback = tf.keras.callbacks.LearningRateScheduler(
        scheduler, verbose=0
    )

    # checkpoint callback
    checkpoint_dir = SAVE_MODEL_DIRECTORY + settings["exp_name"] + "/"
    if not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    if settings["save_best_only"]:
        checkpoint_path = checkpoint_dir + "model_" + settings["exp_name"] + ".ckpt"
    else:
        checkpoint_path = (
            checkpoint_dir + "model_" + settings["exp_name"] + ".{epoch:04d}.ckpt"
        )

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor="val_loss",
        mode="min",
        save_freq="epoch",
        save_weights_only=True,
        save_best_only=settings["save_best_only"],
    )

    # put the callbacks together
    if settings["early_stopping"]:
        callbacks = (
            [earlystopping_callback, learning_rate_callback, checkpoint_callback],
        )
    else:
        callbacks = ([learning_rate_callback, checkpoint_callback],)

    # optimizer
    optimizer = tf.keras.optimizers.Adam(settings["learning_rate"])

    # loss function
    if "loss" in settings.keys():
        if settings["loss"] == "DenseWeightMSE":
            loss = methods.DenseWeightMSE_Loss(tf.constant(denseweight_dist, dtype="float32"))
        elif settings["loss"] == "DenseDualWeightMSE":
            loss = methods.DenseDualWeightMSE_Loss(tf.constant(denseweight_dist, dtype="float32"),
                                                   tf.constant(settings["loss_params"], dtype="float32"))
        else:
            raise NotImplementedError("no such loss defined.")
    else:
        loss = tf.keras.losses.MeanSquaredError()
    print(f"using {loss = }")

    # SET PRE-FETCH ON DATA
    tfds_train = tfds_train.prefetch(tf.data.AUTOTUNE)
    tfds_val = tfds_val.prefetch(tf.data.AUTOTUNE)

    # COMPILE THE MODEL
    model.compile(
        loss=loss,
        optimizer=optimizer,
        metrics=[
            "mae",
        ],
    )

    # PICK-UP TRAINING WHERE LEFT OFF,IF DESIRED
    if settings["pickup_where_leftoff"] and os.path.isfile(
        checkpoint_dir + "checkpoint"
    ):
        print("loading pre-saved weights")
        latest_model = tf.train.latest_checkpoint(checkpoint_dir)
        model.load_weights(latest_model)

    # TRAIN THE MODEL
    start_time = time.time()
    history = model.fit(
        tfds_train,
        validation_data=tfds_val,
        steps_per_epoch=settings["batches_per_epoch"],
        epochs=settings["max_epochs"],
        callbacks=callbacks,
        batch_size=settings["batch_size"],
        verbose=1,
    )
    stop_time = time.time()

    # DISPLAY THE RESULTS
    best_epoch = np.argmin(history.history["val_loss"])
    fit_summary = {
        "elapsed_time": stop_time - start_time,
        "best_epoch": best_epoch,
        "loss_train": history.history["loss"][best_epoch],
        "loss_valid": history.history["val_loss"][best_epoch],
        "mae_train": history.history["mae"][best_epoch],
        "mae_valid": history.history["val_mae"][best_epoch],
    }

    return model, fit_summary, history, settings
