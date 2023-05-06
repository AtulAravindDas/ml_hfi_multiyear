"""Train the CNNs.

Functions
---------
scheduler(epoch, lr)
train_model(settings, model)
"""

import time
import numpy as np
import silence_tensorflow.auto
import tensorflow as tf
import build_model

__author__ = "Elizabeth A. Barnes and Randal J. Barnes"
__version__ = "05 May 2023"


BIG_DATA_DIRECTORY = "/Users/eabarnes/big_data/"


def scheduler(epoch, learning_rate):
    if epoch < 100.:
        return learning_rate
    else:
        return learning_rate * tf.math.exp(-0.1)


def train_model(settings, model, tfds_train, tfds_val):

    # early stopping
    earlystopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=settings["patience"], verbose=1, mode='auto', restore_best_weights=True)

    # learning rate scheduler
    learning_rate_callback = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=0)

    # checkpoint callback
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=BIG_DATA_DIRECTORY + 'saved_model_checkpoints/model_' + settings["exp_name"] + '.{epoch:02d}.h5',
        save_weights_only=False,
        monitor='val_mean_squared_error', mode='min',
        save_freq='epoch', save_best_only=False, )

    # optimizer
    optimizer = tf.keras.optimizers.Adam(settings["learning_rate"])

    # loss function
    # WE WILL WANT TO MODIFY THIS TO INCLUDE THE KLUGE TO ZERO ******
    loss = tf.keras.losses.MeanSquaredError()

    # compile the model
    model.compile(
        loss=loss,
        optimizer=optimizer,
        metrics=["mae", ]
    )

    # train the model
    start_time = time.time()
    history = model.fit(
        tfds_train,
        validation_data=tfds_val,
        epochs=settings["max_epochs"],
        callbacks=[earlystopping_callback, learning_rate_callback, checkpoint_callback],
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
    }

    return model, fit_summary, history, settings
