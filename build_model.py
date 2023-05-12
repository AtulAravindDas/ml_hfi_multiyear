"""Build the CNN model

Functions
---------
build_model(settings, input_shape)
"""
import numpy as np
import tensorflow as tf

__author__ = "Elizabeth A. Barnes and Randal J. Barnes"
__date__ = "10 May 2023"


def build_model(settings, input_shape):

    ACTIVATION = "relu"
    FINAL_ACTIVATION = "sigmoid"

    input = tf.keras.Input(shape=input_shape)

    # add data augmentation layers
    # x = input
    if settings["aug_randomflip"]:
        x = tf.keras.layers.RandomFlip("horizontal_and_vertical",
                                       seed=settings["rng_seed"])(input)
    else:
        x = input

    # Add all the Dense Layers
    for layer, nodes in enumerate(settings["layers_units"]):

        x = tf.keras.layers.Conv2D(
            nodes, activation=ACTIVATION,
            kernel_size=settings["kernel_size"],
            kernel_initializer=tf.keras.initializers.RandomNormal(seed=settings["rng_seed"] + layer),
            bias_initializer=tf.keras.initializers.RandomNormal(seed=settings["rng_seed"] + layer * 2),
            padding="same",
            data_format='channels_last',
        )(x)

        if layer < len(settings["layers_units"]) - 1:

            x = tf.keras.layers.MaxPooling2D(
                pool_size=(settings["max_pool_stride"][0], settings["max_pool_stride"][0]),
                strides=settings["max_pool_stride"], padding='valid',
                data_format='channels_last')(x)

    # # dropout layer
    # x = tf.keras.layers.Dropout(rate=settings["dropout"])(x)

    # flatten layer
    x = tf.keras.layers.Flatten()(x)

    # # add skip connection
    input_flat = tf.keras.layers.Flatten()(input[:, :, :, 2])
    x = tf.keras.layers.Concatenate(axis=1)([x, input_flat])

    # dropout layer
    x = tf.keras.layers.Dropout(rate=settings["dropout"])(x)

    # Final dense layer before the classification layers.
    x = tf.keras.layers.Dense(units=settings["dense_units"],
                              kernel_initializer=tf.keras.initializers.RandomNormal(seed=settings["rng_seed"] + 100),
                              bias_initializer=tf.keras.initializers.Zeros(),
                              activation=ACTIVATION,)(x)

    # Final layer here
    x = tf.keras.layers.Dense(1,
                              bias_initializer=tf.keras.initializers.Zeros(),
                              activation=FINAL_ACTIVATION)(x)

    model = tf.keras.models.Model(inputs=input, outputs=x)

    # model.summary()

    return model
