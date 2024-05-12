"""
Module for analyzing the predictions of a model.

Classes
---------
None


Functions
---------

process_flatten(config, labels, predictions, remove_edges=False):
    Process the predictions and labels based on the given configuration.

"""

import numpy as np
from data_builder import data_methods


def process_flatten(config, labels, predictions, remove_edges=False):
    """
    Process the predictions and labels based on the given configuration.

    Args:
        config (dict): The configuration settings.
        labels (ndarray): The array of labels.
        predictions (ndarray): The array of predictions.
        remove_edges (bool, optional): Whether to remove edges. Defaults to False.

    Returns:
        ndarray: The processed labels.
        ndarray: The processed predictions.
    """

    assert labels.shape == predictions.shape

    # REMOVE EDGES FOR NOW
    if remove_edges:
        if config["inference"]["quicklook"]:
            predictions = predictions[
                config["inference"]["quicklook_skiplen"]
                * 2 : -config["inference"]["quicklook_skiplen"]
                * 2,
                config["inference"]["quicklook_skiplen"]
                * 2 : -config["inference"]["quicklook_skiplen"]
                * 2,
            ]

            labels = labels[
                config["inference"]["quicklook_skiplen"]
                * 2 : -config["inference"]["quicklook_skiplen"]
                * 2,
                config["inference"]["quicklook_skiplen"]
                * 2 : -config["inference"]["quicklook_skiplen"]
                * 2,
            ]

        else:
            predictions = predictions[10:-10, 10:-10]
            labels = labels[10:-10, 10:-10]

    # FLATTEN and REMOVE NON-PREDICTED PIXELS
    predictions = predictions.flatten()
    labels = labels.flatten()
    if config["inference"]["quicklook"]:
        predictions = predictions[:: config["inference"]["quicklook_skiplen"]]
        labels = labels[:: config["inference"]["quicklook_skiplen"]]
    predictions, labels = data_methods.remove_nodata(predictions, labels)

    print(labels.shape, predictions.shape)

    return labels, predictions
