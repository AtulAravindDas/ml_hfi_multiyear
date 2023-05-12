"""Functions for making nice plots

Functions
---------
plot_hfi_tile(xplot, extent)
"""
import numpy as np
import matplotlib.pyplot as plt

__author__ = "Elizabeth A. Barnes and Randal J. Barnes"
__date__ = "11 May 2023"


def plot_hfi_tile(xplot, extent):

    cmap = plt.get_cmap('PiYG_r')
    cmap.set_bad(color='lightgray', alpha=1.)

    xplot = np.asarray(xplot, dtype="float32")
    xplot[xplot == 255] = 0.0
    p = plt.imshow(xplot, cmap=cmap, extent=extent)

    return p
