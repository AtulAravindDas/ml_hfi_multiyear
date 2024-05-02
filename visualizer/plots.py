"""Functions for making nice plots

Functions
---------
savefig(fig, filename, dpi=300)
plot_hfi_tile(xplot, extent)
"""

import numpy as np
import matplotlib.pyplot as plt  # type: ignore
import matplotlib as mpl  # type: ignore
from utils import utils


mpl.rcParams["figure.facecolor"] = "white"
mpl.rcParams["figure.dpi"] = 150

FS = 10
plt.rc("text", usetex=False)
# plt.rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"]})
plt.rc("savefig", facecolor="white")
plt.rc("axes", facecolor="white")
plt.rc("axes", labelcolor="dimgrey")
plt.rc("axes", labelcolor="dimgrey")
plt.rc("xtick", color="dimgrey")
plt.rc("ytick", color="dimgrey")


def savefig(config, filename, format=".png", dpi=300):

    model_name = utils.get_model_name(config["expname"], config["seed"])
    directory_paths = utils.get_directories(config["machine"])
    pathname = directory_paths["figures_dir"] + model_name + "_" + filename

    plt.savefig(pathname + format, bbox_inches="tight", dpi=dpi)


def plot_hfi_tile(xplot, extent):
    # extent = lon_w, lon_e, lat_s, lat_n,

    cmap = plt.get_cmap("PiYG_r")
    cmap.set_bad(color="lightgray", alpha=1.0)

    xplot = np.asarray(xplot, dtype="float32")
    xplot[xplot == 255] = 0.0
    p = plt.imshow(xplot, cmap=cmap, extent=extent)

    return p


def adjust_spines(ax, spines):
    for loc, spine in ax.spines.items():
        if loc in spines:
            spine.set_position(("outward", 5))
        else:
            spine.set_color("none")
    if "left" in spines:
        ax.yaxis.set_ticks_position("left")
    else:
        ax.yaxis.set_ticks([])
    if "bottom" in spines:
        ax.xaxis.set_ticks_position("bottom")
    else:
        ax.xaxis.set_ticks([])


def format_spines(ax):
    adjust_spines(ax, ["left", "bottom"])
    ax.spines["top"].set_color("none")
    ax.spines["right"].set_color("none")
    ax.spines["left"].set_color("dimgrey")
    ax.spines["bottom"].set_color("dimgrey")
    ax.spines["left"].set_linewidth(2)
    ax.spines["bottom"].set_linewidth(2)
    ax.tick_params("both", length=4, width=2, which="major", color="dimgrey")
