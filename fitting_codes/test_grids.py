import numpy as np
import sys
import copy
from configobj import ConfigObj
import matplotlib.pyplot as plt

sys.path.append("../")
from fitting_codes.fitting_utils import (
    FittingData,
    BirdModel,
    create_plot,
    update_plot,
    format_pardict,
    do_optimization,
    get_Planck,
)


def plot_comparison(k, pk_base, pk_grid, pk_ndimage, pk_taylor):

    plt.errorbar(
        k,
        k * pk_base[1],
        marker="None",
        color="k",
        linestyle="-",
        markeredgewidth=1.3,
        zorder=0,
    )
    plt.errorbar(
        k,
        k * pk_grid[1],
        marker="None",
        color="r",
        linestyle="-",
        markeredgewidth=1.3,
        zorder=0,
    )
    plt.errorbar(
        k,
        k * pk_taylor[1],
        marker="None",
        color="b",
        linestyle="-",
        markeredgewidth=1.3,
        zorder=0,
    )
    plt.errorbar(
        k,
        k * pk_ndimage[1],
        marker="None",
        color="g",
        linestyle="-",
        markeredgewidth=1.3,
        zorder=0,
    )
    plt.errorbar(
        k,
        k * pk_base[0],
        marker="None",
        color="k",
        linestyle="--",
        markeredgewidth=1.3,
        zorder=0,
    )
    plt.errorbar(
        k,
        k * pk_grid[0],
        marker="None",
        color="r",
        linestyle="--",
        markeredgewidth=1.3,
        zorder=0,
    )
    plt.errorbar(
        k,
        k * pk_taylor[0],
        marker="None",
        color="b",
        linestyle="--",
        markeredgewidth=1.3,
        zorder=0,
    )
    plt.errorbar(
        k,
        k * pk_ndimage[0],
        marker="None",
        color="g",
        linestyle="--",
        markeredgewidth=1.3,
        zorder=0,
    )

    plt.xlim(np.amin(pardict["xfit_min"]) * 0.95, np.amax(pardict["xfit_max"]) * 1.05)
    plt.xlabel(r"$k\,(h\,\mathrm{Mpc}^{-1})$", fontsize=16)
    plt.ylabel(r"$kP(k)\,(h^{-2}\,\mathrm{Mpc}^{2})$", fontsize=16, labelpad=5)
    plt.tick_params(width=1.3)
    plt.tick_params("both", length=10, which="major")
    plt.tick_params("both", length=5, which="minor")
    for axis in ["top", "left", "bottom", "right"]:
        plt.gca().spines[axis].set_linewidth(1.3)
    for tick in plt.gca().xaxis.get_ticklabels():
        tick.set_fontsize(14)
    for tick in plt.gca().yaxis.get_ticklabels():
        tick.set_fontsize(14)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    # Code to generate a power spectrum template at fixed cosmology using pybird, then fit the AP parameters and fsigma8
    # First read in the config file
    configfile = sys.argv[1]
    pardict = ConfigObj(configfile)

    # Just converts strings in pardicts to numbers in int/float etc.
    pardict = format_pardict(pardict)
    pardict2 = copy.copy(pardict)
    pardict2["taylor_order"] = 4
    pardict3 = copy.copy(pardict)
    pardict3["taylor_order"] = -1

    # Set up the data
    fittingdata = FittingData(pardict)

    # Read in the window functions
    winnames = np.loadtxt(pardict["winfile"], dtype=str)

    # Set up the BirdModels
    birdmodels = []
    birdmodels_grid = []
    birdmodels_ndimage = []
    birdmodels_taylor = []
    for i in range(1):
        birdmodels.append(BirdModel(pardict, direct=True, redindex=i, window=winnames[i]))
        birdmodels_grid.append(BirdModel(pardict, redindex=i))
        birdmodels_ndimage.append(BirdModel(pardict3, redindex=i))
        birdmodels_taylor.append(BirdModel(pardict2, redindex=i))

    index = np.where(birdmodels[0].kin <= 0.25)[0]

    params = birdmodels[0].valueref - np.array([3.5 * birdmodels[0].delta[0], 0.0, 0.0, 0.0])
    params = params[:, None]

    # Do some comparison plots
    Plin, Ploop = birdmodels[0].compute_pk(params)
    Plin_taylor, Ploop_taylor = birdmodels_taylor[0].compute_pk(params)
    Plin_grid, Ploop_grid = birdmodels_grid[0].compute_pk(params)
    Plin_ndimage, Ploop_ndimage = birdmodels_ndimage[0].compute_pk(params)
    print(np.shape(birdmodels[0].kin[index]), np.shape(Plin[:, 0, index, 0]))
    print(np.shape(Plin_grid))
    for i in range(3):
        plot_comparison(
            birdmodels[0].kin[index],
            Plin[:, i, index, 0],
            Plin_grid[:2, i, index, 0],
            Plin_ndimage[:2, i, index, 0],
            Plin_taylor[:2, i, index, 0],
        )
    for i in range(19):
        plot_comparison(
            birdmodels[0].kin[index],
            Ploop[:, index, i, 0],
            Ploop_grid[:2, index, i, 0],
            Ploop_ndimage[:2, index, i, 0],
            Ploop_taylor[:2, index, i, 0],
        )
