import numpy as np
import sys
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


def plot_comparison(k, pk, pk_grid):

    plt.errorbar(
        k,
        pk_grid[1] / pk[1] - 1.0,
        marker="None",
        color="k",
        linestyle="-",
        markeredgewidth=1.3,
        zorder=0,
    )
    plt.errorbar(
        k,
        pk_grid[0] / pk[0] - 1.0,
        marker="None",
        color="k",
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

    # Set up the data
    fittingdata = FittingData(pardict)

    # Read in the window functions
    winnames = np.loadtxt(pardict["winfile"], dtype=str)

    # Set up the BirdModels
    birdmodels = []
    birdmodels_grid = []
    for i in range(len(pardict["z_pk"])):
        birdmodels.append(BirdModel(pardict, direct=True, redindex=i, window=winnames[i]))
        birdmodels_grid.append(BirdModel(pardict, redindex=i))

    params = np.array(
        [float(pardict["ln10^{10}A_s"]), float(pardict["h"]), float(pardict["omega_cdm"]), float(pardict["omega_b"])]
    )[:, None]

    # Do some comparison plots
    Plin, Ploop = birdmodels[0].compute_pk(params)
    Plin_grid, Ploop_grid = birdmodels_grid[0].compute_pk(params)
    print(birdmodels[0].kin, birdmodels_grid[0].kin)
    for i in range(3):
        plot_comparison(birdmodels[0].kin, Plin_grid[:2, i, :, 0] / Plin[:, i, :, 0] - 1.0)
