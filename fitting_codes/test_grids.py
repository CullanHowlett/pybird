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
    plot_flag = sys.argv[2]
    pardict = ConfigObj(configfile)

    # Just converts strings in pardicts to numbers in int/float etc.
    pardict = format_pardict(pardict)
    pardict2 = copy.copy(pardict)
    pardict2["code"] = "CAMB"
    pardict3 = copy.copy(pardict)
    pardict3["taylor_order"] = -1

    # Set up the data
    fittingdata = FittingData(pardict)

    # Read in the window functions
    winnames = np.loadtxt(pardict["winfile"], dtype=str)

    # Set up the BirdModels
    birdmodels = []
    birdmodels_grid = []
    # birdmodels_ndimage = []
    # birdmodels_taylor = []
    for i in range(1):
        birdmodels.append(BirdModel(pardict, direct=True, redindex=i))
        birdmodels_grid.append(BirdModel(pardict, direct=True, redindex=i))
        # birdmodels_grid.append(BirdModel(pardict, redindex=i))
        # birdmodels_ndimage.append(BirdModel(pardict3, redindex=i))
        # birdmodels_taylor.append(BirdModel(pardict2, redindex=i))

    index = np.where(birdmodels[0].kin <= 0.25)[0]
    print(index)

    params = np.array(
        [
            [
                birdmodels[0].valueref[0],
                birdmodels[0].valueref[1],
                birdmodels[0].valueref[2],
                birdmodels[0].valueref[3],
                0.0,
            ],
            [2.65133479, 0.65543637, 0.11155482, 0.0225015, -0.07430659],
        ]
    ).T
    print(params)

    # Fixed sim cosmology
    # [1.82329744, 0.7277994,  0.36883151, 1.82566472, 0.66005296, 0.56958937, 2.01504828, 0.73229193, 0.26549316, 1.97126975, 0.48619364, 0.31502055]
    # 13.975620481579705

    # [2.71847577,  0.69834081,  0.11520184,  0.02231963, -0.23094655,  2.04720901, 0.72866703,  1.19230816,  2.03879757,  0.49749398, - 0.28684672,  2.28461804, 0.72606937,  0.89107149,  2.22612114,  0.30310565,  0.11983974]
    # 14.944924472813728

    # [1.82288593 0.71230644 1.8249855  0.6402078  2.01430063 0.71742001 1.96952603 0.46803072]
    # [63.91720995]
    # 13.895122791901443

    # [1.89332541 0.79972024 1.89362445 0.69606235 2.09136723 0.87192942 2.04276339 0.54878832]
    # [65.35896434]
    # 15.97329758543616

    # [2.78040454  0.69623241  0.1138217   0.02226641 -0.22522994  1.94444006 0.51014521  1.959835    0.50018533  2.15895801  0.38411174  2.12181232 0.20969426]
    # [61.25668875]
    # 14.228549185885775

    # Plotting (for checking/debugging, should turn off for production runs)
    plt = create_plot(pardict, fittingdata, plotindex=plot_flag - 1)

    # Do some comparison plots
    Plin, Ploop = birdmodels[0].get_components(params)
    Plin_grid, Ploop_grid = birdmodels_grid[0].get_components(params)
    """Plin_taylor, Ploop_taylor = birdmodels_taylor[0].compute_pk(params)
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
        )"""
    for i in range(np.shape(params)[1]):
        # plt.plot(birdmodels[0].kin[index], birdmodels[0].kin[index] * Plin[0, 0, index, i])
        plt.plot(birdmodels[0].kin[index], birdmodels[0].kin[index] * Plin_grid[0, 0, index, i])
    plt.ylim(-200.0, 500.0)
    plt.show()
