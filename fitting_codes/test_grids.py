import numpy as np
import sys
import copy
from configobj import ConfigObj
import matplotlib.pyplot as plt
from pybird_dev.pybird import Correlator

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
    plot_flag = int(sys.argv[2])
    pardict = ConfigObj(configfile)

    # Just converts strings in pardicts to numbers in int/float etc.
    pardict = format_pardict(pardict)
    pardict2 = copy.copy(pardict)
    pardict2["code"] = "CAMB"
    pardict3 = copy.copy(pardict)
    pardict3["taylor_order"] = 0

    # Set up the data
    fittingdata = FittingData(pardict)

    # Read in the window functions
    winnames = np.loadtxt(pardict["winfile"], dtype=str)

    # Set up the BirdModel
    birdmodel = BirdModel(pardict, direct=True, redindex=0)

    params = np.array(
        [
            [
                birdmodel.valueref[0],
                birdmodel.valueref[1],
                birdmodel.valueref[2],
                birdmodel.valueref[3],
                birdmodel.valueref[4],
            ]
        ]
    ).T
    bs = np.array(
        [
            [
                1.8,
                0.1,
                0.2,
                0.3,
                0.4,
                0.5,
                0.6,
                0.7,
                0.8,
                0.9,
            ]
        ]
    ).T

    Plin, Ploop = birdmodel.compute_pk(params)
    P_model, P_model_interp = birdmodel.compute_model(bs, Plin, Ploop, fittingdata.data["x_data"][0])

    # Set up a default pybird correlator
    Nl = 3 if pardict["do_hex"] else 2
    optiresum = True if pardict["do_corr"] else False
    output = "bCf" if pardict["do_corr"] else "bPk"
    kmax = None if pardict["do_corr"] else 0.6
    correlator = Correlator()

    # Set up pybird
    correlator.set(
        {
            "output": output,
            "multipole": Nl,
            "z": float(pardict["z_pk"][0]),
            "optiresum": False,
            "with_AP": True,
            "kmax": 0.5,
            "with_bias": True,
            "DA_AP": birdmodel.Da,
            "H_AP": birdmodel.Hz,
        }
    )

    correlator.compute(
        {
            "k11": birdmodel.kmod,
            "P11": birdmodel.Pmod,
            "z": float(pardict["z_pk"][0]),
            "Omega0_m": birdmodel.Om,
            "f": birdmodel.fN,
            "DA": birdmodel.Da,
            "H": birdmodel.Hz,
            "bias": {
                "b1": bs[0, 0],
                "b2": bs[1, 0],
                "b3": bs[2, 0],
                "b4": bs[3, 0],
                "cct": bs[4, 0],
                "cr1": bs[5, 0],
                "cr2": bs[6, 0],
            },
        }
    )

    # Plotting (for checking/debugging, should turn off for production runs)
    plt = create_plot(pardict, fittingdata, plotindex=plot_flag - 1)

    # Do some comparison plot
    plt.plot(birdmodel.kin, birdmodel.kin * P_model[2])
    plt.plot(correlator.co.k, correlator.co.k * correlator.bird.fullPs[2], ls="--")
    plt.ioff()
    plt.show()
