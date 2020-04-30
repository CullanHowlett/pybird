import numpy as np
import sys
from configobj import ConfigObj
import matplotlib.pyplot as plt

sys.path.append("../")
from fitting_codes.fitting_utils import BirdModel, format_pardict

if __name__ == "__main__":

    # Code to generate a power spectrum template at fixed cosmology using pybird, then fit the AP parameters and fsigma8
    # First read in the config file
    configfile = sys.argv[1]
    pardict = ConfigObj(configfile)

    # Just converts strings in pardicts to numbers in int/float etc.
    pardict = format_pardict(pardict)

    # Set up the BirdModel
    birdmodel = BirdModel(pardict, template=False)

    if pardict["do_corr"]:
        shotnoise = None
        bs = [1.3, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    else:
        shotnoise = 400.0
        bs = [1.3, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    Plin, Ploop, Pct, Pst = birdmodel.get_components(birdmodel.valueref, bs, shotnoise=shotnoise)

    if pardict["do_corr"]:
        plt.errorbar(birdmodel.kin, birdmodel.kin ** 2 * Plin[0], color="r", linestyle="-")
        plt.errorbar(birdmodel.kin, birdmodel.kin ** 2 * Plin[1], color="b", linestyle="-")
        plt.errorbar(birdmodel.kin, birdmodel.kin ** 2 * Ploop[0], color="r", linestyle="--")
        plt.errorbar(birdmodel.kin, birdmodel.kin ** 2 * Ploop[1], color="b", linestyle="--")
        plt.errorbar(birdmodel.kin, birdmodel.kin ** 2 * Pct[0], color="r", linestyle=":")
        plt.errorbar(birdmodel.kin, birdmodel.kin ** 2 * Pct[1], color="b", linestyle=":")
        plt.xlim(10.0, pardict["xfit_max"] * 1.05)
        plt.xlabel(r"$s\,(h^{-1}\,\mathrm{Mpc})$", fontsize=22)
        plt.ylabel(r"$s^{2}\xi(s)$", fontsize=22, labelpad=5)
    else:
        plt.errorbar(birdmodel.kin, birdmodel.kin * Plin[0], color="r", linestyle="-")
        plt.errorbar(birdmodel.kin, birdmodel.kin * Plin[1], color="b", linestyle="-")
        plt.errorbar(birdmodel.kin, birdmodel.kin * Ploop[0], color="r", linestyle="--")
        plt.errorbar(birdmodel.kin, birdmodel.kin * Ploop[1], color="b", linestyle="--")
        plt.errorbar(birdmodel.kin, birdmodel.kin * Pct[0], color="r", linestyle=":")
        plt.errorbar(birdmodel.kin, birdmodel.kin * Pct[1], color="b", linestyle=":")
        plt.errorbar(birdmodel.kin, birdmodel.kin * Pst[0], color="r", linestyle="-.")
        plt.errorbar(birdmodel.kin, birdmodel.kin * Pst[1], color="b", linestyle="-.")
        plt.xlim(0.0, pardict["xfit_max"] * 1.05)
        plt.xlabel(r"$k\,(h\,\mathrm{Mpc}^{-1})$", fontsize=22)
        plt.ylabel(r"$kP(k)\,(h^{-3}\,\mathrm{Mpc}^3)$", fontsize=22, labelpad=5)
    plt.tick_params(width=1.3)
    plt.tick_params("both", length=10, which="major")
    plt.tick_params("both", length=5, which="minor")
    for axis in ["top", "left", "bottom", "right"]:
        plt.gca().spines[axis].set_linewidth(1.3)
    for tick in plt.gca().xaxis.get_ticklabels():
        tick.set_fontsize(14)
    for tick in plt.gca().yaxis.get_ticklabels():
        tick.set_fontsize(14)
    plt.show()
