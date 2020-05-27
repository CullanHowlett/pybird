import sys
import numpy as np
import pandas as pd
from configobj import ConfigObj
import matplotlib.pyplot as plt

sys.path.append("../")
from pybird import pybird
from tbird.Grid import run_camb
from fitting_codes.fitting_utils import BirdModel, format_pardict

if __name__ == "__main__":

    # Read in Guido's test files
    dataC11 = np.array(
        pd.read_csv(
            "/Volumes/Work/UQ/DESI/cBIRD/UNIT_output_files/resumCf.dat", header=None, skiprows=0, delim_whitespace=True,
        )
    )
    datas = dataC11[:, 0].reshape((2, -1))
    dataxi11 = dataC11[:, 1].reshape((2, -1))

    # Code to generate a power spectrum template at fixed cosmology using pybird, then fit the AP parameters and fsigma8
    # First read in the config file
    configfile = sys.argv[1]
    pardict = ConfigObj(configfile)
    pardict = format_pardict(pardict)

    # Set up the BirdModel
    sdata = np.linspace(25, 200, 50)
    kin, Pin, Da, Hz, fN, sigma8, sigma12, r_d = run_camb(pardict)
    birdmodel = BirdModel(pardict, template=False)
    Plin, Ploop = birdmodel.compute_pk(birdmodel.valueref)
    bs = [1.3, 0.8, 0.2, 0.8, 0.2, 0, 0]
    P_model = birdmodel.compute_model(bs, Plin, Ploop, sdata)

    plt.errorbar(datas[0], datas[0] ** 2 * dataxi11[0], color="r", linestyle="-")
    plt.errorbar(datas[1], datas[1] ** 2 * dataxi11[1], color="b", linestyle="-")
    plt.errorbar(sdata, sdata ** 2 * P_model[: len(sdata)], color="r", linestyle="--")
    plt.errorbar(sdata, sdata ** 2 * P_model[len(sdata) :], color="b", linestyle="--")
    plt.xlim(10.0, pardict["xfit_max"] * 1.05)
    plt.xlabel(r"$s\,(h^{-1}\,\mathrm{Mpc})$", fontsize=22)
    plt.ylabel(r"$s^{2}\xi(s)$", fontsize=22, labelpad=5)
    plt.show()
