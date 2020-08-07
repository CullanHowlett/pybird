import sys
import numpy as np
import pandas as pd
from configobj import ConfigObj
import matplotlib.pyplot as plt
from scipy.interpolate import splev, splrep

sys.path.append("../")
from pybird import pybird
from tbird.Grid import run_camb, run_class
from fitting_codes.fitting_utils import format_pardict, FittingData

if __name__ == "__main__":

    # Set up the data
    configfile = sys.argv[1]
    pardict = ConfigObj(configfile)
    pardict = format_pardict(pardict)
    fittingdata = FittingData(pardict)
    kin, Plin, Om, Da, Hz, fN, sigma8, sigma12, r_d = run_class(pardict)

    # Read in the UNIT Pk and compare
    datapk = np.array(
        pd.read_csv(
            "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_Stage2/input_data/Pk_Planck15_Table4.txt",
            delim_whitespace=True,
            header=None,
        )
    )
    datapk[:, 1] *= sigma8 ** 2 / 0.8147 ** 2

    # Plot power spectrum ratio. The two methods agree almost perfectly.
    plt.errorbar(datapk[:, 0], datapk[:, 1] / splev(datapk[:, 0], splrep(kin, Plin)) - 1.0, color="r", linestyle="-")
    plt.xlabel(r"$k\,(h\,\mathrm{Mpc}^{-1})$", fontsize=22)
    plt.ylabel(r"$P_{\mathrm{UNIT}}(k)/P_{\mathrm{Cullan}}(k)-1$", fontsize=22, labelpad=5)
    # plt.ylim(-1.0e-5, 1.0e-5)
    plt.show()
