import sys
import numpy as np
import pandas as pd
from configobj import ConfigObj
import matplotlib.pyplot as plt
from scipy.interpolate import splev, splrep

sys.path.append("../")
from pybird import pybird
from tbird.Grid import run_camb
from fitting_codes.fitting_utils import BirdModel, format_pardict, FittingData

if __name__ == "__main__":

    from classy import Class

    # Read in Guido's test files
    dataC11 = np.array(
        pd.read_csv(
            "/Volumes/Work/UQ/DESI/cBIRD/UNIT_output_files/noresumCf.dat",
            header=None,
            skiprows=0,
            delim_whitespace=True,
        )
    )
    datas = dataC11[:, 0].reshape((2, -1))
    dataxi11 = dataC11[:, 1].reshape((2, -1))

    # Code to generate a power spectrum template at fixed cosmology using pybird, then fit the AP parameters and fsigma8
    # First read in the config file
    configfile = sys.argv[1]
    pardict = ConfigObj(configfile)

    # Just converts strings in pardicts to numbers in int/float etc.
    pardict = format_pardict(pardict)
    pardict["omega_cdm"] = 0.1188130

    # Set up the BirdModel
    # kin, Pin, Om, Da, Hz, fN, sigma8, sigma12, r_d = run_camb(pardict)
    kin = np.logspace(-5, 0, 200)
    zpk = 0.9873
    M = Class()
    M.set(
        {
            "ln10^{10}A_s": 3.064325065,
            "n_s": 0.9667,
            "h": 0.6774,
            "omega_b": 0.02230,
            "omega_cdm": 0.118813,
            "N_ur": 0.00641,
            "N_ncdm": 3,
            "m_ncdm": "0.02, 0.02, 0.02",
        }
    )

    M.set({"output": "mPk", "P_k_max_1/Mpc": 1.0, "z_max_pk": zpk})
    M.compute()

    # P(k) in (Mpc/h)**3
    # Pin = Pin[kin <= 1.0]
    # kin = kin[kin <= 1.0]
    Pk = [M.pk(ki * M.h(), zpk) * M.h() ** 3 for ki in kin]

    sdata = np.linspace(25, 200, 50)
    Om_AP = 0.30
    z_AP = float(pardict["z_pk"])

    common = pybird.Common(Nl=2, kmax=0.3, smax=1000, optiresum=True)
    nonlinear = pybird.NonLinear(load=False, save=False, co=common)
    resum = pybird.Resum(co=common)
    projection = pybird.Projection(sdata, pybird.DA(Om_AP, z_AP), pybird.Hubble(Om_AP, z_AP), co=common, cf=True)
    bs = [1.3, 0.8, 0.2, 0.8, 0.2, 0, 0]

    crow = pybird.Bird(
        kin,
        Pk,
        pybird.fN(0.3089, zpk),
        DA=pybird.DA(0.3089, zpk),
        H=pybird.Hubble(0.3089, zpk),
        z=pardict["z_pk"],
        which="full",
        co=common,
    )
    nonlinear.PsCf(crow)
    crow.setPsCf(bs)
    # resum.PsCf(crow)
    # projection.AP(crow)
    # projection.kdata(crow)
    # crow.setreduceCflb(bs)

    # Set up the data
    fittingdata = FittingData(pardict)

    plt.errorbar(datas[0], datas[0] ** 2 * dataxi11[0], color="r", linestyle="-")
    plt.errorbar(datas[1], datas[1] ** 2 * dataxi11[1], color="b", linestyle="-")
    plt.errorbar(common.s, common.s ** 2 * crow.fullCf[0], color="r", linestyle="--")
    plt.errorbar(common.s, common.s ** 2 * crow.fullCf[1], color="b", linestyle="--")
    plt.xlim(10.0, pardict["xfit_max"] * 1.05)
    plt.xlabel(r"$s\,(h^{-1}\,\mathrm{Mpc})$", fontsize=22)
    plt.ylabel(r"$s^{2}\xi(s)$", fontsize=22, labelpad=5)
    plt.show()

    x_data = fittingdata.data["x_data"]
    fit_data = fittingdata.data["fit_data"]
    cov = fittingdata.data["cov"]
    nx = len(x_data)
    plt_err = np.sqrt(cov[np.diag_indices(2 * nx)])

    plt.errorbar(
        x_data,
        splev(x_data, splrep(datas[0], crow.fullCf[0] - dataxi11[0])) / plt_err[:nx],
        color="r",
        linestyle="-",
        label="Monopole",
    )
    plt.errorbar(
        x_data,
        splev(x_data, splrep(datas[1], crow.fullCf[1] - dataxi11[1])) / plt_err[nx : 2 * nx],
        color="b",
        linestyle="-",
        label="Quadrupole",
    )
    plt.xlim(10.0, pardict["xfit_max"] * 1.05)
    plt.ylim(-0.2, 0.2)
    plt.xlabel(r"$s\,(h^{-1}\,\mathrm{Mpc})$", fontsize=14)
    plt.ylabel(r"$(\xi_{\mathrm{Cullan}} - \xi_{\mathrm{Guido}})/\sigma_{\xi}\,(s)$", fontsize=14, labelpad=5)
    plt.legend()
    plt.tight_layout()
    plt.show()
