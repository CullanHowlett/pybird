import os
import sys
import numpy as np
from scipy import interpolate
from configobj import ConfigObj
from chainconsumer import ChainConsumer

sys.path.append("../")
from tbird.Grid import run_camb, run_class
from fitting_codes.fitting_utils import format_pardict, BirdModel, read_chain_backend

if __name__ == "__main__":

    # Reads in a chain containing cosmological parameters output from a fitting routine, and
    # converts the points to alpha_perp, alpha_par, f(z)*sigma8(z) and f(z)*sigma12(z)

    # First, read in the config file used for the fit
    configfile = sys.argv[1]
    pardict = ConfigObj(configfile)

    # Just converts strings in pardicts to numbers in int/float etc.
    pardict = format_pardict(pardict)

    # Set up the BirdModel
    birdmodel = BirdModel(pardict)

    # Compute the values at the central point
    if pardict["code"] == "CAMB":
        _, _, Om_fid, Da_fid, Hz_fid, fN_fid, sigma8_fid, sigma12_fid, r_d_fid = run_camb(pardict)
    else:
        _, _, Om_fid, Da_fid, Hz_fid, fN_fid, sigma8_fid, sigma12_fid, r_d_fid = run_class(pardict)

    marg_str = "marg" if pardict["do_marg"] else "all"
    hex_str = "hex" if pardict["do_hex"] else "nohex"
    dat_str = "xi" if pardict["do_corr"] else "pk"
    fmt_str = "%s_%s_%2d_%3d_%s_%s_%s" if pardict["do_corr"] else "%s_%s_%3.2lf_%3.2lf_%s_%s_%s"

    taylor_strs = ["grid", "1order", "2order", "3order", "4order"]
    chainfile = str(
        fmt_str
        % (
            birdmodel.pardict["fitfile"],
            dat_str,
            birdmodel.pardict["xfit_min"],
            birdmodel.pardict["xfit_max"],
            taylor_strs[pardict["taylor_order"]],
            hex_str,
            marg_str,
        )
    )
    oldfile = chainfile + ".hdf5"
    newfile = chainfile + "_converted.dat"
    burntin, bestfit, like = read_chain_backend(oldfile)

    lower_bounds = birdmodel.valueref - birdmodel.pardict["order"] * birdmodel.delta
    upper_bounds = birdmodel.valueref + birdmodel.pardict["order"] * birdmodel.delta

    # Loop over the parameters in the chain and use the grids to compute the derived parameters
    chainvals = []
    for i, (vals, loglike) in enumerate(zip(burntin, like)):
        if i % 1000 == 0:
            print(i)
        ln10As, h, omega_cdm, b1 = vals[:4]
        omega_b = birdmodel.valueref[3]
        if np.any(np.less([ln10As, h, omega_cdm, omega_b], lower_bounds)) or np.any(
            np.greater([ln10As, h, omega_cdm, omega_b], upper_bounds)
        ):
            continue
        Om, Da, Hz, f, sigma8, sigma12, r_d = birdmodel.compute_params([ln10As, h, omega_cdm, omega_b])
        alpha_perp = (Da / h) * (float(pardict["h"]) / Da_fid) * (r_d_fid * float(pardict["h"]) / (r_d * h))
        alpha_par = (float(pardict["h"]) * Hz_fid) / (h * Hz) * (r_d_fid * float(pardict["h"]) / (r_d * h))
        chainvals.append(
            (
                alpha_perp,
                alpha_par,
                Om,
                2997.92458 * Da / h,
                100.0 * h * Hz,
                f * sigma8,
                f * sigma12,
                b1 * sigma8,
                b1 * sigma12,
                loglike,
            )
        )

    np.savetxt(newfile, np.array(chainvals))
