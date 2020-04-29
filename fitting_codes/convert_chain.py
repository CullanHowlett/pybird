import os
import sys
import numpy as np
from scipy import interpolate
from configobj import ConfigObj
from chainconsumer import ChainConsumer

sys.path.append("../")
from tbird.Grid import run_camb
from fitting_codes.fitting_utils import format_pardict, BirdModel, read_chain

if __name__ == "__main__":

    # Reads in a chain containing cosmological parameters output from a fitting routine, and
    # converts the points to alpha_perp, alpha_par, f(z)*sigma8(z) and f(z)*sigma12(z)

    # First, read in the config file used for the fit
    configfile = sys.argv[1]
    fixed_h = int(sys.argv[2])
    pardict = ConfigObj(configfile)

    # Just converts strings in pardicts to numbers in int/float etc.
    pardict = format_pardict(pardict)

    # Set up the BirdModel
    birdmodel = BirdModel(pardict)

    # Compute the values at the central point
    _, _, Da_fid, Hz_fid, f_fid, sigma8_fid, sigma12_fid, r_d_fid = run_camb(birdmodel.pardict)
    fbc = float(pardict["omega_b"]) / float(pardict["omega_cdm"])

    # Extract the name of the chainfile and read it in
    if fixed_h:
        h_str = "fixedh"
    else:
        h_str = "varyh"
    if pardict["do_marg"]:
        marg_str = "marg"
    else:
        marg_str = "all"
    taylor_strs = ["grid", "1order", "2order", "3order", "4order"]
    if birdmodel.pardict["do_corr"]:
        chainfile = str(
            "%s_xi_%2d_%3d_%s_%s_%s"
            % (
                birdmodel.pardict["fitfile"],
                birdmodel.pardict["xfit_min"],
                birdmodel.pardict["xfit_max"],
                taylor_strs[pardict["taylor_order"]],
                h_str,
                marg_str,
            )
        )
    else:
        chainfile = str(
            "%s_pk_%3.2lf_%3.2lf_%s_%s_%s"
            % (
                birdmodel.pardict["fitfile"],
                birdmodel.pardict["xfit_min"],
                birdmodel.pardict["xfit_max"],
                taylor_strs[pardict["taylor_order"]],
                h_str,
                marg_str,
            )
        )
    oldfile = chainfile + ".dat"
    newfile = chainfile + "_converted.dat"
    burntin, bestfit, like = read_chain(oldfile)

    lower_bounds = birdmodel.valueref - birdmodel.pardict["order"] * birdmodel.delta
    upper_bounds = birdmodel.valueref + birdmodel.pardict["order"] * birdmodel.delta

    # Loop over the parameters in the chain and use the grids to compute the derived parameters
    chainvals = []
    for i, (vals, loglike) in enumerate(zip(burntin, like)):
        if i % 1000 == 0:
            print(i)
        if fixed_h:
            h = float(pardict["h"])
            om = vals[1]
            b1 = vals[2]
        else:
            h = vals[1]
            om = vals[2]
            b1 = vals[3]
        omega_cdm = om * h ** 2 / (1.0 + fbc)
        omega_b = om * h ** 2 - omega_cdm
        if np.any(np.less([vals[0], h, omega_cdm, omega_b], lower_bounds)) or np.any(
            np.greater([vals[0], h, omega_cdm, omega_b], upper_bounds)
        ):
            continue
        Da, Hz, f, sigma8, sigma12, r_d = birdmodel.compute_params([vals[0], h, omega_cdm, omega_b])
        alpha_perp = (Da / h) * (float(pardict["h"]) / Da_fid) * (r_d_fid / r_d)
        alpha_par = (float(pardict["h"]) * Hz_fid) / (h * Hz) * (r_d_fid / r_d)
        chainvals.append((alpha_perp, alpha_par, f * sigma8, f * sigma12, b1 * sigma8, b1 * sigma12, loglike))

    np.savetxt(newfile, np.array(chainvals))
