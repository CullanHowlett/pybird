import sys
import numpy as np
import pandas as pd
from configobj import ConfigObj
from chainconsumer import ChainConsumer

sys.path.append("../")
from tbird.Grid import run_camb, run_class
from fitting_codes.fitting_utils import read_chain_backend


if __name__ == "__main__":

    # First read in the config file and compute Da_fid, Hz_fid and sigma8_fid
    configfile = sys.argv[1]
    pardict = ConfigObj(configfile)
    if pardict["code"] == "CAMB":
        _, _, Om_fid, Da_fid, Hz_fid, fN_fid, sigma8_fid, sigma12_fid, r_d_fid = run_camb(pardict)
    else:
        _, _, Om_fid, Da_fid, Hz_fid, fN_fid, sigma8_fid, sigma12_fid, r_d_fid = run_class(pardict)

    chainfiles = [
        "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_HandShake/chain_UNIT_HODsnap97_ELGv1_pk_0.00_0.30_grid_hex_marg_converted.dat",
        "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_HandShake/chain_UNIT_HODsnap97_ELGv1_xi_30_200_grid_hex_marg_converted.dat",
    ]
    figfile = [
        "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_HandShake/chain_UNIT_HODsnap97_ELGv1_pk_xi_grid_hex_marg_converted.pdf"
    ]
    names = [
        r"$P(k);\,\mathrm{0.00-0.30}h\mathrm{Mpc^{-1}}\,\mathrm{Marg}$",
        r"$\xi(s);\,\mathrm{30-200}h^{-1}\mathrm{Mpc}\,\mathrm{Marg}$",
    ]

    truths = {
        r"$\alpha_{\perp}$": 1.0,
        r"$\alpha_{||}$": 1.0,
        r"$D_{A}(z)$": 2997.92458 * Da_fid / float(pardict["h"]),
        r"$H(z)$": 100.0 * float(pardict["h"]) * Hz_fid,
        r"$f\sigma_{8}$": fN_fid * sigma8_fid,
    }

    # Output name for the figure

    c = ChainConsumer()

    bestfits = []
    for chaini, chainfile in enumerate(chainfiles):

        burntin = np.array(pd.read_csv(chainfile, delim_whitespace=True, header=None))
        like = burntin[:, -1]
        bestfit = burntin[np.argmax(burntin[:, -1]), :-1]
        paramnames = [r"$\alpha_{\perp}$", r"$\alpha_{||}$", r"$f\sigma_{8}$", r"$b_{1}\sigma_{8}$"]
        c.add_chain(burntin[:, [0, 1, 4, 6]], parameters=paramnames, name=names[chaini], posterior=like)
        bestfits.append(bestfit)

    print(bestfits)
    fig = c.plotter.plot(figsize="column", filename=figfile, truth=truths)
    print(c.analysis.get_summary())
