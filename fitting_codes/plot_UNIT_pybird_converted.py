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
        "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_Stage2/output_files/New_chain_UNIT_HODsnap97_ELGv1_3Gpc_FixAmp_pk_0.20hex0.20_3order_hex_marg_BBNprior.hdf5",
        "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_Stage2/output_files/New_chain_UNIT_HODsnap97_ELGv1_3Gpc_FixAmp_pk_0.20hex0.20_3order_hex_marg_fixedrat.hdf5",
        "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_Stage2/output_files/chain_UNIT_HODsnap97_ELGv1_3Gpc_covFixAmp_smallgrid_BBNprior_pk_0.20hex0.20_2order_hex_marg.hdf5",
    ]
    figfile = [
        "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_Stage2/New_chain_UNIT_HODsnap97_ELGv1_3Gpc_FixAmp_pk_0.20hex0.20_4order_hex_marg_converted.pdf"
    ]
    names = [
        r"$\mathrm{Fixed}\,\Omega_{b}/\Omega_{cdm}$",
        r"$\mathrm{BBN\,Prior}$",
    ]

    truths = {
        r"$\Omega_{m}$": Om_fid,
        r"$H_{0}$": 100.0 * float(pardict["h"]),
        r"$\sigma_{8}$": sigma8_fid,
    }

    # Output name for the figure

    c = ChainConsumer()

    bestfits = []
    for chaini, chainfile in enumerate(chainfiles):

        burntin = np.array(pd.read_csv(chainfile, delim_whitespace=True, header=None))
        like = burntin[:, -1]
        bestfit = burntin[np.argmax(burntin[:, -1]), :-1]
        # paramnames = [r"$\alpha_{\perp}$", r"$\alpha_{||}$", r"$f\sigma_{8}$", r"$b_{1}\sigma_{8}$"]
        paramnames = [r"$\Omega_{m}$", r"$H_{0}$", r"$\sigma_{8}$"]
        c.add_chain(burntin[:, [6, 1, 10]], parameters=paramnames, name=names[chaini], posterior=like)
        bestfits.append(bestfit)

    print(bestfits)
    fig = c.plotter.plot(figsize="column", filename=figfile, truth=truths)
    print(c.analysis.get_summary())
