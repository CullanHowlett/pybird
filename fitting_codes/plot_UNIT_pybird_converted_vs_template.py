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
        _, _, Om_fid, Da_fid, Hz_fid, fN_fid, sigma8_fid, sigma8_0_fid, sigma12_fid, r_d_fid = run_camb(pardict)
    else:
        _, _, Om_fid, Da_fid, Hz_fid, fN_fid, sigma8_fid, sigma8_0_fid, sigma12_fid, r_d_fid = run_class(pardict)

    chainfiles = [
        "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_Stage2/output_files/New_chain_UNIT_HODsnap97_ELGv1_3Gpc_FixAmp_pk_0.20hex0.20_3order_hex_marg_BBNprior_converted.dat",
        "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_Stage2/output_files/New_chain_UNIT_HODsnap97_ELGv1_3Gpc_FixAmp_pk_0.20hex0.20_3order_hex_marg_fixedrat_converted.dat",
        "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_Stage2/output_files/New_chain_UNIT_HODsnap97_ELGv1_3Gpc_FixAmp_pk_0.20hex0.20_3order_hex_marg_template.hdf5",
    ]
    templates = [False, False, True]
    figfile = [
        "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_Stage2/New_chain_UNIT_HODsnap97_ELGv1_3Gpc_FixAmp_pk_0.20hex0.20_3order_hex_marg_converted_vs_template.pdf"
    ]
    names = [
        r"$\mathrm{COSMO+BBN}$",
        r"$\mathrm{COSMO+FIXED}\,\Omega_{cdm}/\Omega_{b}$",
        r"$\mathrm{TEMPLATE}$",
        # r"$\xi(s);\,\mathrm{30-200}h^{-1}\mathrm{Mpc}\,\mathrm{COSMO}$",
        # r"$\xi(s);\,\mathrm{30-200}h^{-1}\mathrm{Mpc}\,\mathrm{TEMPLATE}$",
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
    paramnames = [
        r"$\alpha_{\perp}$",
        r"$\alpha_{||}$",
        r"$D_{A}(z)$",
        r"$H(z)$",
        r"$f\sigma_{8}$",
    ]
    for chaini, (chainfile, template) in enumerate(zip(chainfiles, templates)):

        print(chainfile)
        if template:
            burntin, bestfit, like = read_chain_backend(chainfile)
            Da = 2997.92458 * Da_fid / float(pardict["h"]) * burntin[:, 0]
            Hz = 100.0 * float(pardict["h"]) * Hz_fid / burntin[:, 1]
            burntin = np.hstack((burntin, Da[:, None], Hz[:, None]))
            c.add_chain(burntin[:, [0, 1, -2, -1, 2]], parameters=paramnames, name=names[chaini], posterior=like)
        else:
            burntin = np.array(pd.read_csv(chainfile, delim_whitespace=True, header=None))
            like = burntin[:, -1]
            bestfit = burntin[np.argmax(burntin[:, -1]), :-1]
            fsigma8 = burntin[:, 9] * burntin[:, 10]
            burntin = np.hstack((burntin, fsigma8[:, None]))
            c.add_chain(burntin[:, [4, 5, 7, 8, -1]], parameters=paramnames, name=names[chaini], posterior=like)
        bestfits.append(bestfit)

    print(bestfits)
    fig = c.plotter.plot(figsize="column", filename=figfile, truth=truths)
    print(c.analysis.get_summary())
