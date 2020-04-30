import sys
import numpy as np
import pandas as pd
from configobj import ConfigObj
from chainconsumer import ChainConsumer

sys.path.append("../")
from tbird.Grid import run_camb
from fitting_codes.fitting_utils import read_chain


if __name__ == "__main__":

    # First read in the config file and compute Da_fid, Hz_fid and sigma8_fid
    configfile = sys.argv[1]
    pardict = ConfigObj(configfile)
    _, _, Da_fid, Hz_fid, fN_fid, sigma8_fid, sigma12_fid, r_d_fid = run_camb(pardict)

    # Set the chainfiles and names for each chain
    # chainfiles = [
    #    "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_HandShake/chain_UNIT_HODsnap97_ELGv1_pk_0.00_0.30_grid_varyh_all.dat",
    #    "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_HandShake/chain_UNIT_HODsnap97_ELGv1_pk_0.00_0.30_grid_varyh_marg.dat",
    #    "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_HandShake/chain_UNIT_HODsnap97_ELGv1_pk_0.00_0.30_3order_varyh_marg.dat",
    # ]
    # figfile = [
    #    "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_HandShake/chain_UNIT_HODsnap97_ELGv1_pk_0.00_0.30_varyh.pdf"
    # ]
    # fixed_hs = [False, False, False]
    # names = [r"$\mathrm{Grid;\,No\,Marg}$", r"$\mathrm{Grid;\,Marg}$", r"$\mathrm{3^{rd}\,Order;\,Marg}$"]

    chainfiles = [
        "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_HandShake/chain_UNIT_HODsnap97_ELGv1_pk_0.00_0.30_grid_varyh_all.dat",
        "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_HandShake/chain_UNIT_HODsnap97_ELGv1_xi_20_200_3order_varyh_all.dat",
    ]
    figfile = ["/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_HandShake/chain_UNIT_HODsnap97_ELGv1_pk_xi_varyh.pdf"]
    fixed_hs = [False, False]
    names = [r"$P(k)$", r"$\xi(s)$"]

    truths = {
        r"$A_{s}\times 10^{9}$": np.exp(float(pardict["ln10^{10}A_s"])) / 1.0e1,
        r"$h$": float(pardict["h"]),
        r"$\Omega_{m}$": (float(pardict["omega_cdm"]) + float(pardict["omega_b"])) / float(pardict["h"]) ** 2,
    }

    # Output name for the figure

    c = ChainConsumer()

    bestfits = []
    for chaini, (chainfile, fixed_h) in enumerate(zip(chainfiles, fixed_hs)):

        burntin, bestfit, like = read_chain(chainfile, burnlimitup=20000)
        burntin[:, 0] = np.exp(burntin[:, 0]) / 1.0e1
        if fixed_h:
            paramnames = [r"$A_{s}\times 10^{9}$", r"$\Omega_{m}$", r"$b_{1}$"]
            c.add_chain(burntin[:, :3], parameters=paramnames, name=names[chaini], posterior=like)
        else:
            paramnames = [r"$A_{s}\times 10^{9}$", r"$h$", r"$\Omega_{m}$", r"$b_{1}$"]
            c.add_chain(burntin[:, :4], parameters=paramnames, name=names[chaini], posterior=like)
        bestfits.append(bestfit)

    print(bestfits)
    fig = c.plotter.plot(figsize="column", filename=figfile, truth=truths)
    print(c.analysis.get_summary())
