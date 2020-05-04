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
    #    "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_HandShake/chain_UNIT_HODsnap97_ELGv1_pk_0.00_0.30_grid_varyh_nohex_all_converted.dat",
    #    "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_HandShake/chain_UNIT_HODsnap97_ELGv1_pk_0.00_0.30_grid_varyh_nohex_marg_converted.dat",
    #    "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_HandShake/chain_UNIT_HODsnap97_ELGv1_pk_0.00_0.30_3order_varyh_nohex_marg_converted.dat",
    # ]
    # figfile = [
    #    "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_HandShake/chain_UNIT_HODsnap97_ELGv1_pk_0.00_0.30_varyh_converted_alpha.pdf"
    # ]
    # fixed_hs = [False, False, False]
    # names = [r"$\mathrm{Grid;\,No\,Marg}$", r"$\mathrm{Grid;\,Marg}$", r"$\mathrm{3^{rd}\,Order;\,Marg}$"]

    chainfiles = [
        "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_HandShake/chain_UNIT_HODsnap97_ELGv1_pk_0.00_0.30_3order_fixedh_nohex_marg_converted.dat",
        "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_HandShake/chain_UNIT_HODsnap97_ELGv1_pk_0.00_0.30_3order_varyh_nohex_marg_converted.dat",
    ]
    figfile = [
        "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_HandShake/chain_UNIT_HODsnap97_ELGv1_pk_0.00_0.30_nohex_converted_alpha.pdf"
    ]
    fixed_hs = [True, False]
    names = [r"$\mathrm{Fixed\,}h$", r"$\mathrm{Vary\,}h$"]

    # chainfiles = [
    #    "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_HandShake/chain_UNIT_HODsnap97_ELGv1_pk_0.00_0.30_grid_varyh_all.dat",
    #    "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_HandShake/chain_UNIT_HODsnap97_ELGv1_xi_20_200_3order_varyh_all.dat",
    # ]
    # figfile = ["/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_HandShake/chain_UNIT_HODsnap97_ELGv1_pk_xi_varyh.pdf"]
    # fixed_hs = [False, False]
    # names = [r"$P(k)$", r"$\xi(s)$"]

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
    for chaini, (chainfile, fixed_h) in enumerate(zip(chainfiles, fixed_hs)):

        burntin = np.array(pd.read_csv(chainfile, delim_whitespace=True, header=None))
        like = burntin[:, -1]
        bestfit = burntin[np.argmax(burntin[:, -1]), :-1]
        # paramnames = [r"$D_{A}(z)$", r"$H(z)$", r"$f\sigma_{8}$", r"$b_{1}\sigma_{8}$"]
        # c.add_chain(burntin[:, [2, 3, 4, 6]], parameters=paramnames, name=names[chaini], posterior=like)
        paramnames = [r"$\alpha_{\perp}$", r"$\alpha_{||}$", r"$f\sigma_{8}$", r"$b_{1}\sigma_{8}$"]
        c.add_chain(burntin[:, [0, 1, 4, 6]], parameters=paramnames, name=names[chaini], posterior=like)
        bestfits.append(bestfit)

    print(bestfits)
    fig = c.plotter.plot(figsize="column", filename=figfile, truth=truths)
    print(c.analysis.get_summary())
