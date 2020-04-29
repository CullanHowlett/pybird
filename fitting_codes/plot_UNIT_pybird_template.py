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
    do_sigma12 = int(sys.argv[2])
    pardict = ConfigObj(configfile)
    _, _, Da_fid, Hz_fid, fN_fid, sigma8_fid, sigma12_fid, r_d_fid = run_camb(pardict)

    # Set the chainfiles and names for each chain
    chainfiles = [
        "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_HandShake/chain_UNIT_HODsnap97_ELGv1_pk_0.00_0.30_grid_varyh_marg_converted.dat",
        "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_HandShake/chain_UNIT_HODsnap97_ELGv1_pk_0.00_0.30_grid_fixedh_marg_converted.dat",
        "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_HandShake/chain_UNIT_HODsnap97_ELGv1_pk_0.00_0.30_grid_template_marg.dat",
    ]
    preprocessed = [True, True, False]
    names = [r"$\mathrm{Varying\,h}$", r"$\mathrm{Fixed\,h}$", r"$\mathrm{Template}$"]
    if do_sigma12:
        truths = [1.0, 1.0, fN_fid * sigma12_fid]
        paramnames = [r"$\alpha_{\perp}$", r"$\alpha_{||}$", r"$f\sigma_{12}$", r"$b_{1}\sigma_{12}$"]
    else:
        truths = [1.0, 1.0, fN_fid * sigma8_fid]
        paramnames = [r"$\alpha_{\perp}$", r"$\alpha_{||}$", r"$f\sigma_{8}$", r"$b_{1}\sigma_{8}$"]

    # Output name for the figure
    figfile = [
        "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_HandShake/chain_UNIT_HODsnap97_ELGv1_pk_0.00_0.30_grid_template.pdf"
    ]

    c = ChainConsumer()

    bestfits = []
    for chaini, (chainfile, processed) in enumerate(zip(chainfiles, preprocessed)):

        print(chainfile, processed)
        if processed:
            burntin = np.array(pd.read_csv(chainfile, delim_whitespace=True, header=None))
            like = burntin[:, -1]
            bestfit = burntin[np.argmax(burntin[:, -1]), :-1]
            if do_sigma12:
                c.add_chain(
                    burntin[:, [0, 1, 3, 5]], parameters=paramnames, name=names[chaini], posterior=like, plot_point=True
                )
            else:
                c.add_chain(
                    burntin[:, [0, 1, 2, 4]], parameters=paramnames, name=names[chaini], posterior=like, plot_point=True
                )
        else:
            burntin, bestfit, like = read_chain(chainfile, burnlimitup=20000)
            if do_sigma12:
                bestfit[2:4] *= sigma12_fid
                burntin[:, 2:4] *= sigma12_fid
            else:
                bestfit[2:4] *= sigma8_fid
                burntin[:, 2:4] *= sigma8_fid
            c.add_chain(burntin[:, :4], parameters=paramnames, name=names[chaini], posterior=like, plot_point=True)
        bestfits.append(bestfit)

    print(bestfits)
    fig = c.plotter.plot(figsize="column", filename=figfile, truth=truths)
    print(c.analysis.get_summary())
