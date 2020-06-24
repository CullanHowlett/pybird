import sys
import numpy as np
import pandas as pd
from configobj import ConfigObj
from chainconsumer import ChainConsumer

sys.path.append("../")
from tbird.Grid import run_camb, run_class
from fitting_codes.fitting_utils import (
    read_chain_backend,
    BirdModel,
    FittingData,
    create_plot,
    update_plot,
    format_pardict,
)


if __name__ == "__main__":

    # First read in the config file and compute Da_fid, Hz_fid and sigma8_fid
    configfile = sys.argv[1]
    pardict = ConfigObj(configfile)
    pardict = format_pardict(pardict)
    if pardict["code"] == "CAMB":
        _, _, Om_fid, Da_fid, Hz_fid, fN_fid, sigma8_fid, sigma12_fid, r_d_fid = run_camb(pardict)
    else:
        _, _, Om_fid, Da_fid, Hz_fid, fN_fid, sigma8_fid, sigma12_fid, r_d_fid = run_class(pardict)

    # Set the chainfiles and names for each chain
    chainfiles = [
        "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_HandShake/chain_UNIT_HODsnap97_ELGv1_pk_0.00_0.30_grid_hex_marg_template.hdf5",
        "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_HandShake/chain_UNIT_HODsnap97_ELGv1_xi_30_200_grid_hex_marg_template.hdf5",
    ]
    figfile = [
        "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_HandShake/chain_UNIT_HODsnap97_ELGv1_pk_xi_grid_hex_marg_template.pdf"
    ]
    names = [
        r"$P(k);\,\mathrm{0.00-0.30}h\mathrm{Mpc^{-1}}\,\mathrm{Marg}$",
        r"$\xi(s);\,\mathrm{30-200}h^{-1}\mathrm{Mpc}\,\mathrm{Marg}$",
    ]

    truths = {
        r"$\alpha_{\perp}$": 1.0,
        r"$\alpha_{||}$": 1.0,
        r"$f\sigma_{8}$": fN_fid * sigma8_fid,
    }

    # Output name for the figure

    c = ChainConsumer()

    bestfits = []
    for chaini, chainfile in enumerate(chainfiles):

        print(chainfile)
        burntin, bestfit, like = read_chain_backend(chainfile)
        burntin[:, 3] *= sigma8_fid
        paramnames = [r"$\alpha_{\perp}$", r"$\alpha_{||}$", r"$f\sigma_{8}$", r"$b_{1}\sigma_{8}$"]
        c.add_chain(burntin[:, :4], parameters=paramnames, name=names[chaini], posterior=like)
        bestfits.append(bestfit)

    print(bestfits)
    fig = c.plotter.plot(figsize="column", filename=figfile, truth=truths, display=True)
    print(c.analysis.get_summary())

    # Get the bestfit bird model
    if True:
        params = bestfits[0]
        fittingdata = FittingData(pardict, shot_noise=float(pardict["shot_noise"]))

        # Set up the BirdModel
        birdmodel = BirdModel(pardict, template=True)

        # Plotting (for checking/debugging, should turn off for production runs)
        plt = create_plot(pardict, fittingdata)

        if birdmodel.pardict["do_marg"]:
            b2 = (params[-2] + params[-1]) / np.sqrt(2.0)
            b4 = (params[-2] - params[-1]) / np.sqrt(2.0)
            bs = [params[-3], b2, 0.0, b4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        else:
            b2 = (params[-9] + params[-7]) / np.sqrt(2.0)
            b4 = (params[-9] - params[-7]) / np.sqrt(2.0)
            bs = [
                params[-10],
                b2,
                params[-8],
                b4,
                params[-6],
                params[-5],
                params[-4],
                params[-3] * fittingdata.data["shot_noise"],
                params[-2] * fittingdata.data["shot_noise"],
                params[-1] * fittingdata.data["shot_noise"],
            ]

        # Get the bird model
        alpha_perp, alpha_par, fsigma8 = params[:3]
        f = fsigma8 / birdmodel.valueref[3]

        Plin, Ploop = birdmodel.compute_pk([alpha_perp, alpha_par, f, birdmodel.valueref[3]])
        P_model, P_model_interp = birdmodel.compute_model(bs, Plin, Ploop, fittingdata.data["x_data"])
        Pi = birdmodel.get_Pi_for_marg(Ploop, bs[0], fittingdata.data["shot_noise"], fittingdata.data["x_data"])

        chi_squared = birdmodel.compute_chi2(P_model_interp, Pi, fittingdata.data)
        print(params, chi_squared)

        # np.savetxt(
        #    "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_HandShake/chain_UNIT_HODsnap97_ELGv1_pk_0.00_0.30_nohex_all_bestfit_template.dat",
        #    np.c_[
        #        fittingdata.data["x_data"],
        #        P_model[: len(fittingdata.data["x_data"])],
        #        P_model[len(fittingdata.data["x_data"]) :],
        #    ],
        #    header="k       P0          P2",
        # )
