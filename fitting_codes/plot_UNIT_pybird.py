import sys
import numpy as np
import pandas as pd
from configobj import ConfigObj
from chainconsumer import ChainConsumer

sys.path.append("../")
from tbird.Grid import run_camb
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
    _, _, Da_fid, Hz_fid, fN_fid, sigma8_fid, sigma12_fid, r_d_fid = run_camb(pardict)

    # Set the chainfiles and names for each chain
    chainfiles = [
        "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_HandShake/chain_UNIT_HODsnap97_ELGv1_pk_0.00_0.30_grid_varyh_nohex_marg.hdf5",
        "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_HandShake/chain_UNIT_HODsnap97_ELGv1_xi_30_200_grid_varyh_nohex_marg.hdf5",
        "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_HandShake/chain_UNIT_HODsnap97_ELGv1_xi_50_200_grid_varyh_nohex_marg.hdf5",
    ]
    figfile = [
        "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_HandShake/chain_UNIT_HODsnap97_ELGv1_pk_xi_grid_vary_nohex.pdf"
    ]
    fixed_hs = [False, False, False]
    names = [
        r"$P(k)$;\,\mathrm{0.00-0.30}h\mathrm{Mpc^{-1}}",
        r"$\xi(s);\,\mathrm{30-200}h^{-1}\mathrm{Mpc}$",
        r"$\xi(s);\,\mathrm{50-200}h^{-1}\mathrm{Mpc}$",
    ]

    # chainfiles = [
    #    "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_HandShake/chain_UNIT_HODsnap97_ELGv1_pk_0.00_0.30_3order_fixedh_nohex_marg.dat",
    #    "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_HandShake/chain_UNIT_HODsnap97_ELGv1_pk_0.00_0.30_3order_varyh_nohex_marg.dat",
    # ]
    # figfile = [
    #    "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_HandShake/chain_UNIT_HODsnap97_ELGv1_pk_0.00_0.30_nohex.pdf"
    # ]
    # fixed_hs = [True, False]
    # names = [r"$\mathrm{Fixed\,}h$", r"$\mathrm{Vary\,}h$"]

    # chainfiles = [
    #    "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_HandShake/chain_UNIT_HODsnap97_ELGv1_pk_0.00_0.30_grid_varyh_all.dat",
    #    "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_HandShake/chain_UNIT_HODsnap97_ELGv1_xi_20_200_3order_varyh_all.dat",
    # ]
    # figfile = ["/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_HandShake/chain_UNIT_HODsnap97_ELGv1_pk_xi_varyh.pdf"]
    # fixed_hs = [False, False]
    # names = [r"$P(k)$", r"$\xi(s)$"]

    truths = {
        r"$A_{s}\times 10^{9}$": np.exp(float(pardict["ln10^{10}A_s"])) / 1.0e1,
        r"$h$": float(pardict["h"]),
        r"$\Omega_{m}$": (float(pardict["omega_cdm"]) + float(pardict["omega_b"])) / float(pardict["h"]) ** 2,
    }

    # Output name for the figure

    c = ChainConsumer()

    bestfits = []
    for chaini, (chainfile, fixed_h) in enumerate(zip(chainfiles, fixed_hs)):

        burntin, bestfit, like = read_chain_backend(chainfile)
        burntin[:, 0] = np.exp(burntin[:, 0]) / 1.0e1
        if fixed_h:
            paramnames = [r"$A_{s}\times 10^{9}$", r"$\Omega_{m}$", r"$b_{1}$"]
            c.add_chain(burntin[:, :3], parameters=paramnames, name=names[chaini], posterior=like)
        else:
            paramnames = [r"$A_{s}\times 10^{9}$", r"$h$", r"$\Omega_{m}$", r"$b_{1}$"]
            c.add_chain(burntin[:, :4], parameters=paramnames, name=names[chaini], posterior=like)
        bestfits.append(bestfit)

    print(bestfits)
    fig = c.plotter.plot(figsize="column", filename=figfile, truth=truths, display=True)
    print(c.analysis.get_summary())

    # Get the bestfit bird model
    if False:
        params = bestfits[0]
        shot_noise = 309.210197  # Taken from the header of the data power spectrum file.
        fittingdata = FittingData(pardict, shot_noise=shot_noise)

        # Set up the BirdModel
        birdmodel = BirdModel(pardict, template=False)

        # Plotting (for checking/debugging, should turn off for production runs)
        plt = create_plot(pardict, fittingdata)

        ln10As, h, Omega_m = params[:3]
        fbc = float(birdmodel.valueref[3]) / float(birdmodel.valueref[2])
        omega_cdm = Omega_m / (1.0 + fbc) * h ** 2
        omega_b = Omega_m * h ** 2 - omega_cdm

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
        Plin, Ploop = birdmodel.compute_pk([ln10As, h, omega_cdm, omega_b])
        P_model = birdmodel.compute_model(bs, Plin, Ploop, fittingdata.data["x_data"])
        Pi = birdmodel.get_Pi_for_marg(Ploop, bs[0], fittingdata.data["shot_noise"], fittingdata.data["x_data"])

        chi_squared = birdmodel.compute_chi2(P_model, Pi, fittingdata.data)
        update_plot(pardict, fittingdata, P_model, plt, keep=True)
        print(params, chi_squared)

        np.savetxt(
            "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_HandShake/chain_UNIT_HODsnap97_ELGv1_pk_0.00_0.30_varyh_nohex_all_bestfit.dat",
            np.c_[
                fittingdata.data["x_data"],
                P_model[: len(fittingdata.data["x_data"])],
                P_model[len(fittingdata.data["x_data"]) :],
            ],
            header="k       P0          P2",
        )
