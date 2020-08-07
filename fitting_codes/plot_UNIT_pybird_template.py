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
    if pardict["do_corr"]:
        chainfiles = [
            "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_Stage2/output_files/chain_UNIT_HODsnap97_ELGv1_3Gpc_covFixAmp_smallgrid_pk_35hex35_2order_hex_marg_template.hdf5",
            "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_Stage2/output_files/chain_UNIT_HODsnap97_ELGv1_3Gpc_covFixAmp_smallgrid_pk_30hex30_2order_hex_marg_template.hdf5",
            "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_Stage2/output_files/chain_UNIT_HODsnap97_ELGv1_3Gpc_covFixAmp_smallgrid_pk_25hex25_2order_hex_marg_template.hdf5",
            "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_Stage2/output_files/chain_UNIT_HODsnap97_ELGv1_3Gpc_covFixAmp_smallgrid_pk_20hex25_2order_hex_marg_template.hdf5",
            "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_Stage2/output_files/chain_UNIT_HODsnap97_ELGv1_3Gpc_covFixAmp_smallgrid_pk_20hex20_2order_hex_marg_template.hdf5",
            "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_Stage2/output_files/chain_UNIT_HODsnap97_ELGv1_3Gpc_covFixAmp_smallgrid_pk_15hex25_2order_hex_marg_template.hdf5",
            "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_Stage2/output_files/chain_UNIT_HODsnap97_ELGv1_3Gpc_covFixAmp_smallgrid_pk_15hex20_2order_hex_marg_template.hdf5",
            "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_Stage2/output_files/chain_UNIT_HODsnap97_ELGv1_3Gpc_covFixAmp_smallgrid_pk_15hex15_2order_hex_marg_template.hdf5",
            "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_Stage2/output_files/chain_UNIT_HODsnap97_ELGv1_3Gpc_covFixAmp_smallgrid_pk_10hex25_2order_hex_marg_template.hdf5",
            "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_Stage2/output_files/chain_UNIT_HODsnap97_ELGv1_3Gpc_covFixAmp_smallgrid_pk_10hex20_2order_hex_marg_template.hdf5",
            "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_Stage2/output_files/chain_UNIT_HODsnap97_ELGv1_3Gpc_covFixAmp_smallgrid_pk_10hex15_2order_hex_marg_template.hdf5",
            "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_Stage2/output_files/chain_UNIT_HODsnap97_ELGv1_3Gpc_covFixAmp_smallgrid_pk_10hex10_2order_hex_marg_template.hdf5",
        ]
        figfile = [
            "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_Stage2/chain_UNIT_HODsnap97_ELGv1_3Gpc_covFixAmp_smallgrid_pk_2order_hex_marg_template.pdf",
            "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_Stage2/chain_UNIT_HODsnap97_ELGv1_3Gpc_covFixAmp_smallgrid_pk_2order_hex_marg_template_1d.pdf",
            "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_Stage2/chain_UNIT_HODsnap97_ELGv1_3Gpc_covFixAmp_smallgrid_pk_2order_hex_marg_template_summary.pdf",
        ]
        names = [
            r"$s_{\mathrm{min}}=35\,\&\,s^{\ell=4}_{\mathrm{min}}=\mathrm{35}\,h\mathrm{Mpc^{-1}}$",
            r"$s_{\mathrm{min}}=30\,\&\,s^{\ell=4}_{\mathrm{min}}=\mathrm{30}\,h\mathrm{Mpc^{-1}}$",
            r"$s_{\mathrm{min}}=25\,\&\,s^{\ell=4}_{\mathrm{min}}=\mathrm{25}\,h\mathrm{Mpc^{-1}}$",
            r"$s_{\mathrm{min}}=20\,\&\,s^{\ell=4}_{\mathrm{min}}=\mathrm{25}\,h\mathrm{Mpc^{-1}}$",
            r"$s_{\mathrm{min}}=20\,\&\,s^{\ell=4}_{\mathrm{min}}=\mathrm{20}\,h\mathrm{Mpc^{-1}}$",
            r"$s_{\mathrm{min}}=15\,\&\,s^{\ell=4}_{\mathrm{min}}=\mathrm{25}\,h\mathrm{Mpc^{-1}}$",
            r"$s_{\mathrm{min}}=15\,\&\,s^{\ell=4}_{\mathrm{min}}=\mathrm{20}\,h\mathrm{Mpc^{-1}}$",
            r"$s_{\mathrm{min}}=15\,\&\,s^{\ell=4}_{\mathrm{min}}=\mathrm{15}\,h\mathrm{Mpc^{-1}}$",
            r"$s_{\mathrm{min}}=10\,\&\,s^{\ell=4}_{\mathrm{min}}=\mathrm{25}\,h\mathrm{Mpc^{-1}}$",
            r"$s_{\mathrm{min}}=10\,\&\,s^{\ell=4}_{\mathrm{min}}=\mathrm{20}\,h\mathrm{Mpc^{-1}}$",
            r"$s_{\mathrm{min}}=10\,\&\,s^{\ell=4}_{\mathrm{min}}=\mathrm{15}\,h\mathrm{Mpc^{-1}}$",
            r"$s_{\mathrm{min}}=10\,\&\,s^{\ell=4}_{\mathrm{min}}=\mathrm{10}\,h\mathrm{Mpc^{-1}}$",
        ]
    else:
        chainfiles = [
            # "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_Stage2/output_files/chain_UNIT_HODsnap97_ELGv1_3Gpc_covFixAmp_smallgrid_pk_0.10hex0.10_2order_hex_marg_template.hdf5",
            "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_Stage2/output_files/chain_UNIT_HODsnap97_ELGv1_3Gpc_covFixAmp_smallgrid_pk_0.15hex0.15_2order_hex_marg_template.hdf5",
            "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_Stage2/output_files/chain_UNIT_HODsnap97_ELGv1_3Gpc_covFixAmp_smallgrid_pk_0.20hex0.15_2order_hex_marg_template.hdf5",
            "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_Stage2/output_files/chain_UNIT_HODsnap97_ELGv1_3Gpc_covFixAmp_smallgrid_pk_0.20hex0.20_2order_hex_marg_template.hdf5",
            "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_Stage2/output_files/chain_UNIT_HODsnap97_ELGv1_3Gpc_covFixAmp_smallgrid_pk_0.25hex0.15_2order_hex_marg_template.hdf5",
            "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_Stage2/output_files/chain_UNIT_HODsnap97_ELGv1_3Gpc_covFixAmp_smallgrid_pk_0.25hex0.20_2order_hex_marg_template.hdf5",
            "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_Stage2/output_files/chain_UNIT_HODsnap97_ELGv1_3Gpc_covFixAmp_smallgrid_pk_0.25hex0.25_2order_hex_marg_template.hdf5",
            "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_Stage2/output_files/chain_UNIT_HODsnap97_ELGv1_3Gpc_covFixAmp_smallgrid_pk_0.30hex0.15_2order_hex_marg_template.hdf5",
            "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_Stage2/output_files/chain_UNIT_HODsnap97_ELGv1_3Gpc_covFixAmp_smallgrid_pk_0.30hex0.20_2order_hex_marg_template.hdf5",
            "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_Stage2/output_files/chain_UNIT_HODsnap97_ELGv1_3Gpc_covFixAmp_smallgrid_pk_0.30hex0.25_2order_hex_marg_template.hdf5",
            "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_Stage2/output_files/chain_UNIT_HODsnap97_ELGv1_3Gpc_covFixAmp_smallgrid_pk_0.30hex0.30_2order_hex_marg_template.hdf5",
            "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_Stage2/output_files/chain_UNIT_HODsnap97_ELGv1_3Gpc_covFixAmp_smallgrid_pk_0.40hex0.15_2order_hex_marg_template.hdf5",
            "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_Stage2/output_files/chain_UNIT_HODsnap97_ELGv1_3Gpc_covFixAmp_smallgrid_pk_0.40hex0.20_2order_hex_marg_template.hdf5",
            "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_Stage2/output_files/chain_UNIT_HODsnap97_ELGv1_3Gpc_covFixAmp_smallgrid_pk_0.40hex0.25_2order_hex_marg_template.hdf5",
            # "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_Stage2/output_files/chain_UNIT_HODsnap97_ELGv1_3Gpc_covFixAmp_smallgrid_pk_0.40hex0.40_2order_hex_marg_template.hdf5",
        ]
        figfile = [
            "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_Stage2/chain_UNIT_HODsnap97_ELGv1_3Gpc_covFixAmp_smallgrid_pk_2order_hex_marg_template.pdf",
            "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_Stage2/chain_UNIT_HODsnap97_ELGv1_3Gpc_covFixAmp_smallgrid_pk_2order_hex_marg_template_1d.pdf",
            "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_Stage2/chain_UNIT_HODsnap97_ELGv1_3Gpc_covFixAmp_smallgrid_pk_2order_hex_marg_template_summary.pdf",
        ]
        names = [
            # r"$k_{\mathrm{max}}=0.10\,\&\,k^{\ell=4}_{\mathrm{max}}=\mathrm{0.10}\,h\mathrm{Mpc^{-1}}$",
            r"$k_{\mathrm{max}}=0.15\,\&\,k^{\ell=4}_{\mathrm{max}}=\mathrm{0.15}\,h\mathrm{Mpc^{-1}}$",
            r"$k_{\mathrm{max}}=0.20\,\&\,k^{\ell=4}_{\mathrm{max}}=\mathrm{0.15}\,h\mathrm{Mpc^{-1}}$",
            r"$k_{\mathrm{max}}=0.20\,\&\,k^{\ell=4}_{\mathrm{max}}=\mathrm{0.20}\,h\mathrm{Mpc^{-1}}$",
            r"$k_{\mathrm{max}}=0.25\,\&\,k^{\ell=4}_{\mathrm{max}}=\mathrm{0.15}\,h\mathrm{Mpc^{-1}}$",
            r"$k_{\mathrm{max}}=0.25\,\&\,k^{\ell=4}_{\mathrm{max}}=\mathrm{0.20}\,h\mathrm{Mpc^{-1}}$",
            r"$k_{\mathrm{max}}=0.25\,\&\,k^{\ell=4}_{\mathrm{max}}=\mathrm{0.25}\,h\mathrm{Mpc^{-1}}$",
            r"$k_{\mathrm{max}}=0.30\,\&\,k^{\ell=4}_{\mathrm{max}}=\mathrm{0.15}\,h\mathrm{Mpc^{-1}}$",
            r"$k_{\mathrm{max}}=0.30\,\&\,k^{\ell=4}_{\mathrm{max}}=\mathrm{0.20}\,h\mathrm{Mpc^{-1}}$",
            r"$k_{\mathrm{max}}=0.30\,\&\,k^{\ell=4}_{\mathrm{max}}=\mathrm{0.25}\,h\mathrm{Mpc^{-1}}$",
            r"$k_{\mathrm{max}}=0.30\,\&\,k^{\ell=4}_{\mathrm{max}}=\mathrm{0.30}\,h\mathrm{Mpc^{-1}}$",
            r"$k_{\mathrm{max}}=0.40\,\&\,k^{\ell=4}_{\mathrm{max}}=\mathrm{0.15}\,h\mathrm{Mpc^{-1}}$",
            r"$k_{\mathrm{max}}=0.40\,\&\,k^{\ell=4}_{\mathrm{max}}=\mathrm{0.20}\,h\mathrm{Mpc^{-1}}$",
            r"$k_{\mathrm{max}}=0.40\,\&\,k^{\ell=4}_{\mathrm{max}}=\mathrm{0.25}\,h\mathrm{Mpc^{-1}}$",
            # r"$k_{\mathrm{max}}=0.40\,\&\,k^{\ell=4}_{\mathrm{max}}=\mathrm{0.40}\,h\mathrm{Mpc^{-1}}$",
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
        paramnames = [r"$\alpha_{\perp}$", r"$\alpha_{||}$", r"$f\sigma_{8}$"]
        c.add_chain(burntin[:, :3], parameters=paramnames, name=names[chaini], posterior=like)
        bestfits.append(bestfit)

    print(bestfits)
    extents = [(0.985, 1.008), (0.978, 1.024), (0.415, 0.465)]
    c.configure(bar_shade=True)
    fig = c.plotter.plot_summary(filename=figfile[1], truth=truths, display=False, extents=extents)
    print(c.analysis.get_summary())

    # Get the bestfit bird model. For do_marg, we get the analytically marginalised best-fit parameters using
    # least-squares at the mapkmum likelihood value from the chain
    if False:

        import matplotlib.pyplot as plt

        fig = plt.figure()
        params = bestfits[-1]
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

        if birdmodel.pardict["do_marg"]:
            bs_analytic = birdmodel.compute_bestfit_analytic(Pi, fittingdata.data)
            print(bs_analytic)
            pardict["do_marg"] = 0
            b2 = (params[-2] + params[-1]) / np.sqrt(2.0)
            b4 = (params[-2] - params[-1]) / np.sqrt(2.0)
            bs = [
                params[-3],
                b2,
                bs_analytic[0],
                b4,
                bs_analytic[1],
                bs_analytic[2],
                bs_analytic[3],
                bs_analytic[4],
                bs_analytic[5],
                bs_analytic[6],
            ]
            P_model, P_model_interp = birdmodel.compute_model(bs, Plin, Ploop, fittingdata.data["x_data"])
            chi_squared = birdmodel.compute_chi2(P_model_interp, Pi, fittingdata.data)
        print(params, chi_squared)

        update_plot(pardict, fittingdata.data["x_data"], P_model_interp, plt, keep=True)

        # np.savetxt(
        #    "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_HandShake/chain_UNIT_HODsnap97_ELGv1_pk_0.00_0.30_nohex_all_bestfit_template.dat",
        #    np.c_[
        #        fittingdata.data["x_data"],
        #        P_model[: len(fittingdata.data["x_data"])],
        #        P_model[len(fittingdata.data["x_data"]) :],
        #    ],
        #    header="k       P0          P2",
        # )
