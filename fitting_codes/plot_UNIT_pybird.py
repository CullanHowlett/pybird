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
    update_plot_components,
    format_pardict,
)


if __name__ == "__main__":

    # First read in the config file and compute Da_fid, Hz_fid and sigma8_fid
    configfile = sys.argv[1]
    pardict = ConfigObj(configfile)
    pardict = format_pardict(pardict)
    if pardict["code"] == "CAMB":
        _, _, Om_fid, Da_fid, Hz_fid, fN_fid, sigma8_fid, sigma8_0_fid, sigma12_fid, r_d_fid = run_camb(pardict)
    else:
        _, _, Om_fid, Da_fid, Hz_fid, fN_fid, sigma8_fid, sigma8_0_fid, sigma12_fid, r_d_fid = run_class(pardict)

    # Set the chainfiles and names for each chain
    if pardict["do_corr"]:
        chainfiles = [
            "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_Stage2/output_files/chain_UNIT_HODsnap97_ELGv1_3Gpc_Std_xi_35hex35_2order_hex_marg.hdf5",
            "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_Stage2/output_files/chain_UNIT_HODsnap97_ELGv1_3Gpc_Std_xi_30hex30_2order_hex_marg.hdf5",
            "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_Stage2/output_files/chain_UNIT_HODsnap97_ELGv1_3Gpc_Std_xi_25hex25_2order_hex_marg.hdf5",
            "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_Stage2/output_files/chain_UNIT_HODsnap97_ELGv1_3Gpc_Std_xi_20hex25_2order_hex_marg.hdf5",
            "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_Stage2/output_files/chain_UNIT_HODsnap97_ELGv1_3Gpc_Std_xi_20hex20_2order_hex_marg.hdf5",
            "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_Stage2/output_files/chain_UNIT_HODsnap97_ELGv1_3Gpc_Std_xi_15hex25_2order_hex_marg.hdf5",
            "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_Stage2/output_files/chain_UNIT_HODsnap97_ELGv1_3Gpc_Std_xi_15hex20_2order_hex_marg.hdf5",
            "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_Stage2/output_files/chain_UNIT_HODsnap97_ELGv1_3Gpc_Std_xi_15hex15_2order_hex_marg.hdf5",
            "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_Stage2/output_files/chain_UNIT_HODsnap97_ELGv1_3Gpc_Std_xi_10hex25_2order_hex_marg.hdf5",
            "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_Stage2/output_files/chain_UNIT_HODsnap97_ELGv1_3Gpc_Std_xi_10hex20_2order_hex_marg.hdf5",
            "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_Stage2/output_files/chain_UNIT_HODsnap97_ELGv1_3Gpc_Std_xi_10hex15_2order_hex_marg.hdf5",
            "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_Stage2/output_files/chain_UNIT_HODsnap97_ELGv1_3Gpc_Std_xi_10hex10_2order_hex_marg.hdf5",
        ]
        figfile = [
            "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_Stage2/chain_UNIT_HODsnap97_ELGv1_3Gpc_Std_xi_2order_hex_marg.pdf",
            "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_Stage2/chain_UNIT_HODsnap97_ELGv1_3Gpc_Std_xi_2order_hex_marg_1d.pdf",
            "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_Stage2/chain_UNIT_HODsnap97_ELGv1_3Gpc_Std_xi_2order_hex_marg_summary.pdf",
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
            "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_Stage2/output_files/New_chain_UNIT_HODsnap97_ELGv1_3Gpc_FixAmp_pk_0.20hex0.20_3order_hex_marg_fixedrat.hdf5",
            "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_Stage2/output_files/New_chain_UNIT_HODsnap97_ELGv1_3Gpc_FixAmp_pk_0.20hex0.20_3order_hex_marg_BBNprior.hdf5",
            "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_Stage2/output_files/P_cb/chain_UNIT_HODsnap97_ELGv1_3Gpc_covFixAmp_smallgrid_BBNprior_pk_0.20hex0.20_2order_hex_marg.hdf5",
        ]
        figfile = [
            "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_Stage2/New_chain_UNIT_HODsnap97_ELGv1_3Gpc_FixAmp_pk_0.20hex0.20_3order_hex_marg.pdf",
            "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_Stage2/New_chain_UNIT_HODsnap97_ELGv1_3Gpc_FixAmp_pk_0.20hex0.20_3order_hex_marg_1d.pdf",
            "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_Stage2/New_chain_UNIT_HODsnap97_ELGv1_3Gpc_FixAmp_pk_0.20hex0.20_3order_hex_marg_summary.pdf",
        ]
        names = [
            r"$\mathrm{Fixed}\,\Omega_{b}/\Omega_{cdm}$",
            r"$\mathrm{BBN\,Prior}$",
            r"$\mathrm{Old\,Chain\,BBN\,Prior}$",
        ]

    truths = {
        r"$A_{s}\times 10^{9}$": np.exp(float(pardict["ln10^{10}A_s"])) / 1.0e1,
        r"$h$": float(pardict["h"]),
        r"$\Omega_{m}$": Om_fid,
        r"$\omega_{cdm}$": float(pardict["omega_cdm"]),
        r"$\omega_{b}$": float(pardict["omega_b"]),
    }

    # Output name for the figure

    c = ChainConsumer()

    bestfits = []
    for chaini, chainfile in enumerate(chainfiles):

        burntin, bestfit, like = read_chain_backend(chainfile)
        burntin[:, 0] = np.exp(burntin[:, 0]) / 1.0e1
        omega_b = (
            float(pardict["omega_b"]) / float(pardict["omega_cdm"]) * burntin[:, 2] if chaini == 0 else burntin[:, 3]
        )
        Omega_m = (burntin[:, 2] + omega_b + (0.06 / 93.14)) / burntin[:, 1] ** 2
        paramnames = [r"$A_{s}\times 10^{9}$", r"$h$", r"$\omega_{cdm}$", r"$\Omega_{m}$"]
        c.add_chain(
            np.hstack([burntin[:, :3], Omega_m[:, None]]), parameters=paramnames, name=names[chaini], posterior=like
        )
        bestfits.append(bestfit)
        print(np.amax(like))

    print(bestfits)
    extents = [(1.95, 2.45), (0.66, 0.698), (0.108, 0.132), (0.29, 0.33)]
    # extents = [(1.95, 2.45), (0.66, 0.698), (0.108, 0.132), (0.0205, 0.024), (0.29, 0.33)]
    c.configure(bar_shade=True)
    fig = c.plotter.plot(filename=figfile[0], truth=truths, display=False, extents=extents)
    fig = c.plotter.plot_summary(filename=figfile[1], truth=truths, display=False, extents=extents)
    print(c.analysis.get_summary())

    # Get the bestfit bird model
    if False:

        import matplotlib.pyplot as plt

        fig = plt.figure()
        params = bestfits[2]
        fittingdata = FittingData(pardict, shot_noise=float(pardict["shot_noise"]))

        # Set up the BirdModel
        birdmodel = BirdModel(pardict)
        birdmodel_direct = BirdModel(pardict, direct=True)
        print(birdmodel.valueref)
        params = birdmodel.valueref

        # Plotting (for checking/debugging, should turn off for production runs)
        plt = create_plot(pardict, fittingdata)

        if birdmodel.pardict["do_marg"]:
            b2 = (params[-2] + params[-1]) / np.sqrt(2.0)
            b4 = (params[-2] - params[-1]) / np.sqrt(2.0)
            bs = [params[-3], b2, 0.0, b4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        else:
            b2 = (params[-10] + params[-8]) / np.sqrt(2.0)
            b4 = (params[-10] - params[-8]) / np.sqrt(2.0)
            bs = [
                params[-11],
                b2,
                params[-9],
                b4,
                params[-7],
                params[-6],
                params[-5],
                params[-4] * fittingdata.data["shot_noise"],
                params[-3] * fittingdata.data["shot_noise"],
                params[-2] * fittingdata.data["shot_noise"],
                params[-1],
            ]

        ln10As, h, omega_cdm = params[:3]
        omega_b = birdmodel.valueref[3] / birdmodel.valueref[2] * omega_cdm
        print(birdmodel.fN)

        Plin, Ploop = birdmodel.compute_pk([ln10As, h, omega_cdm, omega_b])
        P_model, P_model_interp = birdmodel.compute_model(bs, Plin, Ploop, fittingdata.data["x_data"])
        Pi = birdmodel.get_Pi_for_marg(Ploop, bs[0], fittingdata.data["shot_noise"], fittingdata.data["x_data"])
        chi_squared = birdmodel.compute_chi2(P_model_interp, Pi, fittingdata.data)

        if birdmodel.pardict["do_marg"]:
            bs_analytic = birdmodel.compute_bestfit_analytic(Pi, fittingdata.data, P_model_interp)
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
                bs_analytic[4] * fittingdata.data["shot_noise"],
                bs_analytic[5] * fittingdata.data["shot_noise"],
                bs_analytic[6] * fittingdata.data["shot_noise"],
                bs_analytic[7],
            ]
            print(bs)
            P_model, P_model_interp = birdmodel.compute_model(bs, Plin, Ploop, fittingdata.data["x_data"])
            chi_squared = birdmodel.compute_chi2(P_model_interp, Pi, fittingdata.data)

        bs = np.zeros(11)
        bs = [1.347, 1.078, 0.315, -0.185, -0.741, -1.989, -1.106, 539, 170, -599, 0.909]

        components = birdmodel.get_components([ln10As, h, omega_cdm, omega_b], bs)
        print(components[0][0] + components[1][0])

        plin, ploop = birdmodel.compute_pk([ln10As, h, omega_cdm, omega_b])
        # plin, ploop = birdmodel_direct.compute_model_direct([ln10As, h, omega_cdm, omega_b])

        ploop0, ploop2, ploop4 = ploop

        update_plot_components(pardict, birdmodel.kin, components, plt, keep=True, comp_list=[True, True, True, True])

        np.savetxt(
            "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_Stage2/pkmodel_UNIT_cosmo_P0_bestfit.dat",
            np.c_[
                birdmodel.kin,
                components[0][0],
                components[1][0],
                components[2][0],
                ploop0[0, :],
                ploop0[1, :],
                ploop0[2, :],
                ploop0[3, :],
                ploop0[4, :],
                ploop0[5, :],
                ploop0[6, :],
                ploop0[7, :],
                ploop0[8, :],
                ploop0[9, :],
                ploop0[10, :],
                ploop0[11, :],
                # 2.0 * ploop0[12, :] / birdmodel.k_nl ** 2,
                # 2.0 * ploop0[13, :] / birdmodel.k_m ** 2,
                # 2.0 * ploop0[14, :] / birdmodel.k_m ** 2,
                # 2.0 * ploop0[15, :] / birdmodel.k_nl ** 2,
                # 2.0 * ploop0[16, :] / birdmodel.k_m ** 2,
                # 2.0 * ploop0[17, :] / birdmodel.k_m ** 2,
                # 2.0 * ploop0[18, :] / birdmodel.k_m ** 4,
                ploop0[12, :],
                ploop0[13, :],
                ploop0[14, :],
                ploop0[15, :],
                ploop0[16, :],
                ploop0[17, :],
                ploop0[18, :],
            ],
            header="k    P_lin     P_loop     P_ct     1      b1    b2    b3    b4    b1 * b1    b1 * b2    b1 * b3    b1 * b4    b2 * b2    b2 * b4    b4 * b4    b1 * cct   b1 * cr1   b1 * cr2   cct    cr1    cr2    b1*b1*bnlo",
        )
        np.savetxt(
            "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_Stage2/pkmodel_UNIT_cosmo_P2_bestfit.dat",
            np.c_[
                birdmodel.kin,
                components[0][1],
                components[1][1],
                components[2][1],
                ploop2[0, :],
                ploop2[1, :],
                ploop2[2, :],
                ploop2[3, :],
                ploop2[4, :],
                ploop2[5, :],
                ploop2[6, :],
                ploop2[7, :],
                ploop2[8, :],
                ploop2[9, :],
                ploop2[10, :],
                ploop2[11, :],
                # 2.0 * ploop2[12, :] / birdmodel.k_nl ** 2,
                # 2.0 * ploop2[13, :] / birdmodel.k_m ** 2,
                # 2.0 * ploop2[14, :] / birdmodel.k_m ** 2,
                # 2.0 * ploop2[15, :] / birdmodel.k_nl ** 2,
                # 2.0 * ploop2[16, :] / birdmodel.k_m ** 2,
                # 2.0 * ploop2[17, :] / birdmodel.k_m ** 2,
                # 2.0 * ploop2[18, :] / birdmodel.k_m ** 4,
                ploop2[12, :],
                ploop2[13, :],
                ploop2[14, :],
                ploop2[15, :],
                ploop2[16, :],
                ploop2[17, :],
                ploop2[18, :],
            ],
            header="k    P_lin     P_loop     P_ct     1      b1    b2    b3    b4    b1 * b1    b1 * b2    b1 * b3    b1 * b4    b2 * b2    b2 * b4    b4 * b4    b1 * cct   b1 * cr1   b1 * cr2   cct    cr1    cr2    b1*b1*bnlo",
        )
        np.savetxt(
            "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_Stage2/pkmodel_UNIT_cosmo_P4_bestfit.dat",
            np.c_[
                birdmodel.kin,
                components[0][2],
                components[1][2],
                components[2][2],
                ploop4[0, :],
                ploop4[1, :],
                ploop4[2, :],
                ploop4[3, :],
                ploop4[4, :],
                ploop4[5, :],
                ploop4[6, :],
                ploop4[7, :],
                ploop4[8, :],
                ploop4[9, :],
                ploop4[10, :],
                ploop4[11, :],
                # 2.0 * ploop4[12, :] / birdmodel.k_nl ** 2,
                # 2.0 * ploop4[13, :] / birdmodel.k_m ** 2,
                # 2.0 * ploop4[14, :] / birdmodel.k_m ** 2,
                # 2.0 * ploop4[15, :] / birdmodel.k_nl ** 2,
                # 2.0 * ploop4[16, :] / birdmodel.k_m ** 2,
                # 2.0 * ploop4[17, :] / birdmodel.k_m ** 2,
                # 2.0 * ploop4[18, :] / birdmodel.k_m ** 4,
                ploop4[12, :],
                ploop4[13, :],
                ploop4[14, :],
                ploop4[15, :],
                ploop4[16, :],
                ploop4[17, :],
                ploop4[18, :],
            ],
            header="k    P_lin     P_loop     P_ct     1      b1    b2    b3    b4    b1 * b1    b1 * b2    b1 * b3    b1 * b4    b2 * b2    b2 * b4    b4 * b4    b1 * cct   b1 * cr1   b1 * cr2   cct    cr1    cr2    b1*b1*bnlo",
        )
