import sys
import numpy as np
import pandas as pd
from configobj import ConfigObj
from chainconsumer import ChainConsumer

sys.path.append("../")
from tbird.Grid import run_camb, run_class
from fitting_codes.fitting_utils import read_chain_backend, get_Planck


if __name__ == "__main__":

    # First read in the config file and compute Da_fid, Hz_fid and sigma8_fid
    configfile = sys.argv[1]
    pardict = ConfigObj(configfile)
    if pardict["code"] == "CAMB":
        _, _, Om_fid, Da_fid, Hz_fid, fN_fid, sigma8_fid, sigma8_0_fid, sigma12_fid, r_d_fid = run_camb(pardict)
        omega_nu = float(pardict["Sum_mnu"]) / 93.14
    else:
        _, _, Om_fid, Da_fid, Hz_fid, fN_fid, sigma8_fid, sigma8_0_fid, sigma12_fid, r_d_fid = run_class(pardict)
        omega_nu = float(pardict["m_ncdm"]) / 93.14

    chainfiles = [
        "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_Stage2/output_files/New_chain_UNIT_HODsnap97_ELGv1_3Gpc_FixAmp_pk_0.20hex0.20_3order_hex_marg_template_planck.dat",
        "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_Stage2/output_files/New_chain_UNIT_HODsnap97_ELGv1_3Gpc_FixAmp_pk_0.20hex0.20_3order_hex_marg_hybrid_planck.dat",
        "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_Stage2/output_files/New_chain_UNIT_HODsnap97_ELGv1_3Gpc_FixAmp_pk_0.20hex0.20_3order_hex_marg_planck_converted.dat",
    ]
    templates = [False, False, False]
    figfile = [
        "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_Stage2/New_chain_UNIT_HODsnap97_ELGv1_3Gpc_FixAmp_pk_0.20hex0.20_3order_hex_marg_hybrid_planck_derived.pdf"
    ]
    names = [
        r"$\mathrm{TEMPLATE}+Planck$",
        r"$\mathrm{HYBRID}+Planck$",
        r"$\mathrm{COSMO}+Planck$",
    ]

    truths = {
        r"$\mathrm{ln}10^{10}(A_{s})$": float(pardict["ln10^{10}A_s"]),
        r"$h$": float(pardict["h"]),
        r"$\omega_{cdm}$": float(pardict["omega_cdm"]),
        r"$\omega_{b}$": float(pardict["omega_b"]),
        r"$\Omega_{m}h^{2}$": Om_fid * float(pardict["h"]) ** 2,
        r"$\Omega_{m}$": Om_fid,
        r"$H_{0}$": 100.0 * float(pardict["h"]),
        r"$\sigma_{8}$": sigma8_0_fid,
    }

    c = ChainConsumer()

    bestfits = []
    paramnames = [
        r"$\Omega_{m}$",
        r"$H_{0}$",
        r"$\sigma_{8}$",
        r"$\Omega_{m}h^{2}$",
    ]
    # paramnames = [
    #    r"$\mathrm{ln}10^{10}(A_{s})$",
    #    r"$h$",
    #    r"$\omega_{cdm}$",
    #    r"$\omega_{b}$",
    # ]
    for chaini, (chainfile, template) in enumerate(zip(chainfiles, templates)):

        print(chainfile)
        if template:
            burntin, bestfit, like = read_chain_backend(chainfile)
            omega_b = float(pardict["omega_b"]) / float(pardict["omega_cdm"]) * burntin[:, 1]
            omega_m = omega_b + burntin[:, 1] + omega_nu
            Omega_m = omega_m / burntin[:, 0] ** 2
            H0 = 100.0 * burntin[:, 0]
            burntin = np.hstack((burntin, omega_m[:, None], Omega_m[:, None], H0[:, None]))
            c.add_chain(burntin[:, [4, 5, 2, 3]], parameters=paramnames, name=names[chaini], posterior=like)
            # c.add_chain(burntin[:, [0, 1, 2, 3]], parameters=paramnames, name=names[chaini], posterior=like)
        else:
            burntin = np.array(pd.read_csv(chainfile, delim_whitespace=True, header=None))
            like = burntin[:, -1]
            bestfit = burntin[np.argmax(burntin[:, -1]), :-1]
            omega_m = burntin[:, 2] + burntin[:, 3] + omega_nu
            burntin = np.hstack((burntin, omega_m[:, None]))
            c.add_chain(burntin[:, [6, 1, 11, 14]], parameters=paramnames, name=names[chaini], posterior=like)
            # c.add_chain(burntin[:, [0, 1, 2, 3]], parameters=paramnames, name=names[chaini], posterior=like)
        bestfits.append(bestfit)

    """Planck_file = "/Volumes/Work/UQ/CAMB/COM_CosmoParams_fullGrid_R3.01/base/plikHM_TTTEEE_lowl_lowE_lensing/base_plikHM_TTTEEE_lowl_lowE_lensing"
    weights, chain = get_Planck(
        Planck_file,
        4,
        usecols=[6, 29, 3, 2],
        raw=True,
    )
    c.add_chain(
        chain,
        parameters=paramnames,
        weights=weights,
        name=r"$Planck$",
        color="k",
        linestyle="--",
        shade=False,
    )"""

    print(bestfits)
    fig = c.plotter.plot(figsize="column", filename=figfile, truth=truths)
    print(c.analysis.get_summary())
