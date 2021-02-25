import sys
import numpy as np
import pandas as pd
from configobj import ConfigObj
from chainconsumer import ChainConsumer

sys.path.append("../")
from tbird.Grid import run_camb, run_class
from fitting_codes.fitting_utils import read_chain_backend, format_pardict


if __name__ == "__main__":

    # First read in the config file and compute Da_fid, Hz_fid and sigma8_fid
    configfile = sys.argv[1]
    pardict = ConfigObj(configfile)
    pardict = format_pardict(pardict)
    template = sys.argv[2]

    if pardict["code"] == "CAMB":
        _, _, Om_fid, Da_fid, Hz_fid, fN_fid, sigma8_fid, sigma8_0_fid, sigma12_fid, r_d_fid = run_camb(pardict)
    else:
        _, _, Om_fid, Da_fid, Hz_fid, fN_fid, sigma8_fid, sigma8_0_fid, sigma12_fid, r_d_fid = run_class(pardict)

    # Set the chainfiles and names for each chain
    chainfiles = [
        "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_Stage2/output_files/New_chain_UNIT_HODsnap97_ELGv1_3Gpc_FixAmp_pk_0.20hex0.20_3order_hex_marg_template.hdf5",
    ]

    root_names = [
        "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_Stage2/prepared/New_chain_UNIT_HODsnap97_ELGv1_3Gpc_FixAmp_pk_0.20hex0.20_3order_hex_marg_template",
    ]

    if template:
        paramnames = [r"$\alpha_{\perp}$", r"$\alpha_{||}$", r"$f\sigma_{8}$", r"$b_{1}\sigma_{8}$"]
        print(paramnames)

        for chaini, (chainfile, root_name) in enumerate(zip(chainfiles, root_names)):
            c = ChainConsumer()
            burntin, bestfit, like = read_chain_backend(chainfile)
            burntin[:, 3] *= sigma8_fid
            c.add_chain(burntin[:, :4], parameters=paramnames, posterior=like)
            marginalised = c.analysis.get_summary()
            alphaperp = [
                marginalised[r"$\alpha_{\perp}$"][1],
                marginalised[r"$\alpha_{\perp}$"][1] - marginalised[r"$\alpha_{\perp}$"][0],
                marginalised[r"$\alpha_{\perp}$"][2] - marginalised[r"$\alpha_{\perp}$"][1],
            ]
            alphapar = [
                marginalised[r"$\alpha_{||}$"][1],
                marginalised[r"$\alpha_{||}$"][1] - marginalised[r"$\alpha_{||}$"][0],
                marginalised[r"$\alpha_{||}$"][2] - marginalised[r"$\alpha_{||}$"][1],
            ]
            fsigma8 = [
                marginalised[r"$f\sigma_{8}$"][1],
                marginalised[r"$f\sigma_{8}$"][1] - marginalised[r"$f\sigma_{8}$"][0],
                marginalised[r"$f\sigma_{8}$"][2] - marginalised[r"$f\sigma_{8}$"][1],
            ]
            b1sigma8 = [
                marginalised[r"$b_{1}\sigma_{8}$"][1],
                marginalised[r"$b_{1}\sigma_{8}$"][1] - marginalised[r"$b_{1}\sigma_{8}$"][0],
                marginalised[r"$b_{1}\sigma_{8}$"][2] - marginalised[r"$b_{1}\sigma_{8}$"][1],
            ]
            np.savetxt(root_name + ".txt", burntin[:, :4])
            np.savetxt(root_name + ".paramnames", np.c_[paramnames], fmt="%s")
            np.savetxt(
                root_name + ".marginalised",
                np.c_[alphaperp, alphapar, fsigma8, b1sigma8].T,
                header="MLE   68% Lower Bound   68% Upper Bound",
                fmt="%12.6lf %12.6lf %12.6lf",
            )

    else:
        paramnames = [
            r"$A_{s}\times 10^{9}$",
            r"$h$",
            r"$\omega_{cdm}$",
            r"$\omega_{b}$",
            r"$\Omega_{m}$",
            r"$b_{1}\sigma_{8}$",
        ]
        print(paramnames)

        for chaini, (chainfile, root_name) in enumerate(zip(chainfiles, root_names)):
            c = ChainConsumer()
            burntin, bestfit, like = read_chain_backend(chainfile)
            burntin[:, 0] = np.exp(burntin[:, 0]) / 1.0e1
            omega_b = float(pardict["omega_b"]) / float(pardict["omega_cdm"]) * burntin[:, 2]
            print(omega_b)
            Omega_m = (burntin[:, 2] + omega_b + (0.06 / 93.14)) / burntin[:, 1] ** 2
            new_chain = np.hstack([burntin[:, :4], Omega_m[:, None], burntin[:, 5, None]])
            c.add_chain(new_chain, parameters=paramnames, posterior=like)
            marginalised = c.analysis.get_summary()
            A_s = [
                marginalised[r"$A_{s}\times 10^{9}$"][1],
                marginalised[r"$A_{s}\times 10^{9}$"][1] - marginalised[r"$A_{s}\times 10^{9}$"][0],
                marginalised[r"$A_{s}\times 10^{9}$"][2] - marginalised[r"$A_{s}\times 10^{9}$"][1],
            ]
            h = [
                marginalised[r"$h$"][1],
                marginalised[r"$h$"][1] - marginalised[r"$h$"][0],
                marginalised[r"$h$"][2] - marginalised[r"$h$"][1],
            ]
            omega_cdm = [
                marginalised[r"$\omega_{cdm}$"][1],
                marginalised[r"$\omega_{cdm}$"][1] - marginalised[r"$\omega_{cdm}$"][0],
                marginalised[r"$\omega_{cdm}$"][2] - marginalised[r"$\omega_{cdm}$"][1],
            ]
            omega_b = [
                marginalised[r"$\omega_{b}$"][1],
                marginalised[r"$\omega_{b}$"][1] - marginalised[r"$\omega_{b}$"][0],
                marginalised[r"$\omega_{b}$"][2] - marginalised[r"$\omega_{b}$"][1],
            ]
            Omega_m = [
                marginalised[r"$\Omega_{m}$"][1],
                marginalised[r"$\Omega_{m}$"][1] - marginalised[r"$\Omega_{m}$"][0],
                marginalised[r"$\Omega_{m}$"][2] - marginalised[r"$\Omega_{m}$"][1],
            ]
            b1sigma8 = [
                marginalised[r"$b_{1}\sigma_{8}$"][1],
                marginalised[r"$b_{1}\sigma_{8}$"][1] - marginalised[r"$b_{1}\sigma_{8}$"][0],
                marginalised[r"$b_{1}\sigma_{8}$"][2] - marginalised[r"$b_{1}\sigma_{8}$"][1],
            ]
            np.savetxt(root_name + ".txt", new_chain)
            np.savetxt(root_name + ".paramnames", np.c_[paramnames], fmt="%s")
            np.savetxt(
                root_name + ".marginalised",
                np.c_[A_s, h, omega_cdm, omega_b, Omega_m, b1sigma8].T,
                header="MLE   68% Lower Bound   68% Upper Bound",
                fmt="%12.6lf %12.6lf %12.6lf",
            )
