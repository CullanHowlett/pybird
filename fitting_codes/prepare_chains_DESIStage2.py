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
    if pardict["code"] == "CAMB":
        _, _, Om_fid, Da_fid, Hz_fid, fN_fid, sigma8_fid, sigma12_fid, r_d_fid = run_camb(pardict)
    else:
        _, _, Om_fid, Da_fid, Hz_fid, fN_fid, sigma8_fid, sigma12_fid, r_d_fid = run_class(pardict)

    # Set the chainfiles and names for each chain
    chainfiles = [
        "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_Stage2/output_files/chain_UNIT_HODsnap97_ELGv1_3Gpc_Std_pk_0.10hex0.10_2order_hex_marg_template.hdf5",
        "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_Stage2/output_files/chain_UNIT_HODsnap97_ELGv1_3Gpc_Std_pk_0.15hex0.15_2order_hex_marg_template.hdf5",
        "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_Stage2/output_files/chain_UNIT_HODsnap97_ELGv1_3Gpc_Std_pk_0.20hex0.15_2order_hex_marg_template.hdf5",
        "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_Stage2/output_files/chain_UNIT_HODsnap97_ELGv1_3Gpc_Std_pk_0.20hex0.20_2order_hex_marg_template.hdf5",
        "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_Stage2/output_files/chain_UNIT_HODsnap97_ELGv1_3Gpc_Std_pk_0.25hex0.15_2order_hex_marg_template.hdf5",
        "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_Stage2/output_files/chain_UNIT_HODsnap97_ELGv1_3Gpc_Std_pk_0.25hex0.20_2order_hex_marg_template.hdf5",
        "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_Stage2/output_files/chain_UNIT_HODsnap97_ELGv1_3Gpc_Std_pk_0.25hex0.25_2order_hex_marg_template.hdf5",
        "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_Stage2/output_files/chain_UNIT_HODsnap97_ELGv1_3Gpc_Std_pk_0.30hex0.15_2order_hex_marg_template.hdf5",
        "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_Stage2/output_files/chain_UNIT_HODsnap97_ELGv1_3Gpc_Std_pk_0.30hex0.20_2order_hex_marg_template.hdf5",
        "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_Stage2/output_files/chain_UNIT_HODsnap97_ELGv1_3Gpc_Std_pk_0.30hex0.25_2order_hex_marg_template.hdf5",
        "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_Stage2/output_files/chain_UNIT_HODsnap97_ELGv1_3Gpc_Std_pk_0.30hex0.30_2order_hex_marg_template.hdf5",
        "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_Stage2/output_files/chain_UNIT_HODsnap97_ELGv1_3Gpc_Std_pk_0.40hex0.15_2order_hex_marg_template.hdf5",
        "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_Stage2/output_files/chain_UNIT_HODsnap97_ELGv1_3Gpc_Std_pk_0.40hex0.20_2order_hex_marg_template.hdf5",
        "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_Stage2/output_files/chain_UNIT_HODsnap97_ELGv1_3Gpc_Std_pk_0.40hex0.25_2order_hex_marg_template.hdf5",
        # "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_Stage2/output_files/chain_UNIT_HODsnap97_ELGv1_3Gpc_Std_pk_0.40hex0.40_2order_hex_marg_template.hdf5",
    ]

    root_names = [
        "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_Stage2/prepared/Queensland_CosmoUNIT_3Gpc_0.10hex0.10_covStd_fitTemplate",
        "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_Stage2/prepared/Queensland_CosmoUNIT_3Gpc_0.15hex0.15_covStd_fitTemplate",
        "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_Stage2/prepared/Queensland_CosmoUNIT_3Gpc_0.20hex0.15_covStd_fitTemplate",
        "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_Stage2/prepared/Queensland_CosmoUNIT_3Gpc_0.20hex0.20_covStd_fitTemplate",
        "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_Stage2/prepared/Queensland_CosmoUNIT_3Gpc_0.25hex0.15_covStd_fitTemplate",
        "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_Stage2/prepared/Queensland_CosmoUNIT_3Gpc_0.25hex0.20_covStd_fitTemplate",
        "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_Stage2/prepared/Queensland_CosmoUNIT_3Gpc_0.25hex0.25_covStd_fitTemplate",
        "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_Stage2/prepared/Queensland_CosmoUNIT_3Gpc_0.30hex0.15_covStd_fitTemplate",
        "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_Stage2/prepared/Queensland_CosmoUNIT_3Gpc_0.30hex0.20_covStd_fitTemplate",
        "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_Stage2/prepared/Queensland_CosmoUNIT_3Gpc_0.30hex0.25_covStd_fitTemplate",
        "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_Stage2/prepared/Queensland_CosmoUNIT_3Gpc_0.30hex0.30_covStd_fitTemplate",
        "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_Stage2/prepared/Queensland_CosmoUNIT_3Gpc_0.40hex0.15_covStd_fitTemplate",
        "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_Stage2/prepared/Queensland_CosmoUNIT_3Gpc_0.40hex0.20_covStd_fitTemplate",
        "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_Stage2/prepared/Queensland_CosmoUNIT_3Gpc_0.40hex0.25_covStd_fitTemplate",
        # "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_Stage2/prepared/Queensland_CosmoUNIT_3Gpc_0.40hex0.40_covStd_fitTemplate",
    ]

    paramnames = [r"$\alpha_{\perp}$", r"$\alpha_{||}$", r"$f\sigma_{8}$", r"$b_{1}\sigma_{8}$"]

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
