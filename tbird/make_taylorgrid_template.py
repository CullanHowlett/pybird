import numpy as np
import os
import sys
import copy
from configobj import ConfigObj

sys.path.append("../")
import pandas as pd
from pybird_dev import pybird
from tbird.Grid import grid_properties_template, run_camb, run_class

if __name__ == "__main__":

    # Read in the config file, job number and total number of jobs
    configfile = sys.argv[1]
    job_no = int(sys.argv[2])
    njobs = int(sys.argv[3])
    pardict = ConfigObj(configfile)

    # Get some cosmological values at the grid centre
    if pardict["code"] == "CAMB":
        kin, Pin, Om, Da_fid, Hz_fid, fN_fid, sigma8_fid, sigma12, r_d = run_camb(pardict)
    else:
        kin, Pin, Om, Da_fid, Hz_fid, fN_fid, sigma8_fid, sigma12, r_d = run_class(pardict)

    datapk = np.array(
        pd.read_csv(
            "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_Stage2/input_data/Pk_Planck15_Table4.txt",
            delim_whitespace=True,
            header=None,
        )
    )
    datapk[:, 1] *= sigma8_fid ** 2 / 0.8147 ** 2
    kin, Pin = datapk[:, 0], datapk[:, 1]

    # Compute the values of the growth rate that this job will do
    valueref, delta, flattenedgrid, _ = grid_properties_template(pardict, fN_fid, sigma8_fid)
    lenrun = int(len(flattenedgrid) / njobs)
    start = job_no * lenrun
    final = min((job_no + 1) * lenrun, len(flattenedgrid))
    arrayred = flattenedgrid[start:final]
    print(start, final, arrayred)

    # Set up pybird
    Nl = 3
    z_pk = float(pardict["z_pk"])
    correlator = pybird.Correlator()
    correlatorcf = pybird.Correlator()

    correlator.set(
        {
            "output": "bPk",
            "multipole": Nl,
            "z": z_pk,
            "optiresum": False,
            "with_bias": False,
            "with_exact_time": True,
            "kmax": 0.5,
            "with_AP": True,
            "DA_AP": Da_fid,
            "H_AP": Hz_fid,
        }
    )
    correlatorcf.set(
        {
            "output": "bCf",
            "multipole": Nl,
            "z": z_pk,
            "optiresum": True,
            "with_bias": False,
            "with_exact_time": True,
            "kmax": 0.5,
            "with_AP": True,
            "DA_AP": Da_fid,
            "H_AP": Hz_fid,
        }
    )

    # Now loop over all grid cells and compute the EFT model for different values of the growth rate
    allPlin = []
    allPloop = []
    allClin = []
    allCloop = []
    allParams = []
    for i, theta in enumerate(arrayred):
        parameters = copy.deepcopy(pardict)
        truetheta = valueref + theta * delta
        idx = i
        print("i on tot", i, len(arrayred))

        Da = Da_fid * truetheta[0]
        Hz = Hz_fid / truetheta[1]
        sigma8_scale = truetheta[3] / sigma8_fid
        Pin_scaled = Pin * sigma8_scale ** 2

        # Get non-linear power spectrum from pybird
        correlator.compute(
            {"k11": kin, "P11": Pin_scaled, "z": z_pk, "Omega0_m": Om, "f": truetheta[2], "DA": Da, "H": Hz}
        )
        correlatorcf.compute(
            {"k11": kin, "P11": Pin_scaled, "z": z_pk, "Omega0_m": Om, "f": truetheta[2], "DA": Da, "H": Hz}
        )

        Plin, Ploop = correlator.bird.formatTaylorPs()
        Clin, Cloop = correlatorcf.bird.formatTaylorCf()
        idxcol = np.full([Plin.shape[0], 1], idx)
        allPlin.append(np.hstack([Plin, idxcol]))
        allPloop.append(np.hstack([Ploop, idxcol]))
        idxcol = np.full([Clin.shape[0], 1], idx)
        allClin.append(np.hstack([Clin, idxcol]))
        allCloop.append(np.hstack([Cloop, idxcol]))
        if (i == 0) or ((i + 1) % 10 == 0):
            print("theta check: ", arrayred[idx], theta, truetheta)
        np.save(os.path.join(pardict["outpk"], "Plin_run%s_template.npy" % (str(job_no))), np.array(allPlin))
        np.save(os.path.join(pardict["outpk"], "Ploop_run%s_template.npy" % (str(job_no))), np.array(allPloop))
        np.save(os.path.join(pardict["outpk"], "Clin_run%s_template.npy" % (str(job_no))), np.array(allClin))
        np.save(os.path.join(pardict["outpk"], "Cloop_run%s_template.npy" % (str(job_no))), np.array(allCloop))
