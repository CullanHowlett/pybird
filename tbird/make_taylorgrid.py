import numpy as np
import os
import sys
import copy
from configobj import ConfigObj

sys.path.append("../")
from pybird_dev import pybird
from tbird.Grid import grid_properties, run_camb, run_class

if __name__ == "__main__":

    # Read in the config file, job number and total number of jobs
    configfile = sys.argv[1]
    job_no = int(sys.argv[2])
    njobs = int(sys.argv[3])
    pardict = ConfigObj(configfile)

    # Compute some stuff for the grid based on the config file
    valueref, delta, flattenedgrid, _ = grid_properties(pardict)
    lenrun = int(len(flattenedgrid) / njobs)
    start = job_no * lenrun
    final = min((job_no + 1) * lenrun, len(flattenedgrid))
    arrayred = flattenedgrid[start:final]

    # Get some cosmological values at the grid centre
    if pardict["code"] == "CAMB":
        kin, Pin, Om, Da_fid, Hz_fid, fN_fid, sigma8_fid, sigma12, r_d = run_camb(pardict)
    else:
        kin, Pin, Om, Da_fid, Hz_fid, fN_fid, sigma8_fid, sigma12, r_d = run_class(pardict)

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
            "with_nlo_bias": True,
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
            "with_nlo_bias": True,
            "with_exact_time": True,
            "with_AP": True,
            "DA_AP": Da_fid,
            "H_AP": Hz_fid,
        }
    )

    # Now loop over all grid cells and compute the EFT model
    allPlin = []
    allPloop = []
    allClin = []
    allCloop = []
    allParams = []
    allPin = []
    for i, theta in enumerate(arrayred):
        parameters = copy.deepcopy(pardict)
        truetheta = valueref + theta * delta
        idx = i
        print("i on tot", i, len(arrayred))

        for k, var in enumerate(pardict["freepar"]):
            parameters[var] = truetheta[k]
        if parameters["code"] == "CAMB":
            kin, Pin, Om, Da, Hz, fN, sigma8, sigma8_0, sigma12, r_d = run_camb(parameters)
        else:
            kin, Pin, Om, Da, Hz, fN, sigma8, sigma8_0, sigma12, r_d = run_class(parameters)

        # Get non-linear power spectrum from pybird
        correlator.compute({"k11": kin, "P11": Pin, "z": z_pk, "Omega0_m": Om, "f": fN, "DA": Da, "H": Hz})
        correlatorcf.compute({"k11": kin, "P11": Pin, "z": z_pk, "Omega0_m": Om, "f": fN, "DA": Da, "H": Hz})

        Params = np.array([Om, Da, Hz, fN, sigma8, sigma8_0, sigma12, r_d])
        Plin, Ploop = correlator.bird.formatTaylorPs()
        Clin, Cloop = correlatorcf.bird.formatTaylorCf()
        Pin = np.c_[kin, Pin]
        idxcol = np.full([Pin.shape[0], 1], idx)
        allPin.append(np.hstack([Pin, idxcol]))
        idxcol = np.full([Plin.shape[0], 1], idx)
        allPlin.append(np.hstack([Plin, idxcol]))
        allPloop.append(np.hstack([Ploop, idxcol]))
        idxcol = np.full([Clin.shape[0], 1], idx)
        allClin.append(np.hstack([Clin, idxcol]))
        allCloop.append(np.hstack([Cloop, idxcol]))
        allParams.append(np.hstack([Params, [idx]]))
        if (i == 0) or ((i + 1) % 10 == 0):
            print("theta check: ", arrayred[idx], theta, truetheta)
        if parameters["code"] == "CAMB":
            np.save(os.path.join(pardict["outpk"], "CAMB_run%s.npy" % (str(job_no))), np.array(allPin))
        else:
            np.save(os.path.join(pardict["outpk"], "CLASS_run%s.npy" % (str(job_no))), np.array(allPin))
        np.save(os.path.join(pardict["outpk"], "Plin_run%s.npy" % (str(job_no))), np.array(allPlin))
        np.save(os.path.join(pardict["outpk"], "Ploop_run%s.npy" % (str(job_no))), np.array(allPloop))
        np.save(os.path.join(pardict["outpk"], "Clin_run%s.npy" % (str(job_no))), np.array(allClin))
        np.save(os.path.join(pardict["outpk"], "Cloop_run%s.npy" % (str(job_no))), np.array(allCloop))
        np.save(os.path.join(pardict["outpk"], "Params_run%s.npy" % (str(job_no))), np.array(allParams))
