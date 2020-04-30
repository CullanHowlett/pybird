import numpy as np
import os
import sys
import copy
from configobj import ConfigObj

sys.path.append("../")
from pybird import pybird
from tbird.Grid import grid_properties, run_camb

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

    # Set up pybird
    Nl = 2
    common = pybird.Common(Nl=Nl, kmax=0.5, optiresum=False)
    nonlinear = pybird.NonLinear(load=False, save=False, co=common)
    resum = pybird.Resum(co=common)

    # Get some cosmological values at the grid centre
    kin, Plin, Da, Hz, fN, sigma8, sigma12, r_d = run_camb(pardict)

    # Set up the window function and projection effects. No window at the moment for the UNIT sims,
    # so we'll create an identity matrix for this. I'm also assuming that the fiducial cosmology
    # used to make the measurements is the same as Grid centre
    sout, nsout = np.linspace(1.0, 200.0, 200), 200
    kout, nkout = common.k, len(common.k)
    projection = pybird.Projection(kout, DA=Da, H=Hz, co=common)
    projection.p = kout
    window = np.zeros((Nl, Nl, nkout, nkout))
    for i in range(Nl):
        window[i, i, :, :] = np.eye(nkout)
    projection.Waldk = window
    projectioncf = pybird.Projection(sout, DA=Da, H=Hz, co=common, cf=True)
    window = np.zeros((Nl, Nl, len(common.s)))
    for i in range(Nl):
        window[i, i, :] = np.ones(len(common.s))
    projectioncf.Qal = window

    # Now loop over all grid cells and compute the EFT model
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

        for k, var in enumerate(pardict["freepar"]):
            parameters[var] = truetheta[k]
        kin, Plin, Da, Hz, fN, sigma8, sigma12, r_d = run_camb(parameters)

        # Get non-linear power spectrum from pybird
        bird = pybird.Bird(kin, Plin, fN, DA=Da, H=Hz, z=pardict["z_pk"], which="all", co=common)
        nonlinear.PsCf(bird)
        bird.setPsCfl()
        resum.PsCf(bird)

        projection.AP(bird)
        projectioncf.AP(bird)
        projection.Window(bird)
        projectioncf.Window(bird)
        projectioncf.kdata(bird)

        Params = np.array([Da, Hz, fN, sigma8, sigma12, r_d])
        Plin, Ploop = bird.formatTaylorPs(kdata=kout)
        Clin, Cloop = bird.formatTaylorCf(sdata=sout)
        idxcol = np.full([Plin.shape[0], 1], idx)
        allPlin.append(np.hstack([Plin, idxcol]))
        allPloop.append(np.hstack([Ploop, idxcol]))
        idxcol = np.full([Clin.shape[0], 1], idx)
        allClin.append(np.hstack([Clin, idxcol]))
        allCloop.append(np.hstack([Cloop, idxcol]))
        allParams.append(np.hstack([Params, [idx]]))
        if (i == 0) or ((i + 1) % 10 == 0):
            print("theta check: ", arrayred[idx], theta, truetheta)
        np.save(os.path.join(pardict["outpk"], "Plin_run%s.npy" % (str(job_no))), np.array(allPlin))
        np.save(os.path.join(pardict["outpk"], "Ploop_run%s.npy" % (str(job_no))), np.array(allPloop))
        np.save(os.path.join(pardict["outpk"], "Clin_run%s.npy" % (str(job_no))), np.array(allClin))
        np.save(os.path.join(pardict["outpk"], "Cloop_run%s.npy" % (str(job_no))), np.array(allCloop))
        np.save(os.path.join(pardict["outpk"], "Params_run%s.npy" % (str(job_no))), np.array(allParams))
