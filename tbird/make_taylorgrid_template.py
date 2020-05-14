import numpy as np
import os
import sys
import copy
from configobj import ConfigObj

sys.path.append("../")
from pybird import pybird
from tbird.Grid import grid_properties_template, run_camb

if __name__ == "__main__":

    # Read in the config file, job number and total number of jobs
    configfile = sys.argv[1]
    job_no = int(sys.argv[2])
    njobs = int(sys.argv[3])
    pardict = ConfigObj(configfile)

    # Get some cosmological values at the grid centre
    kin, Pin, Da_fid, Hz_fid, fN, sigma8, sigma12, r_d = run_camb(pardict)

    # Compute the values of the growth rate that this job will do
    valueref, delta, flattenedgrid, _ = grid_properties_template(pardict, fN)
    lenrun = int(len(flattenedgrid) / njobs)
    start = job_no * lenrun
    final = min((job_no + 1) * lenrun, len(flattenedgrid))
    arrayred = flattenedgrid[start:final]
    print(start, final, arrayred)

    # Set up pybird
    Nl = 3
    common = pybird.Common(Nl=Nl, kmax=0.5, optiresum=False)
    commoncf = pybird.Common(Nl=Nl, smax=1000)
    nonlinear = pybird.NonLinear(load=False, save=False, co=common)
    nonlinearcf = pybird.NonLinear(load=False, save=False, co=commoncf)
    resum = pybird.Resum(co=common)
    resumcf = pybird.Resum(co=commoncf)

    # Set up the window function and projection effects. No window at the moment for the UNIT sims,
    # so we'll create an identity matrix for this. I'm also assuming that the fiducial cosmology
    # used to make the measurements is the same as Grid centre
    kout, nkout = common.k, len(common.k)
    projection = pybird.Projection(kout, DA=Da_fid, H=Hz_fid, window_fourier_name=None, co=common)
    projection.p = kout
    window = np.zeros((2, 2, nkout, nkout))
    window[0, 0, :, :] = np.eye(nkout)
    window[1, 1, :, :] = np.eye(nkout)
    projection.Waldk = window

    # Set up the window function and projection effects. No window at the moment for the UNIT sims,
    # so we'll create an identity matrix for this. I'm also assuming that the fiducial cosmology
    # used to make the measurements is the same as Grid centre
    sout, nsout = np.linspace(1.0, 200.0, 200), 200
    kout, nkout = common.k, len(common.k)
    projection = pybird.Projection(kout, Da_fid, Hz_fid, co=common)
    projection.p = kout
    window = np.zeros((Nl, Nl, nkout, nkout))
    for i in range(Nl):
        window[i, i, :, :] = np.eye(nkout)
    projection.Waldk = window
    projectioncf = pybird.Projection(sout, Da_fid, Hz_fid, co=commoncf, cf=True)

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

        # Get non-linear power spectrum from pybird
        bird = pybird.Bird(kin, Pin, truetheta[2], DA=Da, H=Hz, z=pardict["z_pk"], which="all", co=common)
        nonlinear.PsCf(bird)
        bird.setPsCfl()
        resum.PsCf(bird)
        projection.AP(bird)
        projection.Window(bird)

        crow = pybird.Bird(kin, Pin, truetheta[2], DA=Da, H=Hz, z=pardict["z_pk"], which="all", co=commoncf)
        nonlinearcf.PsCf(crow)
        crow.setPsCfl()
        resumcf.PsCf(crow)
        projectioncf.AP(crow)
        projectioncf.kdata(crow)

        Plin, Ploop = bird.formatTaylorPs(kdata=kout)
        Clin, Cloop = crow.formatTaylorCf(sdata=sout)
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
