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

    # Compute the values of the growth rate that this job will do
    fvals = np.linspace(float(pardict["growth_min"]), float(pardict["growth_max"]), int(pardict["ngrowth"]))
    nvals = int(len(fvals) / njobs)
    startval = int(job_no * nvals)
    endval = int((job_no + 1) * nvals)

    # Set up the model
    common = pybird.Common(Nl=2, kmax=5.0, optiresum=False)
    nonlinear = pybird.NonLinear(load=False, save=False, co=common)
    resum = pybird.Resum(co=common)

    # Get some cosmological values at the grid centre
    kin, Plin, Da, Hz, fN, sigma8 = run_camb(pardict)

    # Generate the model components
    bird = pybird.Bird(kin, Plin, fN, DA=Da, H=Hz, z=pardict["z_pk"], which="all", co=common)
    nonlinear.PsCf(bird)
    bird.setPsCfl()

    # Now loop over all grid cells and compute the EFT model for different values of the growth rate
    allPlin = []
    allPloop = []
    allParams = []
    for i in range(startval, endval):
        idx = i - startval
        print("i on tot", i, nvals)

        bird.f = fvals[i]

        # Compute all the components and resummation
        bird.reducePsCfl()
        resum.Ps(bird)

        Plin, Ploop = bird.formatTaylor()
        idxcol = np.full([Plin.shape[0], 1], idx)
        allPlin.append(np.hstack([Plin, idxcol]))
        allPloop.append(np.hstack([Ploop, idxcol]))
        if (i == 0) or (i % 10 == 0):
            print("theta check: ", i - startval, i, bird.f)
        np.save(os.path.join(pardict["outpk"], "Plin_template_run%s.npy" % (str(job_no))), np.array(allPlin))
        np.save(os.path.join(pardict["outpk"], "Ploop_template_run%s.npy" % (str(job_no))), np.array(allPloop))
