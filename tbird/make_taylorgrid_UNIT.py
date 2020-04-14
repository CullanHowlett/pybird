import numpy as np
import scipy as sp
import os
import sys
sys.path.append('../pybird/')
import Grid
import pybird
import copy
import time
import pandas as pd
import matplotlib.pyplot as plt

#time.sleep(4)

outpk = "/Volumes/Work/UQ/DESI/cBIRD/UNIT_output_files/Pk/"

nrun = int(sys.argv[1])
runs = int(sys.argv[2])
lenrun = int(len(Grid.flattenedgrid) / runs)
start = nrun * lenrun
final = min((nrun+1) * lenrun, len(Grid.flattenedgrid))
arrayred = Grid.flattenedgrid[start:final]
print(nrun, start, final)
sizearray = len(arrayred)

freepar = Grid.freepar

### To create outside the grid
common = pybird.Common(Nl=2, kmax=0.4, optiresum=True)
nonlinear = pybird.NonLinear(load=False, save=False, co=common)
resum = pybird.Resum(LambdaIR=0.1, co=common)

# Get some cosmological values at the central point
parameters = copy.deepcopy(Grid.parref)
truetheta = Grid.valueref
for k, var in enumerate(freepar):
    parameters[var] = truetheta[k]
kin, Plin, z, Omega_m, Da, Hz, fN = Grid.CompPterms_camb(parameters)

# Now window at the moment for the UNIT sims, so we'll create an identity matrix for this. I'm also
# assuming that the fiducial cosmology used to make the measurements is the same as Grid centre
kmin, kmax, nkout, nkth = 0.0, 0.4, 40, 40
kin = np.linspace(kmin, kmax, nkth, endpoint=False) + 0.5 * (kmax - kmin) / nkth
kout = np.linspace(kmin, kmax, nkout, endpoint=False) + 0.5 * (kmax - kmin) / nkout
projection = pybird.Projection(kout, Omega_m_fid, z, DA=Da, H=Hz, window_fourier_name=None, co=common)
projection.p = kin
projection.Waldk = np.eye(2*len(kin))

allPlin = []
allPloop = []
for i, theta in enumerate(arrayred):
    parameters = copy.deepcopy(Grid.parref)
    truetheta = Grid.valueref + theta * Grid.delta
    idx = i
    print ("i on tot", i, sizearray)

    kin, Plin, z, Omega_m, Da, Hz, fN = Grid.CompPterms_camb(parameters)

    # Get non-linear power spectrum from pybird
    bird = pybird.Bird(kin, Plin, fN, DA=Da, H=Hz, z=z, which='all', co=common)
    nonlinear.PsCf(bird)
    bird.setPsCfl()
    resum.Ps(bird)

    projection.AP(bird)
    projection.Window(bird)
    
    Plin, Ploop = bird.formatTaylor()
    idxcol = np.full([Plin.shape[0], 1], idx)
    allPlin.append(np.hstack([Plin, idxcol]))
    allPloop.append(np.hstack([Ploop, idxcol]))
    if (i == 0) or ((i+1) % 100 == 0):
        print("theta check: ", arrayred[idx], theta, truetheta)
    np.save(os.path.join(outpk, "Plin_run%s.npy" % (str(nrun))), np.array(allPlin))
    np.save(os.path.join(outpk, "Ploop_run%s.npy" % (str(nrun))), np.array(allPloop))
