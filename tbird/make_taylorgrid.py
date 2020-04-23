import numpy as np
import scipy as sp
import os
import sys

sys.path.append("../")
import Grid
import pybird
import copy
import time
import pandas as pd
import matplotlib.pyplot as plt

# time.sleep(4)

basedir = "/Volumes/Work/UQ/DESI/cBIRD/"
OUTPATH = os.path.join(basedir, "UNIT_output_files/grid_outputs/")
outpk = os.path.join(basedir, "UNIT_output_files/Pk/")

nrun = int(sys.argv[1])
runs = int(sys.argv[2])
lenrun = int(len(Grid.flattenedgrid) / runs)
start = nrun * lenrun
final = min((nrun + 1) * lenrun, len(Grid.flattenedgrid))
arrayred = Grid.flattenedgrid[start:final]
print(nrun, start, final)
sizearray = len(arrayred)

freepar = Grid.freepar

### To create outside the grid
common = pybird.Common(Nl=2, kmax=0.4, optiresum=True)
nonlinear = pybird.NonLinear(load=False, save=False, co=common)
resum = pybird.Resum(LambdaIR=0.1, co=common)
kbird = pybird.common.k
allk = np.concatenate([kbird, kbird]).reshape(-1, 1)

# Get some cosmological values at the central point
parameters = copy.deepcopy(Grid.parref)
truetheta = Grid.valueref
for k, var in enumerate(freepar):
    parameters[var] = truetheta[k]
kin, Plin, z, Omega_m = Grid.CompPterms(parameters)

# Read in the window function files
kmin, kmax, nkout, nkth = 0.0, 0.4, 40, 400
win_file_NGC = os.path.join(basedir, "beutler_2019_dr12_z061_pk/W2D_pk_BOSS_DR12_NGC_z3_1_1_1_1_1_10_10.dat")
m_file_NGC = os.path.join(basedir, "beutler_2019_dr12_z061_pk/M2D_pk_BOSS_DR12_NGC_z3_1_1_1_1_1_10_10.dat")
win_file_SGC = os.path.join(basedir, "beutler_2019_dr12_z061_pk/W2D_pk_BOSS_DR12_SGC_z3_1_1_1_1_1_10_10.dat")
m_file_SGC = os.path.join(basedir, "beutler_2019_dr12_z061_pk/M2D_pk_BOSS_DR12_SGC_z3_1_1_1_1_1_10_10.dat")

df = pd.read_csv(win_file_NGC, comment="#", delim_whitespace=True, header=None)
W_mat_NGC = df.to_numpy().astype(np.float32)
print(np.shape(W_mat_NGC))
df = pd.read_csv(m_file_NGC, comment="#", delim_whitespace=True, header=None)
print(np.shape(df.to_numpy().astype(np.float32)))
WM_mat_NGC = W_mat_NGC @ df.to_numpy().astype(np.float32)

df = pd.read_csv(win_file_SGC, comment="#", delim_whitespace=True, header=None)
W_mat_SGC = df.to_numpy().astype(np.float32)
df = pd.read_csv(m_file_SGC, comment="#", delim_whitespace=True, header=None)
WM_mat_SGC = W_mat_SGC @ df.to_numpy().astype(np.float32)

Omega_m_fid = 0.307115  # Fiducial Omega_m value used to compute the clustering measurements. Not necessarily the same as the value used for the Grid centre.
kin = np.linspace(kmin, kmax, nkth, endpoint=False) + 0.5 * (kmax - kmin) / nkth
kout = np.linspace(kmin, kmax, nkout, endpoint=False) + 0.5 * (kmax - kmin) / nkout
projection_NGC = pybird.Projection(kout, Omega_m_fid, z, window_fourier_name=None, co=common)
projection_SGC = pybird.Projection(kout, Omega_m_fid, z, window_fourier_name=None, co=common)
projection_NGC.p = kin
projection_SGC.p = kin
WM_NGC_reshape = WM_mat_NGC.reshape(5, nkout, 3, nkth)
WM_SGC_reshape = WM_mat_SGC.reshape(5, nkout, 3, nkth)
projection_NGC.Waldk = np.transpose(WM_NGC_reshape[[0, 2], :, :2, :], axes=(0, 2, 1, 3))
projection_SGC.Waldk = np.transpose(WM_SGC_reshape[[0, 2], :, :2, :], axes=(0, 2, 1, 3))
print(np.shape(projection_SGC.Waldk))

allPlin_NGC = []
allPloop_NGC = []
allPlin_SGC = []
allPloop_SGC = []
for i, theta in enumerate(arrayred):
    parameters = copy.deepcopy(Grid.parref)
    truetheta = Grid.valueref + theta * Grid.delta
    idx = i
    print("i on tot", i, sizearray)

    # Get linear power spectrum from Class
    # TBD: Update this to use Classy (python-Class), which will also allow for a self-consistent
    # calculation of Omega_m, Da and H. This is already the case in the monte-python likelihood
    parameters["PathToOutput"] = os.path.join(OUTPATH, "output_%s_%s" % (str(nrun), str(i)))
    for k, var in enumerate(freepar):
        parameters[var] = truetheta[k]
    kin, Plin, z, Omega_m = Grid.CompPterms(parameters)

    # Warning: Assumes LCDM
    Da = pybird.DA(Omega_m, z)
    Hz = pybird.Hubble(Omega_m, z)
    fN = pybird.fN(Omega_m, z)

    # Get non-linear power spectrum from pybird
    bird = pybird.Bird(kin, Plin, fN, DA=Da, H=Hz, z=z, which="all", co=common)
    nonlinear.PsCf(bird)
    bird.setPsCfl()
    resum.Ps(bird)

    # Add AP and window function
    bird_NGC = copy.deepcopy(bird)
    projection_NGC.AP(bird_NGC)
    projection_NGC.Window(bird_NGC)

    projection_SGC.AP(bird)
    projection_SGC.Window(bird)

    Plin_NGC, Ploop_NGC = bird_NGC.formatTaylor(kdata=kout)
    Plin_SGC, Ploop_SGC = bird.formatTaylor(kdata=kout)
    idxcol = np.full([Plin_NGC.shape[0], 1], idx)
    allPlin_NGC.append(np.hstack([Plin_NGC, idxcol]))
    allPloop_NGC.append(np.hstack([Ploop_NGC, idxcol]))
    allPlin_SGC.append(np.hstack([Plin_SGC, idxcol]))
    allPloop_SGC.append(np.hstack([Ploop_SGC, idxcol]))
    if (i == 0) or ((i + 1) % 100 == 0):
        print("theta check: ", arrayred[idx], theta, truetheta)
    np.save(os.path.join(outpk, "Plin_NGC_run%s.npy" % (str(nrun))), np.array(allPlin_NGC))
    np.save(os.path.join(outpk, "Ploop_NGC_run%s.npy" % (str(nrun))), np.array(allPloop_NGC))
    np.save(os.path.join(outpk, "Plin_SGC_run%s.npy" % (str(nrun))), np.array(allPlin_SGC))
    np.save(os.path.join(outpk, "Ploop_SGC_run%s.npy" % (str(nrun))), np.array(allPloop_SGC))
