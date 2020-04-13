import os
import sys
import numpy as np
import Grid
# from findiff import FinDiff, coefficients, Coefficient

basedir = '/Volumes/Work/UQ/DESI/cBIRD/output_files/'
pathpk = os.path.join(basedir, "Pk")
pathgrid = os.path.join(basedir, "GridsEFT")
gridname = Grid.gridname

nruns = int(sys.argv[1])
lenbatch = int(sys.argv[2])
ntot = nruns*lenbatch

linfailed = []
loopfailed = []
for area in ["NGC", "SGC"]:
    for i in range(nruns):
        print(i, area)
        checklin = os.path.isfile(os.path.join(pathpk, "Plin_%s_run%d.npy" % (area, i)))
        checkloop = os.path.isfile(os.path.join(pathpk, "Ploop_%s_run%d.npy" % (area, i)))
        if not checklin:
            print("Failed linear run %s-%d" % (area, i))
            linfailed.append((area, i))
        else:
            Plin = np.load(os.path.join(pathpk, "Plin_%s_run%d.npy" % (area, i)))
            if (lenbatch != len(Plin)):
                print("Failed length linear run %s-%d" % (area, i))
                linfailed.append((area, i))
        if not checkloop:
            print("Failed loop run %s-%d" % (area, i))
            loopfailed.append((area, i))
        else:
            Ploop = np.load(os.path.join(pathpk, "Ploop_%s_run%d.npy" % (area, i)))
            if (lenbatch != len(Ploop)):
                print("Failed length loop run %s-%d" % (area, i))
                loopfailed.append((area, i))

print("Linear failed: %d over %d, %f %%" % (len(linfailed), ntot, 100 * float(len(linfailed)) / ntot))
print("Loop failed: %d over %d, %f %%" % (len(loopfailed), ntot, 100 * float(len(loopfailed)) / ntot))

if (len(linfailed) + len(loopfailed)) > 0:
    print(linfailed, loopfailed)
    raise Exception("Some processes have failed!")

for area in ["NGC", "SGC"]:
    gridlin = []
    gridloop = []
    for i in range(nruns):
        print("Run ", area, i)
        Plin = np.load(os.path.join(pathpk, "Plin_%s_run%d.npy" % (area, i)))
        Ploop = np.load(os.path.join(pathpk, "Ploop_%s_run%d.npy" % (area, i)))
        gridlin.append(Plin[:, :, :-1])
        gridloop.append(Ploop[:, :, :-1])
        checklin = (lenbatch == len(Plin))
        checkloop = (lenbatch == len(Ploop))
        if not checklin:
            print("Problem in linear PS: ", i, i * lenbatch, Plin[0, 0, -1])
        if not checkloop:
            print("Problem in loop PS: ", i, i * lenbatch, Ploop[0, 0, -1])

    np.save(os.path.join(pathgrid, "Tablecoord_%s_%s.npy" % (area, gridname)), Grid.truegrid)
    g1 = np.concatenate(gridlin)
    np.save(os.path.join(pathgrid, "TablePlin_%s_%s.npy" % (area, gridname)), g1)
    g2 = np.concatenate(gridloop)
    np.save(os.path.join(pathgrid, "TablePloop_%s_%s.npy" % (area, gridname)), g2)


