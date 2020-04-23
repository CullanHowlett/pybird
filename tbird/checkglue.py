import os
import sys
import numpy as np
from configobj import ConfigObj

if __name__ == "__main__":

    # Read in the config file and total number of jobs
    configfile = sys.argv[1]
    njobs = int(sys.argv[2])
    pardict = ConfigObj(configfile)

    ntot = (2 * float(pardict["order"]) + 1) ** len(pardict["freepar"])
    lenbatch = ntot / njobs

    linfailed = []
    loopfailed = []
    for i in range(njobs):
        print(i)
        checklin = os.path.isfile(os.path.join(pardict["outpk"], "Plin_run%d    .npy" % (i)))
        checkloop = os.path.isfile(os.path.join(pardict["outpk"], "Ploop_run%d.npy" % (i)))
        if not checklin:
            print("Failed linear run %d" % i)
            linfailed.append(i)
        else:
            Plin = np.load(os.path.join(pardict["outpk"], "Plin_run%d.npy" % (i)))
            if lenbatch != len(Plin):
                print("Failed length linear run %d" % (i))
                linfailed.append((i))
        if not checkloop:
            print("Failed loop run %d" % (i))
            loopfailed.append((i))
        else:
            Ploop = np.load(os.path.join(pardict["outpk"], "Ploop_run%d.npy" % (i)))
            if lenbatch != len(Ploop):
                print("Failed length loop run %d" % (i))
                loopfailed.append((i))

    print("Linear failed: %d over %d, %f %%" % (len(linfailed), ntot, 100 * float(len(linfailed)) / ntot))
    print("Loop failed: %d over %d, %f %%" % (len(loopfailed), ntot, 100 * float(len(loopfailed)) / ntot))

    if (len(linfailed) + len(loopfailed)) > 0:
        print(linfailed, loopfailed)
        raise Exception("Some processes have failed!")

    gridlin = []
    gridloop = []
    gridparams = []
    for i in range(njobs):
        print("Run ", i)
        Params = np.load(os.path.join(pardict["outpk"], "Params_run%d.npy" % i))
        Plin = np.load(os.path.join(pardict["outpk"], "Plin_run%d.npy" % i))
        Ploop = np.load(os.path.join(pardict["outpk"], "Ploop_run%d.npy" % i))
        gridparams.append(Params[:, -1])
        gridlin.append(Plin[:, :, :-1])
        gridloop.append(Ploop[:, :, :-1])
        checklin = lenbatch == len(Plin)
        checkloop = lenbatch == len(Ploop)
        if not checklin:
            print("Problem in linear PS: ", i, i * lenbatch, Plin[0, 0, -1])
        if not checkloop:
            print("Problem in loop PS: ", i, i * lenbatch, Ploop[0, 0, -1])

    np.save(os.path.join(pardict["outgrid"], "TablePlin_%s.npy" % pardict["gridname"]), np.concatenate(gridlin))
    np.save(os.path.join(pardict["outgrid"], "TablePloop_%s.npy" % pardict["gridname"]), np.concatenate(gridloop))
    np.save(os.path.join(pardict["outgrid"], "TableParams_%s.npy" % pardict["gridname"]), np.concatenate(gridparams))
