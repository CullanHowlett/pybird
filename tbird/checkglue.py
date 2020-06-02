import os
import sys
import numpy as np
from configobj import ConfigObj

if __name__ == "__main__":

    # Read in the config file and total number of jobs
    configfile = sys.argv[1]
    njobs = int(sys.argv[2])
    pardict = ConfigObj(configfile)
    gridname = pardict["Code"].lower() + "-" + pardict["gridname"]

    ntot = (2 * float(pardict["order"]) + 1) ** len(pardict["freepar"])
    lenbatch = ntot / njobs

    linfailed = []
    loopfailed = []
    for i in range(njobs):
        print(i)
        checklin = os.path.isfile(os.path.join(pardict["outpk"], "Plin_run%d.npy" % i))
        checkloop = os.path.isfile(os.path.join(pardict["outpk"], "Ploop_run%d.npy" % i))
        checkCflin = os.path.isfile(os.path.join(pardict["outpk"], "Clin_run%d.npy" % i))
        checkCfloop = os.path.isfile(os.path.join(pardict["outpk"], "Cloop_run%d.npy" % i))
        if not checklin and checkCflin:
            print("Failed linear run %d" % i)
            linfailed.append(i)
        else:
            Plin = np.load(os.path.join(pardict["outpk"], "Plin_run%d.npy" % i))
            Clin = np.load(os.path.join(pardict["outpk"], "Clin_run%d.npy" % i))
            if lenbatch != len(Plin) and lenbatch != len(Clin):
                print("Failed length linear run %d" % i)
                linfailed.append(i)
        if not checkloop and checkCfloop:
            print("Failed loop run %d" % i)
            loopfailed.append(i)
        else:
            Ploop = np.load(os.path.join(pardict["outpk"], "Ploop_run%d.npy" % i))
            Cloop = np.load(os.path.join(pardict["outpk"], "Cloop_run%d.npy" % i))
            if lenbatch != len(Ploop) and lenbatch != len(Cloop):
                print("Failed length loop run %d" % i)
                loopfailed.append(i)

    print("Linear failed: %d over %d, %f %%" % (len(linfailed), ntot, 100 * float(len(linfailed)) / ntot))
    print("Loop failed: %d over %d, %f %%" % (len(loopfailed), ntot, 100 * float(len(loopfailed)) / ntot))

    if (len(linfailed) + len(loopfailed)) > 0:
        print(linfailed, loopfailed)
        raise Exception("Some processes have failed!")

    gridPin = []
    gridlin = []
    gridloop = []
    gridCflin = []
    gridCfloop = []
    gridparams = []
    for i in range(njobs):
        print("Run ", i)
        Params = np.load(os.path.join(pardict["outpk"], "Params_run%d.npy" % i))
        if pardict["Code"] == "CAMB":
            Pin = np.load(os.path.join(pardict["outpk"], "CAMB_run%d.npy" % i))
        else:
            Pin = np.load(os.path.join(pardict["outpk"], "CLASS_run%d.npy" % i))
        Plin = np.load(os.path.join(pardict["outpk"], "Plin_run%d.npy" % i))
        Ploop = np.load(os.path.join(pardict["outpk"], "Ploop_run%d.npy" % i))
        Clin = np.load(os.path.join(pardict["outpk"], "Clin_run%d.npy" % i))
        Cloop = np.load(os.path.join(pardict["outpk"], "Cloop_run%d.npy" % i))
        gridparams.append(Params[:, :-1])
        gridPin.append(Pin[:, :-1])
        gridlin.append(Plin[:, :, :-1])
        gridloop.append(Ploop[:, :, :-1])
        gridCflin.append(Clin[:, :, :-1])
        gridCfloop.append(Cloop[:, :, :-1])
        checklin = lenbatch == len(Plin)
        checkloop = lenbatch == len(Ploop)
        checkCflin = lenbatch == len(Clin)
        checkCfloop = lenbatch == len(Cloop)
        if not checklin:
            print("Problem in linear PS: ", i, i * lenbatch, Plin[0, 0, -1])
        if not checkloop:
            print("Problem in loop PS: ", i, i * lenbatch, Ploop[0, 0, -1])
        if not checkCflin:
            print("Problem in linear CF: ", i, i * lenbatch, Clin[0, 0, -1])
        if not checkCfloop:
            print("Problem in loop CF: ", i, i * lenbatch, Cloop[0, 0, -1])

    if pardict["code"] == "CAMB":
        np.save(os.path.join(pardict["outgrid"], "TableCAMB_%s.npy" % gridname), np.concatenate(gridPin))
    else:
        np.save(os.path.join(pardict["outgrid"], "TableCLASS_%s.npy" % gridname), np.concatenate(gridPin))
    np.save(os.path.join(pardict["outgrid"], "TablePlin_%s.npy" % gridname), np.concatenate(gridlin))
    np.save(os.path.join(pardict["outgrid"], "TablePloop_%s.npy" % gridname), np.concatenate(gridloop))
    np.save(os.path.join(pardict["outgrid"], "TableClin_%s.npy" % gridname), np.concatenate(gridCflin))
    np.save(os.path.join(pardict["outgrid"], "TableCloop_%s.npy" % gridname), np.concatenate(gridCfloop))
    np.save(os.path.join(pardict["outgrid"], "TableParams_%s.npy" % gridname), np.concatenate(gridparams))
