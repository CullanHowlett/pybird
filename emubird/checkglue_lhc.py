import os
import sys
import numpy as np
from configobj import ConfigObj

if __name__ == "__main__":

    # Read in the config file and total number of jobs
    configfile = sys.argv[1]
    njobs = int(sys.argv[2])
    redindex = int(sys.argv[3])
    pardict = ConfigObj(configfile)
    gridnames = np.loadtxt(pardict["gridname"], dtype=str)
    outgrids = np.loadtxt(pardict["outgrid"], dtype=str)
    gridname = pardict["code"].lower() + "-" + gridnames[redindex]

    # Read in a Latin Hypercube with the correct number of dimensions
    order = float(pardict["order"])
    ndims = len(pardict["freepar"])
    valueref = np.array([float(pardict[k]) for k in pardict["freepar"]])
    delta = np.fabs(np.array(pardict["dx"], dtype=np.float) * valueref)
    lhc = np.load(str("lhc_%dD_10000.npy" % ndims))
    lhc = (2.0 * lhc - 1.0) * order
    ntot = len(lhc)
    lenbatch = ntot / njobs
    print(lenbatch)

    linfailed = []
    loopfailed = []
    for i in range(njobs):
        print(i)
        checklin = os.path.isfile(os.path.join(pardict["outpk"], "redindex%d" % redindex, "Plin_lhc_run%d.npy" % i))
        checkloop = os.path.isfile(os.path.join(pardict["outpk"], "redindex%d" % redindex, "Ploop_lhc_run%d.npy" % i))
        checkCflin = os.path.isfile(os.path.join(pardict["outpk"], "redindex%d" % redindex, "Clin_lhc_run%d.npy" % i))
        checkCfloop = os.path.isfile(os.path.join(pardict["outpk"], "redindex%d" % redindex, "Cloop_lhc_run%d.npy" % i))
        if not checklin or not checkCflin:
            print("Failed linear run %d" % i)
            linfailed.append(i)
        else:
            Plin = np.load(os.path.join(pardict["outpk"], "redindex%d" % redindex, "Plin_lhc_run%d.npy" % i))
            Clin = np.load(os.path.join(pardict["outpk"], "redindex%d" % redindex, "Clin_lhc_run%d.npy" % i))
            if lenbatch != len(Plin) or lenbatch != len(Clin):
                print("Failed length linear run %d" % i)
                linfailed.append(i)
        if not checkloop or not checkCfloop:
            print("Failed loop run %d" % i)
            loopfailed.append(i)
        else:
            Ploop = np.load(os.path.join(pardict["outpk"], "redindex%d" % redindex, "Ploop_lhc_run%d.npy" % i))
            Cloop = np.load(os.path.join(pardict["outpk"], "redindex%d" % redindex, "Cloop_lhc_run%d.npy" % i))
            if lenbatch != len(Ploop) or lenbatch != len(Cloop):
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
        Params = np.load(os.path.join(pardict["outpk"], "redindex%d" % redindex, "Params_lhc_run%d.npy" % i))
        if pardict["code"] == "CAMB":
            Pin = np.load(os.path.join(pardict["outpk"], "redindex%d" % redindex, "CAMB_lhc_run%d.npy" % i))
        else:
            Pin = np.load(os.path.join(pardict["outpk"], "redindex%d" % redindex, "CLASS_lhc_run%d.npy" % i))
        Plin = np.load(os.path.join(pardict["outpk"], "redindex%d" % redindex, "Plin_lhc_run%d.npy" % i))
        Ploop = np.load(os.path.join(pardict["outpk"], "redindex%d" % redindex, "Ploop_lhc_run%d.npy" % i))
        Clin = np.load(os.path.join(pardict["outpk"], "redindex%d" % redindex, "Clin_lhc_run%d.npy" % i))
        Cloop = np.load(os.path.join(pardict["outpk"], "redindex%d" % redindex, "Cloop_lhc_run%d.npy" % i))
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
        np.save(os.path.join(outgrids[redindex], "TableCAMB_lhc_%s.npy" % gridname), np.concatenate(gridPin))
    else:
        np.save(os.path.join(outgrids[redindex], "TableCLASS_lhc_%s.npy" % gridname), np.concatenate(gridPin))
    np.save(os.path.join(outgrids[redindex], "TablePlin_lhc_%s.npy" % gridname), np.concatenate(gridlin))
    np.save(os.path.join(outgrids[redindex], "TablePloop_lhc_%s.npy" % gridname), np.concatenate(gridloop))
    np.save(os.path.join(outgrids[redindex], "TableClin_lhc_%s.npy" % gridname), np.concatenate(gridCflin))
    np.save(os.path.join(outgrids[redindex], "TableCloop_lhc_%s.npy" % gridname), np.concatenate(gridCfloop))
    np.save(os.path.join(outgrids[redindex], "TableParams_lhc_%s.npy" % gridname), np.concatenate(gridparams))
