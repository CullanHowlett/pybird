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

    ntot = (2 * float(pardict["order"]) + 1) ** len(pardict["freepar"])
    lenbatch = ntot / njobs
    print(lenbatch)

    linfailed = []
    loopfailed = []
    for i in range(njobs):
        print(i)
        checklin = os.path.isfile(os.path.join(pardict["outpk"], "redindex%d" % redindex, "Plin_run%d.npy" % i))
        checkloop = os.path.isfile(os.path.join(pardict["outpk"], "redindex%d" % redindex, "Ploop_run%d.npy" % i))
        checkCflin = os.path.isfile(os.path.join(pardict["outpk"], "redindex%d" % redindex, "Clin_run%d.npy" % i))
        checkCfloop = os.path.isfile(os.path.join(pardict["outpk"], "redindex%d" % redindex, "Cloop_run%d.npy" % i))
        # checklin_noAP = os.path.isfile(os.path.join(pardict["outpk"], "Plin_run%d_noAP.npy" % i))
        # checkloop_noAP = os.path.isfile(os.path.join(pardict["outpk"], "Ploop_run%d_noAP.npy" % i))
        # checkCflin_noAP = os.path.isfile(os.path.join(pardict["outpk"], "Clin_run%d_noAP.npy" % i))
        # checkCfloop_noAP = os.path.isfile(os.path.join(pardict["outpk"], "Cloop_run%d_noAP.npy" % i))
        if not checklin or not checkCflin:
            print("Failed linear run %d" % i)
            linfailed.append(i)
        else:
            Plin = np.load(os.path.join(pardict["outpk"], "redindex%d" % redindex, "Plin_run%d.npy" % i))
            Clin = np.load(os.path.join(pardict["outpk"], "redindex%d" % redindex, "Clin_run%d.npy" % i))
            # Plin_noAP = np.load(os.path.join(pardict["outpk"], "Plin_run%d_noAP.npy" % i))
            # Clin_noAP = np.load(os.path.join(pardict["outpk"], "Clin_run%d_noAP.npy" % i))
            if lenbatch != len(Plin) or lenbatch != len(Clin):
                print("Failed length linear run %d" % i)
                linfailed.append(i)
        if not checkloop or not checkCfloop:
            print("Failed loop run %d" % i)
            loopfailed.append(i)
        else:
            Ploop = np.load(os.path.join(pardict["outpk"], "redindex%d" % redindex, "Ploop_run%d.npy" % i))
            Cloop = np.load(os.path.join(pardict["outpk"], "redindex%d" % redindex, "Cloop_run%d.npy" % i))
            # Ploop_noAP = np.load(os.path.join(pardict["outpk"], "Ploop_run%d_noAP.npy" % i))
            # Cloop_noAP = np.load(os.path.join(pardict["outpk"], "Cloop_run%d_noAP.npy" % i))
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
    # gridlin_noAP = []
    # gridloop_noAP = []
    # gridCflin_noAP = []
    # gridCfloop_noAP = []
    gridparams = []
    for i in range(njobs):
        print("Run ", i)
        Params = np.load(os.path.join(pardict["outpk"], "redindex%d" % redindex, "Params_run%d.npy" % i))
        if pardict["code"] == "CAMB":
            Pin = np.load(os.path.join(pardict["outpk"], "redindex%d" % redindex, "CAMB_run%d.npy" % i))
        else:
            Pin = np.load(os.path.join(pardict["outpk"], "redindex%d" % redindex, "CLASS_run%d.npy" % i))
        Plin = np.load(os.path.join(pardict["outpk"], "redindex%d" % redindex, "Plin_run%d.npy" % i))
        Ploop = np.load(os.path.join(pardict["outpk"], "redindex%d" % redindex, "Ploop_run%d.npy" % i))
        Clin = np.load(os.path.join(pardict["outpk"], "redindex%d" % redindex, "Clin_run%d.npy" % i))
        Cloop = np.load(os.path.join(pardict["outpk"], "redindex%d" % redindex, "Cloop_run%d.npy" % i))
        # Plin_noAP = np.load(os.path.join(pardict["outpk"], "Plin_run%d_noAP.npy" % i))
        # Ploop_noAP = np.load(os.path.join(pardict["outpk"], "Ploop_run%d_noAP.npy" % i))
        # Clin_noAP = np.load(os.path.join(pardict["outpk"], "Clin_run%d_noAP.npy" % i))
        # Cloop_noAP = np.load(os.path.join(pardict["outpk"], "Cloop_run%d_noAP.npy" % i))
        gridparams.append(Params[:, :-1])
        gridPin.append(Pin[:, :-1])
        gridlin.append(Plin[:, :, :-1])
        gridloop.append(Ploop[:, :, :-1])
        gridCflin.append(Clin[:, :, :-1])
        gridCfloop.append(Cloop[:, :, :-1])
        # gridlin_noAP.append(Plin_noAP[:, :, :-1])
        # gridloop_noAP.append(Ploop_noAP[:, :, :-1])
        # gridCflin_noAP.append(Clin_noAP[:, :, :-1])
        # gridCfloop_noAP.append(Cloop_noAP[:, :, :-1])
        checklin = lenbatch == len(Plin)
        checkloop = lenbatch == len(Ploop)
        checkCflin = lenbatch == len(Clin)
        checkCfloop = lenbatch == len(Cloop)
        # checklin_noAP = lenbatch == len(Plin_noAP)
        # checkloop_noAP = lenbatch == len(Ploop_noAP)
        # checkCflin_noAP = lenbatch == len(Clin_noAP)
        # checkCfloop_noAP = lenbatch == len(Cloop_noAP)
        if not checklin:
            print("Problem in linear PS: ", i, i * lenbatch, Plin[0, 0, -1])
        if not checkloop:
            print("Problem in loop PS: ", i, i * lenbatch, Ploop[0, 0, -1])
        if not checkCflin:
            print("Problem in linear CF: ", i, i * lenbatch, Clin[0, 0, -1])
        if not checkCfloop:
            print("Problem in loop CF: ", i, i * lenbatch, Cloop[0, 0, -1])
        """if not checklin_noAP:
            print("Problem in linear PS without AP effect: ", i, i * lenbatch, Plin_noAP[0, 0, -1])
        if not checkloop_noAP:
            print("Problem in loop PS without AP effect: ", i, i * lenbatch, Ploop_noAP[0, 0, -1])
        if not checkCflin_noAP:
            print("Problem in linear CF without AP effect: ", i, i * lenbatch, Clin_noAP[0, 0, -1])
        if not checkCfloop_noAP:
            print("Problem in loop CF without AP effect: ", i, i * lenbatch, Cloop_noAP[0, 0, -1])"""

    if pardict["code"] == "CAMB":
        np.save(os.path.join(outgrids[redindex], "TableCAMB_%s.npy" % gridname), np.concatenate(gridPin))
    else:
        np.save(os.path.join(outgrids[redindex], "TableCLASS_%s.npy" % gridname), np.concatenate(gridPin))
    np.save(os.path.join(outgrids[redindex], "TablePlin_%s.npy" % gridname), np.concatenate(gridlin))
    np.save(os.path.join(outgrids[redindex], "TablePloop_%s.npy" % gridname), np.concatenate(gridloop))
    np.save(os.path.join(outgrids[redindex], "TableClin_%s.npy" % gridname), np.concatenate(gridCflin))
    np.save(os.path.join(outgrids[redindex], "TableCloop_%s.npy" % gridname), np.concatenate(gridCfloop))
    # np.save(os.path.join(pardict["outgrid"], "TablePlin_%s_noAP.npy" % gridname), np.concatenate(gridlin_noAP))
    # np.save(os.path.join(pardict["outgrid"], "TablePloop_%s_noAP.npy" % gridname), np.concatenate(gridloop_noAP))
    # np.save(os.path.join(pardict["outgrid"], "TableClin_%s_noAP.npy" % gridname), np.concatenate(gridCflin_noAP))
    # np.save(os.path.join(pardict["outgrid"], "TableCloop_%s_noAP.npy" % gridname), np.concatenate(gridCfloop_noAP))
    np.save(os.path.join(outgrids[redindex], "TableParams_%s.npy" % gridname), np.concatenate(gridparams))
