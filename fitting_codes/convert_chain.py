import sys
import numpy as np
import copy
from configobj import ConfigObj
from chainconsumer import ChainConsumer

sys.path.append("../")
from tbird.Grid import run_camb


def read_chain(chainfile, burnlimitlow=1000, burnlimitup=None):

    # Read in the samples
    walkers = []
    samples = []
    like = []
    infile = open(chainfile, "r")
    for line in infile:
        ln = line.split()
        samples.append(list(map(float, ln[1:-1])))
        walkers.append(int(ln[0]))
        like.append(float(ln[-1]))
    infile.close()

    like = np.array(like)
    walkers = np.array(walkers)
    samples = np.array(samples)
    nwalkers = max(walkers)

    if burnlimitup is None:
        bestid = np.argmax(like)
    else:
        bestid = np.argmax(like[: np.amax(walkers) * burnlimitup])

    burntin = []
    burntlike = []
    weightsarray = []
    nburntin = 0

    for i in range(nwalkers + 1):
        ind = np.where(walkers == i)[0]
        if len(ind) == 0:
            continue
        x = [j for j in range(len(ind))]
        if burnlimitup is None:
            ind2 = np.where(np.asarray(x) >= burnlimitlow)[0]
        else:
            ind2 = np.where(np.logical_and(np.asarray(x) >= burnlimitlow, np.asarray(x) <= burnlimitup))[0]
        for k in range(len(ind2 + 1)):
            burntin.append(samples[ind[ind2[k]]])
            burntlike.append(like[ind[ind2[k]]])
        nburntin += len(ind2)
    burntin = np.array(burntin)
    burntlike = np.array(burntlike)

    return burntin, samples[bestid], burntlike


if __name__ == "__main__":

    # Reads in a chain containing cosmological parameters output from a fitting routine, and
    # converts the points to Da(z), H(z) and f(z)*sigma8(z)

    # First, read in the config file used for the fit
    configfile = sys.argv[1]
    pardict = ConfigObj(configfile)
    fbc = float(pardict["omega_b"]) / float(pardict["omega_cdm"])

    # Compute the values at the central point
    _, _, Da, Hz, f, sigma8 = run_camb(pardict)
    truth = 2997.92458 * Da / float(pardict["h"]), 100.0 * float(pardict["h"]) * Hz, f * sigma8
    print(truth)

    # Extract the name of the chainfile and read it in
    burntin, bestfit, like = read_chain(pardict["fitfile"] + ".dat", burnlimitup=20000)

    # Loop over the parameters in the chain and run camb to get Da, H and fsigma8
    chainvals = np.empty((3, 1000))
    for i, vals in enumerate(burntin):
        print(i)
        if i == 1000:
            break
        parref = copy.deepcopy(pardict)
        omega_cdm = vals[2] * vals[1] ** 2 / (1.0 + fbc)
        omega_b = vals[2] * vals[1] ** 2 - omega_cdm
        parref["ln10^{10}A_s"], parref["h"], parref["omega_b"], parref["omega_cdm"] = (
            vals[0],
            vals[1],
            omega_b,
            omega_cdm,
        )
        _, _, Da, Hz, f, sigma8 = run_camb(parref)
        chainvals[:, i] = 2997.92458 * Da / vals[1], 100.0 * vals[1] * Hz, f * sigma8

    figfile = pardict["fitfile"] + "_converted.pdf"

    c = ChainConsumer()
    fig = c.plotter.plot(figsize="column", filename=figfile, truth=truth)
    print(c.analysis.get_summary())
