import os
import sys
import numpy as np
from scipy import interpolate
from configobj import ConfigObj
from chainconsumer import ChainConsumer

sys.path.append("../")
from tbird.computederivs import get_grids
from tbird.Grid import grid_properties


def get_PSTaylor(dtheta, derivatives, order=3):
    # Shape of dtheta: number of free parameters
    # Shape of derivatives: tuple up to third derivative where each element has shape (num free par, multipoles, lenk, columns)
    t1 = np.einsum("p,pm->m", dtheta, derivatives[1])
    t2diag = np.einsum("p,pm->m", dtheta ** 2, derivatives[2])
    t2nondiag = np.sum([dtheta[d[0]] * dtheta[d[1]] * d[2] for d in derivatives[3]], axis=0)
    t3diag = np.einsum("p,pm->m", dtheta ** 3, derivatives[4])
    t3semidiagx = np.sum([dtheta[d[0]] ** 2 * dtheta[d[1]] * d[2] for d in derivatives[5]], axis=0)
    t3semidiagy = np.sum([dtheta[d[0]] * dtheta[d[1]] ** 2 * d[2] for d in derivatives[6]], axis=0)
    t3nondiag = np.sum([dtheta[d[0]] * dtheta[d[1]] * dtheta[d[2]] * d[3] for d in derivatives[7]], axis=0)
    t4diag = np.einsum("p,pm->m", dtheta ** 4, derivatives[8])
    t4semidiagx = np.sum([dtheta[d[0]] ** 3 * dtheta[d[1]] * d[2] for d in derivatives[9]], axis=0)
    t4semidiagy = np.sum([dtheta[d[0]] * dtheta[d[1]] ** 3 * d[2] for d in derivatives[10]], axis=0)
    t4semidiagx2 = np.sum([dtheta[d[0]] ** 2 * dtheta[d[1]] * dtheta[d[2]] * d[3] for d in derivatives[11]], axis=0)
    t4semidiagy2 = np.sum([dtheta[d[0]] * dtheta[d[1]] ** 2 * dtheta[d[2]] * d[3] for d in derivatives[12]], axis=0)
    t4semidiagz2 = np.sum([dtheta[d[0]] * dtheta[d[1]] * dtheta[d[2]] ** 2 * d[3] for d in derivatives[13]], axis=0)
    t4nondiag = np.sum(
        [dtheta[d[0]] * dtheta[d[1]] * dtheta[d[2]] * dtheta[d[3]] * d[4] for d in derivatives[14]], axis=0
    )
    allPS = derivatives[0] + t1
    if order > 1:
        allPS += 0.5 * t2diag + t2nondiag
        if order > 2:
            allPS += t3diag / 6.0 + t3semidiagx / 2.0 + t3semidiagy / 2.0 + t3nondiag
            if order > 3:
                allPS += (
                    t4diag / 24.0
                    + t4semidiagx / 6.0
                    + t4semidiagy / 6.0
                    + t4semidiagx2 / 2.0
                    + t4semidiagy2 / 2.0
                    + t4semidiagz2 / 2.0
                    + t4nondiag
                )

    return allPS


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

    # Read in the derived parameters grid and set it up for interpolation
    valueref, delta, flattenedgrid, truecrd = grid_properties(pardict)
    paramsder = np.load(os.path.join(pardict["outgrid"], "DerParams_%s.npy" % pardict["gridname"]), allow_pickle=True)
    paramsgrid, plingrid, ploopgrid = get_grids(pardict, pad=False)
    paramsinterp = interpolate.RegularGridInterpolator(truecrd, paramsgrid)

    # Compute the values at the central point
    values = [
        float(pardict["ln10^{10}A_s"]),
        float(pardict["h"]),
        float(pardict["omega_cdm"]),
        float(pardict["omega_b"]),
    ]
    Da, Hz, f, sigma8 = paramsinterp(values)[0]
    truth = [2997.92458 * Da / float(pardict["h"]), 100.0 * float(pardict["h"]) * Hz, f * sigma8]
    print(truth)

    # Extract the name of the chainfile and read it in
    burntin, bestfit, like = read_chain(pardict["fitfile"] + ".dat")
    omega_cdm = bestfit[2] * bestfit[1] ** 2 / (1.0 + fbc)
    omega_b = bestfit[2] * bestfit[1] ** 2 - omega_cdm
    Da, Hz, f, sigma8 = paramsinterp([bestfit[0], bestfit[1], omega_cdm, omega_b])[0]
    print(2997.92458 * Da / bestfit[1], 100.0 * bestfit[1] * Hz, f * sigma8)

    # Loop over the parameters in the chain and run camb to get Da, H and fsigma8
    chainvals = np.empty((len(burntin), 3))
    for i, vals in enumerate(burntin):
        if i % 1000 == 0:
            print(i)
        omega_cdm = vals[2] * vals[1] ** 2 / (1.0 + fbc)
        omega_b = vals[2] * vals[1] ** 2 - omega_cdm
        Da, Hz, f, sigma8 = paramsinterp([vals[0], vals[1], omega_cdm, omega_b])[0]
        chainvals[i, :] = 2997.92458 * Da / vals[1], 100.0 * vals[1] * Hz, f * sigma8

    figfile = pardict["fitfile"] + "_converted.pdf"

    c = ChainConsumer()
    c.add_chain(chainvals, parameters=[r"$D_{A}(z)$", r"$H(z)$", r"$f\sigma_{8}(z)$"], posterior=like, plot_point=True)
    fig = c.plotter.plot(figsize="column", filename=figfile, truth=truth)
    print(c.analysis.get_summary())
