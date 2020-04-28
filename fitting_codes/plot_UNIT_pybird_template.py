import numpy as np
import sys
import matplotlib.pyplot as plt
from configobj import ConfigObj
from chainconsumer import ChainConsumer

sys.path.append("../")
from tbird.Grid import run_camb


def read_chain(chainfile, burnlimitlow=1000, burnlimitup=20000):

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
        ind2 = np.where(np.logical_and(np.asarray(x) >= burnlimitlow, np.asarray(x) <= burnlimitup))[0]
        for k in range(len(ind2 + 1)):
            burntin.append(samples[ind[ind2[k]]])
            burntlike.append(like[ind[ind2[k]]])
        nburntin += len(ind2)
    burntin = np.array(burntin)
    burntlike = np.array(burntlike)

    return burntin, samples[bestid], burntlike


if __name__ == "__main__":

    # First read in the config file and compute Da_fid, Hz_fid and sigma8_fid
    configfile = sys.argv[1]
    pardict = ConfigObj(configfile)
    kin, Plin, Da_fid, Hz_fid, fN, sigma8_fid = run_camb(pardict)
    Da_fid *= 2997.92458 / float(pardict["h"])
    Hz_fid *= 100.0 * float(pardict["h"])

    # Set the chainfiles and names for each chain
    chainfiles = [
        "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_HandShake/chain_UNIT_HODsnap97_ELGv1_pk_0.00_ 0.30_template.dat"
    ]
    names = [r"$\mathrm{Template, Grid}$"]
    paramnames = [r"$\alpha_{\perp}$", r"$\alpha_{||}$", r"$f\sigma_{8}$", r"$b_{1}\sigma_{8}$"]

    # Output name for the figure
    figfile = [
        "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_HandShake/chain_UNIT_HODsnap97_ELGv1_pk_0.00_ 0.30_template.pdf"
    ]

    c = ChainConsumer()

    bestfits = []
    for chaini, chainfile in enumerate(chainfiles):

        burntin, bestfit, like = read_chain(chainfile, burnlimitup=20000)
        # burntin[:, 0] *= Da_fid
        # burntin[:, 1] = Hz_fid / burntin[:, 1]
        burntin[:, 2] *= sigma8_fid
        burntin[:, 3] *= sigma8_fid
        c.add_chain(burntin[:, :4], parameters=paramnames, name=names[chaini], posterior=like, plot_point=True)
        bestfits.append(bestfit)

    print(bestfits[0])
    fig = c.plotter.plot(figsize="column", filename=figfile, truth=[1.0, 1.0, fN * sigma8_fid])
    print(c.analysis.get_summary())
