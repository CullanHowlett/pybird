import os
from math import *
import numpy as np
import scipy as sp
import sys

sys.path.append("/Volumes/Work/UQ/DESI/cBIRD/pybird/")
from numpy import linalg
import emcee
from numpy.linalg import lapack_lite
from scipy.linalg import lapack
import pandas as pd
import matplotlib as mpl

mpl.use("Qt5Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as tk
import matplotlib.font_manager
from matplotlib import gridspec
import pybird
import configobj as cfg
from chainconsumer import ChainConsumer

mpl.rc("text", usetex=True)
mpl.rcParams["text.latex.preamble"] = [
    r"\usepackage{siunitx}",  # i need upright \micro symbols, but you need...
    r"\sisetup{detect-all}",  # ...this to force siunitx to actually use your fonts
    r"\usepackage{ClearSans}",  # set the normal font here
    r"\usepackage{sansmath}",  # load up the sansmath so that math -> helvet
    r"\sansmath",  # <- tricky! -- gotta actually tell tex to use!
]


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


# Converts Pk to Xi. Taken from Barry
class PowerToCorrelationGauss:
    """ A pk2xi implementation using manual numeric integration with Gaussian dampening factor
    """

    def __init__(self, ks, interpolateDetail=2, a=0.25, ell=0):
        self.ell = ell
        self.ks = ks
        self.ks2 = np.logspace(
            np.log(1.001 * np.min(ks)), np.log(0.999 * np.max(ks)), interpolateDetail * ks.size, base=np.e
        )
        self.precomp = (
            self.ks2 * np.exp(-self.ks2 * self.ks2 * a * a) / (2 * np.pi * np.pi)
        )  # Precomp a bunch of things

    def __call__(self, ks, pks, ss):
        pks2 = sp.interpolate.interp1d(ks, pks, kind="linear")(self.ks2)
        # Set up output array
        xis = np.zeros(ss.size)

        # Precompute k^2 and gauss (note missing a ks factor below because integrating in log space)
        kkpks = self.precomp * pks2

        # Iterate over all values in desired output array of distances (s)
        for i, s in enumerate(ss):
            z = self.ks2 * s
            if self.ell == 0:
                bessel = np.sin(z) / s
            elif self.ell == 2:
                bessel = (1.0 - 3.0 / z ** 2) * np.sin(z) / s + 3.0 * np.cos(z) / (z * s)
            elif self.ell == 4:
                bessel = (105.0 / z ** 4 - 45.0 / z ** 2 + 1.0) * np.sin(z) / s - (105.0 / z ** 2 - 10.0) * np.cos(
                    z
                ) / (z * s)
            else:
                bessel = spherical_jn(self.ell, z)
            integrand = kkpks * bessel
            xis[i] = sp.integrate.trapz(integrand, self.ks2)

        return xis


def read_pk(inputfile, kmin, kmax, step_size):

    dataframe = pd.read_csv(
        inputfile,
        comment="#",
        skiprows=10,
        delim_whitespace=True,
        names=["k", "kmean", "pk0", "pk2", "pk4", "nk", "shot"],
    )
    k = dataframe["k"].values
    if step_size == 1:
        k_rebinned = k
        pk0_rebinned = dataframe["pk0"].values
        pk2_rebinned = dataframe["pk2"].values
    else:
        add = k.size % step_size
        weight = dataframe["nk"].values
        if add:
            to_add = step_size - add
            k = np.concatenate((k, [k[-1]] * to_add))
            pk = np.concatenate((pk, [pk[-1]] * to_add))
            weight = np.concatenate((weight, [0] * to_add))
        k = k.reshape((-1, step_size))
        pk0 = (dataframe["pk0"].values).reshape((-1, step_size))
        pk2 = (dataframe["pk2"].values).reshape((-1, step_size))
        weight = weight.reshape((-1, step_size))
        # Take the average of every group of step_size rows to rebin
        k_rebinned = np.average(k, axis=1)
        pk0_rebinned = np.average(pk0, axis=1, weights=weight)
        pk2_rebinned = np.average(pk2, axis=1, weights=weight)

    mask = (k_rebinned >= kmin) & (k_rebinned <= kmax)
    return k_rebinned[mask], pk0_rebinned[mask], pk2_rebinned[mask]


def lnpost(params, print_flag):

    # This returns the posterior distribution which is given by the log prior plus the log likelihood
    prior = lnprior(params, print_flag)
    if not np.isfinite(prior):
        return -np.inf
    like = lnlike(params, print_flag)
    return prior + like


def lnprior(params, print_flag):

    # Here we define the prior for all the parameters. We'll ignore the constants as they
    # cancel out when subtracting the log posteriors
    b3, cct, cr1, cr2, ce1, cemono, cequad = params

    # Gaussian prior for b3 of width 2 centred on 0
    b3_prior = -0.5 * 0.25 * b3 ** 2

    # Gaussian prior for cct of width 2 centred on 0
    cct_prior = -0.5 * 0.25 * cct ** 2

    # Gaussian prior for cr1 of width 4 centred on 0
    cr1_prior = -0.5 * 0.0625 * cr1 ** 2

    # Gaussian prior for cr1 of width 4 centred on 0
    cr2_prior = -0.5 * 0.0625 * cr2 ** 2

    # Gaussian prior for ce1 of width 1 centred on 0
    ce1_prior = -0.5 * 0.25 * ce1 ** 2

    # Gaussian prior for cemono of width 2 centred on 0
    cemono_prior = -0.5 * 0.25 * cemono ** 2

    # Gaussian prior for cequad of width 2 centred on 0
    cequad_prior = -0.5 * 0.25 * cequad ** 2

    return b3_prior + cct_prior + cr1_prior + cr2_prior + ce1_prior + cemono_prior + cequad_prior


def lnlike(params, print_flag):

    b3, cct, cr1, cr2, ce1, cemono, cequad = params

    # Compute the model power spectrum given the bias and EFT terms
    bs = np.array(
        [
            b1,
            (c2 + c4) / np.sqrt(2.0),
            b3,
            (c2 - c4) / np.sqrt(2.0),
            cct,
            cr1,
            cr2,
            ce1 * shot_noise,
            cemono * shot_noise,
            cequad * shot_noise,
        ]
    )
    P_model = computePS(bs, Plin, Ploop, kfull)
    if do_corr:
        P_model_noresum = computePS(bs, Plin_noresum, Ploop_noresum, kfull)

    if do_corr:
        P_model = np.concatenate(
            [
                P_model[: len(kfull)][kfull <= kcutoff],
                P_model_noresum[: len(kfull)][kfull > kcutoff],
                P_model[len(kfull) :][kfull <= kcutoff],
                P_model_noresum[len(kfull) :][kfull > kcutoff],
            ]
        )
        P_model = np.concatenate(
            [pk2xi_0(kfull, P_model[: len(kfull)], x_data), pk2xi_2(kfull, P_model[len(kfull) :], x_data)]
        )

    # Compute the chi_squared
    chi_squared = 0.0
    for i in range(2 * len(x_data)):
        chi_squared += (P_model[i] - fit_data[i]) * np.sum(cov_inv[i, 0:] * (P_model - fit_data))

    if print_flag and (np.random.rand() < 0.1):
        print(params, chi_squared)

    return -0.5 * chi_squared


def get_PSTaylor(dtheta, derivatives):
    # Shape of dtheta: number of free parameters
    # Shape of derivatives: tuple up to third derivative where each element has shape (num free par, multipoles, lenk, columns)
    t1 = np.einsum("p,pmkb->mkb", dtheta, derivatives[1])
    t2diag = np.einsum("p,pmkb->mkb", dtheta ** 2, derivatives[2])
    t2nondiag = np.sum([dtheta[d[0]] * dtheta[d[1]] * d[2] for d in derivatives[3]], axis=0)
    t3diag = np.einsum("p,pmkb->mkb", dtheta ** 3, derivatives[4])
    t3semidiagx = np.sum([dtheta[d[0]] ** 2 * dtheta[d[1]] * d[2] for d in derivatives[5]], axis=0)
    t3semidiagy = np.sum([dtheta[d[0]] * dtheta[d[1]] ** 2 * d[2] for d in derivatives[6]], axis=0)
    t3nondiag = np.sum([dtheta[d[0]] * dtheta[d[1]] * dtheta[d[2]] * d[3] for d in derivatives[7]], axis=0)
    t4diag = np.einsum("p,pmkb->mkb", dtheta ** 4, derivatives[8])
    t4semidiagx = np.sum([dtheta[d[0]] ** 3 * dtheta[d[1]] * d[2] for d in derivatives[9]], axis=0)
    t4semidiagy = np.sum([dtheta[d[0]] * dtheta[d[1]] ** 3 * d[2] for d in derivatives[10]], axis=0)
    t4semidiagx2 = np.sum([dtheta[d[0]] ** 2 * dtheta[d[1]] * dtheta[d[2]] * d[3] for d in derivatives[11]], axis=0)
    t4semidiagy2 = np.sum([dtheta[d[0]] * dtheta[d[1]] ** 2 * dtheta[d[2]] * d[3] for d in derivatives[12]], axis=0)
    t4semidiagz2 = np.sum([dtheta[d[0]] * dtheta[d[1]] * dtheta[d[2]] ** 2 * d[3] for d in derivatives[13]], axis=0)
    t4nondiag = np.sum(
        [dtheta[d[0]] * dtheta[d[1]] * dtheta[d[2]] * dtheta[d[3]] * d[4] for d in derivatives[14]], axis=0
    )
    allPS = (
        derivatives[0]
        + t1
        + 0.5 * t2diag
        + t2nondiag
        + t3diag / 6.0
        + t3semidiagx / 2.0
        + t3semidiagy / 2.0
        + t3nondiag
    )
    # + t4diag / 24.0  + t4semidiagx / 6.0 + t4semidiagy / 6.0 + t4semidiagx2 / 2.0 + t4semidiagy2 / 2.0 + t4semidiagz2 / 2.0 + t4nondiag)
    return allPS


def computePS(cvals, plin, ploop, kvals):
    plin0, plin2 = plin
    ploop0, ploop2 = ploop[:, :18, :]
    b1, b2, b3, b4, cct, cr1, cr2, ce1, cemono, cequad = cvals

    # the columns of the Ploop data files.
    cvals = np.array(
        [
            1,
            b1,
            b2,
            b3,
            b4,
            b1 * b1,
            b1 * b2,
            b1 * b3,
            b1 * b4,
            b2 * b2,
            b2 * b4,
            b4 * b4,
            b1 * cct / k_nl ** 2,
            b1 * cr1 / k_m ** 2,
            b1 * cr2 / k_m ** 2,
            cct / k_nl ** 2,
            cr1 / k_m ** 2,
            cr2 / k_m ** 2,
        ]
    )

    P0 = np.dot(cvals, ploop0) + plin0[0] + b1 * plin0[1] + b1 * b1 * plin0[2]
    P2 = np.dot(cvals, ploop2) + plin2[0] + b1 * plin2[1] + b1 * b1 * plin2[2]

    if do_corr:
        P0 += ce1 + cemono * kvals ** 2 / k_m ** 2
        P2 += cequad * kvals ** 2 / k_m ** 2
    else:
        P0 = sp.interpolate.splev(x_data, sp.interpolate.splrep(kvals, P0)) + ce1 + cemono * x_data ** 2 / k_m ** 2
        P2 = sp.interpolate.splev(x_data, sp.interpolate.splrep(kvals, P2)) + cequad * x_data ** 2 / k_m ** 2

    return np.concatenate([P0, P2])


def get_Pi_for_marg(ploop, b1, shot_noise, kvals):

    ploop0, ploop2 = ploop[:, :18, :]

    Pb3 = np.array(
        [
            sp.interpolate.splev(x_data, sp.interpolate.splrep(kvals, ploop0[3] + b1 * ploop0[7])),
            sp.interpolate.splev(x_data, sp.interpolate.splrep(kvals, ploop2[3] + b1 * ploop2[7])),
        ]
    )
    Pcct = np.array(
        [
            sp.interpolate.splev(x_data, sp.interpolate.splrep(kvals, ploop0[15] + b1 * ploop0[12])),
            sp.interpolate.splev(x_data, sp.interpolate.splrep(kvals, ploop2[15] + b1 * ploop2[12])),
        ]
    )
    Pcr1 = np.array(
        [
            sp.interpolate.splev(x_data, sp.interpolate.splrep(kvals, ploop0[16] + b1 * ploop0[13])),
            sp.interpolate.splev(x_data, sp.interpolate.splrep(kvals, ploop2[16] + b1 * ploop2[13])),
        ]
    )
    Pcr2 = np.array(
        [
            sp.interpolate.splev(x_data, sp.interpolate.splrep(kvals, ploop0[17] + b1 * ploop0[14])),
            sp.interpolate.splev(x_data, sp.interpolate.splrep(kvals, ploop2[17] + b1 * ploop2[14])),
        ]
    )

    Pi = np.array(
        [
            Pb3,  # *b3
            2.0 * Pcct / k_nl ** 2,  # *cct
            2.0 * Pcr1 / k_m ** 2,  # *cr1
            2.0 * Pcr2 / k_m ** 2,  # *cr2
            Onel0 * shot_noise,  # *ce1
            kl0 ** 2 / k_m ** 2 * shot_noise,  # *cemono
            kl2 ** 2 / k_m ** 2 * shot_noise,  # *cequad
        ]
    )

    Pi = Pi.reshape((Pi.shape[0], -1))

    return Pi


def get_Xi_for_marg(ploop, ploop_noresum, b1, shot_noise, kvals):

    ploop0, ploop2 = ploop[:, :18, :]
    ploop0_noresum, ploop2_noresum = ploop_noresum[:, :18, :]

    ploop0 = np.concatenate([ploop0[:, kvals <= kcutoff], ploop0_noresum[:, kvals > kcutoff]], axis=1)
    ploop2 = np.concatenate([ploop2[:, kvals <= kcutoff], ploop2_noresum[:, kvals > kcutoff]], axis=1)

    Xib3 = np.array(
        [pk2xi_0(kvals, ploop0[3] + b1 * ploop0[7], x_data), pk2xi_2(kvals, ploop2[3] + b1 * ploop2[7], x_data)]
    )
    Xicct = np.array(
        [pk2xi_0(kvals, ploop0[15] + b1 * ploop0[12], x_data), pk2xi_2(kvals, ploop2[15] + b1 * ploop2[12], x_data)]
    )
    Xicr1 = np.array(
        [pk2xi_0(kvals, ploop0[16] + b1 * ploop0[13], x_data), pk2xi_2(kvals, ploop2[16] + b1 * ploop2[13], x_data)]
    )
    Xicr2 = np.array(
        [pk2xi_0(kvals, ploop0[17] + b1 * ploop0[14], x_data), pk2xi_2(kvals, ploop2[17] + b1 * ploop2[14], x_data)]
    )

    Xi = np.array(
        [
            Xib3,  # *b3
            2.0 * Xicct / k_nl ** 2,  # *cct
            2.0 * Xicr1 / k_m ** 2,  # *cr1
            2.0 * Xicr2 / k_m ** 2,  # *cr2
            Onel0 * shot_noise,  # *ce1
            kl0 ** 2 / k_m ** 2 * shot_noise,  # *cemono
            kl2 ** 2 / k_m ** 2 * shot_noise,  # *cequad
        ]
    )

    Xi = Xi.reshape((Xi.shape[0], -1))

    return Xi


# Tested, good
def get_grids(mydir, name, nmult=2, nout=2):
    # Coordinates have shape (len(freepar), 2 * order_1 + 1, ..., 2 * order_n + 1)
    # order_i is the number of points away from the origin for parameter i
    # The len(freepar) sub-arrays are the outputs of a meshgrid, which I feed to findiff
    # Power spectra needs to be reshaped.
    shapecrd = np.concatenate([[len(valueref)], np.full(4, 2 * gridorder + 1)])
    plin = np.load(os.path.join(mydir, "TablePlin_%s.npy" % name))
    plin = plin.reshape(
        (*shapecrd[1:], nmult, plin.shape[-2] // nmult, plin.shape[-1])
    )  # This won't work with Python 2 :(
    ploop = np.load(os.path.join(mydir, "TablePloop_%s.npy" % name))
    ploop = ploop.reshape(
        (*shapecrd[1:], nmult, ploop.shape[-2] // nmult, ploop.shape[-1])
    )  # This won't work with Python 2 :(
    # The output is not concatenated for multipoles since we remove the hexadecapole
    return plin[..., :nout, :, :], ploop[..., :nout, :, :]


if __name__ == "__main__":

    # Fitting parameters
    print_flag = 1
    do_corr = 0  # Whether to fit the power spectrum or correlation function
    do_taylor = 1
    zeff = 0.9873
    shot_noise = 309.210197  # Taken from the header of the data power spectrum file.
    xfit_min = 0.0
    xfit_max = 0.3

    # Filenames
    configfile = "/Volumes/Work/UQ/DESI/cBIRD/input_files/tbird_UNIT.ini"
    gridpath = "/Volumes/Work/UQ/DESI/cBIRD/UNIT_output_files/GridsEFT/"
    gridname = "camb-z0p9873-A_s-h-omega_cdm-omega_b"
    chainfiles = [
        "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_HandShake/chain_pk_UNIT_HODsnap97_ELGv1_k0p00-0p30_3order.dat"
    ]
    figfile = [
        "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_HandShake/chain_pk_UNIT_HODsnap97_ELGv1_k0p00-0p30_3order.pdf"
    ]

    c = ChainConsumer()

    bestfits = []
    # names=[r"$\mathrm{Marg:}\,k_{\mathrm{max}}=0.20h\,\mathrm{Mpc}^{-1}$",r"$\mathrm{Marg:}\,k_{\mathrm{max}}=0.25h\,\mathrm{Mpc}^{-1}$",r"$\mathrm{Marg:}\,k_{\mathrm{max}}=0.30h\,\mathrm{Mpc}^{-1}$"]
    names = [r"$\mathrm{Marg:\,4^{th}\,Order\,Taylor}$"]
    paramnames = [r"$A_{s}\times 10^{-9}$", r"$h$", r"$\Omega_{m}$", r"$b_{1}$"]
    for chaini, chainfile in enumerate(chainfiles):

        burntin, bestfit, like = read_chain(chainfile, burnlimitup=20000)
        burntin[:, 0] = np.exp(burntin[:, 0]) / 1.0e1
        c.add_chain(burntin[:, :4], parameters=paramnames, name=names[chaini], posterior=like, plot_point=True)
        bestfits.append(bestfit)

    print(bestfits[0])
    fig = c.plotter.plot(figsize="column", filename=figfile, truth=[2.142, 0.6774, 0.3072218093])
    print(c.analysis.get_summary())

    # Get the best-fit model. Because we analytically marginalised over the higher-order bias parameters
    # we don't have values for those. So we fix the parameters we did fit to their best-fit values
    # then fit the other bias parameters separately.

    ln10As, h, omega_m, b1, c2, c4 = bestfits[0]

    # Read in the data
    if do_corr:
        datafile = "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_HandShake/xil_rsd.txt"
        data = np.array(pd.read_csv(datafile, delim_whitespace=True, header=None))
        x_data = data[:, 0]
        fitmask = (np.where(np.logical_and(x_data >= xfit_min, x_data <= xfit_max))[0]).astype(int)
        x_data = x_data[fitmask]
        fit_data = np.concatenate([data[fitmask, 1], data[fitmask, 2]])
        print(x_data, fitmask)

        # Read in, reshape and mask the covariance matrix
        covfile = "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_HandShake/cov_matrix_rsd.txt"
        cov_flat = np.array(pd.read_csv(covfile, delim_whitespace=True, header=None))
        cov_size = np.array([cov_flat[-1, 0] + 1, cov_flat[-1, 1] + 1]).astype(int)
        cov_input = cov_flat[:, 2].reshape(cov_size)
        cov_size = (cov_size / 3).astype(int)
        cov = np.empty((2 * len(x_data), 2 * len(x_data)))
        cov[: len(x_data), : len(x_data)] = cov_input[fitmask[:, None], fitmask[None, :]]
        cov[: len(x_data), len(x_data) :] = cov_input[cov_size[0] + fitmask[:, None], fitmask[None, :]]
        cov[len(x_data) :, : len(x_data)] = cov_input[fitmask[:, None], cov_size[1] + fitmask[None, :]]
        cov[len(x_data) :, len(x_data) :] = cov_input[cov_size[0] + fitmask[:, None], cov_size[1] + fitmask[None, :]]

    else:
        datafile = (
            "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_HandShake/Power_Spectrum_UNIT_HODsnap97_ELGv1_redshift.txt"
        )
        x_data, pk0, pk2 = read_pk(datafile, xfit_min, xfit_max, 1)
        fit_data = np.concatenate([pk0, pk2])

        # Read in, reshape and mask the covariance matrix
        covfile = "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_HandShake/cov_matrix_pk-EZmocks_rsd.txt"
        cov_flat = np.array(pd.read_csv(covfile, delim_whitespace=True, header=None))
        cov_size = np.array([cov_flat[-1, 0] + 1, cov_flat[-1, 1] + 1]).astype(int)
        cov_input = cov_flat[:, 2].reshape(cov_size)
        cov_size = (cov_size / 3).astype(int)
        cov = np.empty((2 * len(x_data), 2 * len(x_data)))
        cov[: len(x_data), : len(x_data)] = cov_input[: len(x_data), : len(x_data)]
        cov[: len(x_data), len(x_data) :] = cov_input[cov_size[0] : cov_size[0] + len(x_data), : len(x_data)]
        cov[len(x_data) :, : len(x_data)] = cov_input[: len(x_data), cov_size[1] : cov_size[1] + len(x_data)]
        cov[len(x_data) :, len(x_data) :] = cov_input[
            cov_size[0] : cov_size[0] + len(x_data), cov_size[1] : cov_size[1] + len(x_data)
        ]

    # Invert the covariance matrix
    identity = np.eye(2 * len(x_data))
    cov_lu, pivots, cov_inv, info = lapack.dgesv(cov, identity)

    # Set up the model. Based on Likelihood_taylor in the likelihood_class.py file
    # Some things used to create the grids and models
    gridorder = 4
    k_m, k_nl = 0.7, 0.7
    eft_priors = np.array([2.0, 2.0, 4.0, 4.0, 2.0, 2.0, 2.0])
    priormat = np.diagflat(1.0 / eft_priors ** 2)
    freepar = ["ln10^{10}A_s", "h", "omega_cdm", "omega_b"]
    dx = np.array([0.06, 0.02, 0.04, 0.02])
    parref = cfg.ConfigObj(configfile)
    valueref = np.array([float(parref[k]) for k in freepar])
    delta = dx * valueref
    truecrd = [valueref[l] + delta[l] * np.arange(-gridorder, gridorder + 1) for l in range(len(freepar))]

    fbc = valueref[3] / valueref[2]
    omega_cdm = omega_m * h ** 2 / (1.0 + fbc)
    omega_b = omega_m * h ** 2 - omega_cdm

    # Load in the model derivatives
    if do_taylor:
        linder = np.load(os.path.join(gridpath, "DerPlin_%s.npy" % gridname), allow_pickle=True)
        loopder = np.load(os.path.join(gridpath, "DerPloop_%s.npy" % gridname), allow_pickle=True)
        kin = linder[0][0, :, 0]
    else:
        lintab, looptab = get_grids(gridpath, gridname, nmult=2, nout=2)
        lininterp = sp.interpolate.RegularGridInterpolator(truecrd, lintab)
        loopinterp = sp.interpolate.RegularGridInterpolator(truecrd, looptab)
        kin = lintab[..., 0, :, 0]

    if do_corr:
        if do_taylor:
            linder_noresum = np.load(os.path.join(gridpath, "DerPlin_%s_noresum.npy" % gridname), allow_pickle=True)
            loopder_noresum = np.load(os.path.join(gridpath, "DerPloop_%s_noresum.npy" % gridname), allow_pickle=True)
        else:
            lintab_noresum, looptab_noresum = get_grids(gridpath, gridname + "_noresum", nmult=2, nout=2)
            lininterp_noresum = sp.interpolate.RegularGridInterpolator(truecrd, lintab_noresum)
            loopinterp_noresum = sp.interpolate.RegularGridInterpolator(truecrd, looptab_noresum)

        afac = 1.0
        kcutoff = 0.40
        pk2xi_0 = PowerToCorrelationGauss(kin, a=afac, ell=0)
        pk2xi_2 = PowerToCorrelationGauss(kin, a=afac, ell=2)

        # These are now the integrals of 0, 1 and k^2 times the relevant bessel function, computed analytically for a=0.25.
        # These do not converge unless we include the exponential damping factors used in the conversion from Pk to Xi in our class.
        Onel0 = np.array(
            [
                np.exp(-(x_data ** 2) / (4.0 * afac ** 2)) / (8.0 * (afac ** 2 * np.pi) ** (3.0 / 2.0)),
                np.zeros(len(x_data)),
            ]
        )  # shot-noise mono
        kl0 = np.array(
            [
                np.exp(-(x_data ** 2) / (4.0 * afac ** 2))
                * (x_data ** 2 - 6.0 * afac ** 2)
                / (32.0 * (afac ** (14.0 / 3.0) * np.pi) ** (3.0 / 2.0)),
                np.zeros(len(x_data)),
            ]
        )  # k^2 mono
        kl2 = np.array(
            [
                np.zeros(len(x_data)),
                np.exp(-(x_data ** 2) / (4.0 * afac ** 2))
                * x_data ** 2
                / (32.0 * (afac ** (14.0 / 3.0) * np.pi) ** (3.0 / 2.0)),
            ]
        )  # k^2 quad
    else:
        fitmask = np.where(np.logical_and(kin >= xfit_min, kin <= xfit_max))[0]
        Onel0 = np.array([np.ones(len(x_data)), np.zeros(len(x_data))])  # shot-noise mono
        kl0 = np.array([x_data, np.zeros(len(x_data))])  # k^2 mono
        kl2 = np.array([np.zeros(len(x_data)), x_data])  # k^2 quad

    # Compute the linear and loop terms of the model using the precomputed derivatives and a Taylor expansion
    dtheta = [ln10As, h, omega_cdm, omega_b] - valueref
    if do_taylor:
        Plin = get_PSTaylor(dtheta, linder)
        Ploop = get_PSTaylor(dtheta, loopder)
    else:
        Plin = lininterp(bestfits[0])[0]
        Ploop = loopinterp(bestfits[0])[0]
    kfull = Plin[0, :, 0]
    Plin = np.swapaxes(Plin, axis1=1, axis2=2)[:, 1:, :]
    Ploop = np.swapaxes(Ploop, axis1=1, axis2=2)[:, 1:, :]

    if do_corr:
        if do_taylor:
            Plin_noresum = get_PSTaylor(dtheta, linder_noresum)
            Ploop_noresum = get_PSTaylor(dtheta, loopder_noresum)
        else:
            Plin_noresum = lininterp_noresum(bestfits[0])[0]
            Ploop_noresum = loopinterp_noresum(bestfits[0])[0]
        Plin_noresum = np.swapaxes(Plin_noresum, axis1=1, axis2=2)[:, 1:, :]
        Ploop_noresum = np.swapaxes(Ploop_noresum, axis1=1, axis2=2)[:, 1:, :]

    start = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    nll = lambda *args: -lnpost(*args)
    result = sp.optimize.basinhopping(
        nll,
        start,
        niter_success=10,
        niter=100,
        stepsize=0.05,
        minimizer_kwargs={"args": (print_flag), "method": "Nelder-Mead", "tol": 1.0e-4, "options": {"maxiter": 40000}},
    )
    print("#-------------- Best-fit----------------")
    print(result)

    b3, cct, cr1, cr2, ce1, cemono, cequad = result["x"]

    # Compute the model power spectrum given the bias and EFT terms
    bs = np.array(
        [
            b1,
            (c2 + c4) / np.sqrt(2.0),
            b3,
            (c2 - c4) / np.sqrt(2.0),
            cct,
            cr1,
            cr2,
            ce1 * shot_noise,
            cemono * shot_noise,
            cequad * shot_noise,
        ]
    )
    P_model = computePS(bs, Plin, Ploop, kfull)
    if do_corr:
        P_model_noresum = computePS(bs, Plin_noresum, Ploop_noresum, kfull)

    if do_corr:
        P_model = np.concatenate(
            [
                P_model[: len(kfull)][kfull <= kcutoff],
                P_model_noresum[: len(kfull)][kfull > kcutoff],
                P_model[len(kfull) :][kfull <= kcutoff],
                P_model_noresum[len(kfull) :][kfull > kcutoff],
            ]
        )
        P_model = np.concatenate(
            [pk2xi_0(kfull, P_model[: len(kfull)], x_data), pk2xi_2(kfull, P_model[len(kfull) :], x_data)]
        )

    np.savetxt(
        "chain_pk_UNIT_HODsnap97_ELGv1_k0p00-0p30_3order_bestfit.dat",
        np.c_[x_data, P_model[: len(x_data)], P_model[len(x_data) :]],
        fmt="%12.6lf  %12.6lf  %12.6lf",
        header="k_cen     P0(k)     P2(k)",
    )

    fig = plt.figure(0)
    ax = fig.add_axes([0.15, 0.14, 0.84, 0.85])
    if do_corr:
        plt.errorbar(
            x_data,
            x_data ** 2 * fit_data[: len(x_data)],
            yerr=x_data ** 2 * np.sqrt(cov[np.diag_indices(2 * len(x_data))][: len(x_data)]),
            marker="o",
            markerfacecolor="r",
            markeredgecolor="k",
            color="r",
            linestyle="None",
            markeredgewidth=1.3,
            zorder=5,
        )
        plt.errorbar(
            x_data,
            x_data ** 2 * fit_data[len(x_data) : 2 * len(x_data)],
            yerr=x_data ** 2 * np.sqrt(cov[np.diag_indices(2 * len(x_data))][len(x_data) :]),
            marker="o",
            markerfacecolor="b",
            markeredgecolor="k",
            color="b",
            linestyle="None",
            markeredgewidth=1.3,
            zorder=5,
        )
        plt.errorbar(
            x_data,
            x_data ** 2 * P_model[: len(x_data)],
            marker="None",
            color="r",
            linestyle="-",
            markeredgewidth=1.3,
            zorder=0,
        )
        plt.errorbar(
            x_data,
            x_data ** 2 * P_model[len(x_data) :],
            marker="None",
            color="b",
            linestyle="-",
            markeredgewidth=1.3,
            zorder=0,
        )
        plt.xlim(0.0, xfit_max * 1.05)
        plt.xlabel(r"$s\,(h^{-1}\,\mathrm{Mpc})$", fontsize=22)
        plt.ylabel(r"$s^{2}\xi(s)$", fontsize=22, labelpad=5)
    else:
        plt.errorbar(
            x_data,
            x_data * fit_data[: len(x_data)],
            yerr=x_data * np.sqrt(cov[np.diag_indices(2 * len(x_data))][: len(x_data)]),
            marker="o",
            markerfacecolor="r",
            markeredgecolor="k",
            color="r",
            linestyle="None",
            markeredgewidth=1.3,
            zorder=5,
        )
        plt.errorbar(
            x_data,
            x_data * fit_data[len(x_data) : 2 * len(x_data)],
            yerr=x_data * np.sqrt(cov[np.diag_indices(2 * len(x_data))][len(x_data) :]),
            marker="o",
            markerfacecolor="b",
            markeredgecolor="k",
            color="b",
            linestyle="None",
            markeredgewidth=1.3,
            zorder=5,
        )
        plt.errorbar(
            x_data,
            x_data * P_model[: len(x_data)],
            marker="None",
            color="r",
            linestyle="-",
            markeredgewidth=1.3,
            zorder=0,
        )
        plt.errorbar(
            x_data,
            x_data * P_model[len(x_data) :],
            marker="None",
            color="b",
            linestyle="-",
            markeredgewidth=1.3,
            zorder=0,
        )
        plt.xlim(0.0, xfit_max * 1.05)
        plt.xlabel(r"$k\,(h\,\mathrm{Mpc}^{-1})$", fontsize=22)
        plt.ylabel(r"$kP(k)\,(h^{-3}\,\mathrm{Mpc}^3)$", fontsize=22, labelpad=5)
    ax.tick_params(width=1.3)
    ax.tick_params("both", length=10, which="major")
    ax.tick_params("both", length=5, which="minor")
    for axis in ["top", "left", "bottom", "right"]:
        plt.gca().spines[axis].set_linewidth(1.3)
    for tick in plt.gca().xaxis.get_ticklabels():
        tick.set_fontsize(14)
    for tick in plt.gca().yaxis.get_ticklabels():
        tick.set_fontsize(14)

    plt.show()
