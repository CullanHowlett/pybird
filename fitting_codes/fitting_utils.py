# This file contains useful routines that should work regardless of whether you are fitting
# with fixed or varying template, and for any number of cosmological parameters

import os
import numpy as np
import scipy as sp
from scipy.interpolate import splrep, splev
import sys
from scipy.linalg import lapack
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append("../")
from tbird.Grid import grid_properties, grid_properties_template, run_camb
from tbird.computederivs import get_grids, get_template_grids, get_PSTaylor, get_ParamsTaylor

# Wrapper around the pybird data and model evaluation
class BirdModel:
    def __init__(self, pardict, template=False):

        self.pardict = pardict
        self.template = template

        # Some constants for the EFT model
        self.k_m, self.k_nl = 0.7, 0.7
        self.eft_priors = np.array([2.0, 2.0, 4.0, 4.0, 2.0, 2.0, 2.0])
        self.priormat = np.diagflat(1.0 / self.eft_priors ** 2)

        # Get some values at the grid centre
        if self.template:
            _, _, self.Da, self.Hz, self.fN, self.sigma8, self.sigma12, self.r_d = run_camb(pardict)
            self.valueref, self.delta, self.flattenedgrid, self.truecrd = grid_properties_template(pardict, self.fN)
        else:
            self.valueref, self.delta, self.flattenedgrid, self.truecrd = grid_properties(pardict)

        self.kin, self.paramsmod, self.linmod, self.loopmod = self.load_model()

    def load_model(self):

        # Load in the model components
        if self.pardict["taylor_order"]:
            if self.template:
                paramsmod = None
                if self.pardict["do_corr"]:
                    linmod = np.load(
                        os.path.join(self.pardict["outgrid"], "DerClin_template_%s.npy" % self.pardict["gridname"]),
                        allow_pickle=True,
                    )
                    loopmod = np.load(
                        os.path.join(self.pardict["outgrid"], "DerCloop_template_%s.npy" % self.pardict["gridname"]),
                        allow_pickle=True,
                    )
                else:
                    linmod = np.load(
                        os.path.join(self.pardict["outgrid"], "DerPlin_template_%s.npy" % self.pardict["gridname"]),
                        allow_pickle=True,
                    )
                    loopmod = np.load(
                        os.path.join(self.pardict["outgrid"], "DerPloop_template_%s.npy" % self.pardict["gridname"]),
                        allow_pickle=True,
                    )
            else:
                paramsmod = np.load(
                    os.path.join(self.pardict["outgrid"], "DerParams_%s.npy" % self.pardict["gridname"]),
                    allow_pickle=True,
                )
                if self.pardict["do_corr"]:
                    linmod = np.load(
                        os.path.join(self.pardict["outgrid"], "DerClin_%s.npy" % self.pardict["gridname"]),
                        allow_pickle=True,
                    )
                    loopmod = np.load(
                        os.path.join(self.pardict["outgrid"], "DerCloop_%s.npy" % self.pardict["gridname"]),
                        allow_pickle=True,
                    )
                else:
                    linmod = np.load(
                        os.path.join(self.pardict["outgrid"], "DerPlin_%s.npy" % self.pardict["gridname"]),
                        allow_pickle=True,
                    )
                    loopmod = np.load(
                        os.path.join(self.pardict["outgrid"], "DerPloop_%s.npy" % self.pardict["gridname"]),
                        allow_pickle=True,
                    )
            kin = linmod[0][0, :, 0]
        else:
            if self.template:
                lintab, looptab = get_template_grids(self.pardict, nmult=2, nout=2, pad=False)
                paramsmod = None
                kin = lintab[..., 0, :, 0][(0,) * 3]
            else:
                paramstab, lintab, looptab = get_grids(
                    self.pardict, nmult=2, nout=2, pad=False, cf=self.pardict["do_corr"]
                )
                paramsmod = sp.interpolate.RegularGridInterpolator(self.truecrd, paramstab)
                kin = lintab[..., 0, :, 0][(0,) * len(self.pardict["freepar"])]
            linmod = sp.interpolate.RegularGridInterpolator(self.truecrd, lintab)
            loopmod = sp.interpolate.RegularGridInterpolator(self.truecrd, looptab)

        return kin, paramsmod, linmod, loopmod

    def compute_params(self, coords):

        if self.pardict["taylor_order"]:
            dtheta = np.array(coords) - self.valueref
            Params = get_ParamsTaylor(dtheta, self.paramsmod, self.pardict["taylor_order"])
        else:
            Params = self.paramsmod(coords)[0]

        return Params

    def compute_pk(self, coords):

        if self.pardict["taylor_order"]:
            dtheta = np.array(coords) - self.valueref
            Plin = get_PSTaylor(dtheta, self.linmod, self.pardict["taylor_order"])
            Ploop = get_PSTaylor(dtheta, self.loopmod, self.pardict["taylor_order"])
        else:
            Plin = self.linmod(coords)[0]
            Ploop = self.loopmod(coords)[0]
        Plin = np.swapaxes(Plin, axis1=1, axis2=2)[:, 1:, :]
        Ploop = np.swapaxes(Ploop, axis1=1, axis2=2)[:, 1:, :]

        return Plin, Ploop

    def compute_model(self, cvals, plin, ploop, x_data):

        plin0, plin2 = plin
        ploop0, ploop2 = ploop
        if self.pardict["do_corr"]:
            b1, b2, b3, b4, cct, cr1, cr2 = cvals
        else:
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
                b1 * cct / self.k_nl ** 2,
                b1 * cr1 / self.k_m ** 2,
                b1 * cr2 / self.k_m ** 2,
                cct / self.k_nl ** 2,
                cr1 / self.k_m ** 2,
                cr2 / self.k_m ** 2,
            ]
        )

        P0 = np.dot(cvals, ploop0) + plin0[0] + b1 * plin0[1] + b1 * b1 * plin0[2]
        P2 = np.dot(cvals, ploop2) + plin2[0] + b1 * plin2[1] + b1 * b1 * plin2[2]

        P0 = sp.interpolate.splev(x_data, sp.interpolate.splrep(self.kin, P0))
        P2 = sp.interpolate.splev(x_data, sp.interpolate.splrep(self.kin, P2))

        if not self.pardict["do_corr"]:
            P0 += ce1 + cemono * x_data ** 2 / self.k_m ** 2
            P2 += cequad * x_data ** 2 / self.k_m ** 2

        return np.concatenate([P0, P2])

    def compute_chi2(self, P_model, Pi, data):

        if self.pardict["do_marg"]:

            Covbi = np.dot(Pi, np.dot(data["cov_inv"], Pi.T)) + self.priormat
            Cinvbi = np.linalg.inv(Covbi)
            vectorbi = np.dot(P_model, np.dot(data["cov_inv"], Pi.T)) - np.dot(data["invcovdata"], Pi.T)
            chi2nomar = (
                np.dot(P_model, np.dot(data["cov_inv"], P_model))
                - 2.0 * np.dot(data["invcovdata"], P_model)
                + data["chi2data"]
            )
            chi2mar = -np.dot(vectorbi, np.dot(Cinvbi, vectorbi)) + np.log(np.linalg.det(Covbi))
            chi_squared = chi2nomar + chi2mar

        else:

            # Compute the chi_squared
            chi_squared = 0.0
            for i in range(2 * len(data["x_data"])):
                chi_squared += (P_model[i] - data["fit_data"][i]) * np.sum(
                    data["cov_inv"][i, 0:] * (P_model - data["fit_data"])
                )

        return chi_squared

    # Ignore names, works for both power spectrum and correlation function
    def get_Pi_for_marg(self, ploop, b1, shot_noise, x_data):

        if self.pardict["do_marg"]:

            ploop0, ploop2 = ploop

            Onel0 = np.array([np.ones(len(x_data)), np.zeros(len(x_data))])  # shot-noise mono
            kl0 = np.array([x_data, np.zeros(len(x_data))])  # k^2 mono
            kl2 = np.array([np.zeros(len(x_data)), x_data])  # k^2 quad

            Pb3 = np.array(
                [
                    splev(x_data, splrep(self.kin, ploop0[3] + b1 * ploop0[7])),
                    splev(x_data, splrep(self.kin, ploop2[3] + b1 * ploop2[7])),
                ]
            )
            Pcct = np.array(
                [
                    splev(x_data, splrep(self.kin, ploop0[15] + b1 * ploop0[12])),
                    splev(x_data, splrep(self.kin, ploop2[15] + b1 * ploop2[12])),
                ]
            )
            Pcr1 = np.array(
                [
                    splev(x_data, splrep(self.kin, ploop0[16] + b1 * ploop0[13])),
                    splev(x_data, splrep(self.kin, ploop2[16] + b1 * ploop2[13])),
                ]
            )
            Pcr2 = np.array(
                [
                    splev(x_data, splrep(self.kin, ploop0[17] + b1 * ploop0[14])),
                    splev(x_data, splrep(self.kin, ploop2[17] + b1 * ploop2[14])),
                ]
            )

            if self.pardict["do_corr"]:

                Pi = np.array(
                    [
                        Pb3,  # *b3
                        2.0 * Pcct / self.k_nl ** 2,  # *cct
                        2.0 * Pcr1 / self.k_m ** 2,  # *cr1
                        2.0 * Pcr2 / self.k_m ** 2,  # *cr2
                    ]
                )

            else:

                Pi = np.array(
                    [
                        Pb3,  # *b3
                        2.0 * Pcct / self.k_nl ** 2,  # *cct
                        2.0 * Pcr1 / self.k_m ** 2,  # *cr1
                        2.0 * Pcr2 / self.k_m ** 2,  # *cr2
                        Onel0 * shot_noise,  # *ce1
                        kl0 ** 2 / self.k_m ** 2 * shot_noise,  # *cemono
                        kl2 ** 2 / self.k_m ** 2 * shot_noise,  # *cequad
                    ]
                )

                Pi = Pi.reshape((Pi.shape[0], -1))

        else:

            Pi = None

        return Pi

    def get_components(self, coords, cvals, shotnoise=None):

        plin, ploop = self.compute_pk(coords)

        plin0, plin2 = plin
        ploop0, ploop2 = ploop
        if self.pardict["do_corr"]:
            b1, b2, b3, b4, cct, cr1, cr2 = cvals
        else:
            b1, b2, b3, b4, cct, cr1, cr2, ce1, cemono, cequad = cvals

        # the columns of the Ploop data files.
        cloop = np.array([1, b1, b2, b3, b4, b1 * b1, b1 * b2, b1 * b3, b1 * b4, b2 * b2, b2 * b4, b4 * b4])
        cvalsct = np.array(
            [
                b1 * cct / self.k_nl ** 2,
                b1 * cr1 / self.k_m ** 2,
                b1 * cr2 / self.k_m ** 2,
                cct / self.k_nl ** 2,
                cr1 / self.k_m ** 2,
                cr2 / self.k_m ** 2,
            ]
        )

        P0lin = plin0[0] + b1 * plin0[1] + b1 * b1 * plin0[2]
        P2lin = plin2[0] + b1 * plin2[1] + b1 * b1 * plin2[2]
        P0loop = np.dot(cloop, ploop0[:12, :])
        P2loop = np.dot(cloop, ploop2[:12, :])
        P0ct = np.dot(cvalsct, ploop0[12:, :])
        P2ct = np.dot(cvalsct, ploop2[12:, :])

        if self.pardict["do_corr"]:
            P0st, P2st = None, None
        else:
            P0st = ce1 * shotnoise + cemono * shotnoise * self.kin ** 2 / self.k_m ** 2
            P2st = cequad * shotnoise * self.kin ** 2 / self.k_m ** 2

        return [P0lin, P2lin], [P0loop, P2loop], [P0ct, P2ct], [P0st, P2st]


# Holds all the data in a convenient dictionary
class FittingData:
    def __init__(self, pardict, shot_noise=100.0):

        x_data, fit_data, cov, cov_inv, chi2data, invcovdata, fitmask = self.read_data(pardict)

        self.data = {
            "x_data": x_data,
            "fit_data": fit_data,
            "cov": cov,
            "cov_inv": cov_inv,
            "chi2data": chi2data,
            "invcovdata": invcovdata,
            "fitmask": fitmask,
            "shot_noise": shot_noise,
        }

    def read_pk(self, inputfile, kmin, kmax, step_size):

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
                dataframe["pk0"].values = np.concatenate(
                    (dataframe["pk0"].values, [dataframe["pk0"].values[-1]] * to_add)
                )
                dataframe["pk2"].values = np.concatenate(
                    (dataframe["pk2"].values, [dataframe["pk2"].values[-1]] * to_add)
                )
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

    def read_data(self, pardict):

        # Read in the data
        if pardict["do_corr"]:
            print(pardict["datafile"])
            data = np.array(pd.read_csv(pardict["datafile"], delim_whitespace=True, header=None))
            x_data = data[:, 0]
            fitmask = (
                np.where(np.logical_and(x_data >= pardict["xfit_min"], x_data <= pardict["xfit_max"]))[0]
            ).astype(int)
            x_data = x_data[fitmask]
            fit_data = np.concatenate([data[fitmask, 1], data[fitmask, 2]])

            # Read in, reshape and mask the covariance matrix
            cov_flat = np.array(pd.read_csv(pardict["covfile"], delim_whitespace=True, header=None))
            cov_size = np.array([cov_flat[-1, 0] + 1, cov_flat[-1, 1] + 1]).astype(int)
            cov_input = cov_flat[:, 2].reshape(cov_size)
            cov_size = (cov_size / 3).astype(int)
            cov = np.empty((2 * len(x_data), 2 * len(x_data)))
            cov[: len(x_data), : len(x_data)] = cov_input[fitmask[:, None], fitmask[None, :]]
            cov[: len(x_data), len(x_data) :] = cov_input[cov_size[0] + fitmask[:, None], fitmask[None, :]]
            cov[len(x_data) :, : len(x_data)] = cov_input[fitmask[:, None], cov_size[1] + fitmask[None, :]]
            cov[len(x_data) :, len(x_data) :] = cov_input[
                cov_size[0] + fitmask[:, None], cov_size[1] + fitmask[None, :]
            ]

        else:
            x_data, pk0, pk2 = self.read_pk(pardict["datafile"], pardict["xfit_min"], pardict["xfit_max"], 1)
            fit_data = np.concatenate([pk0, pk2])
            fitmask = None

            # Read in, reshape and mask the covariance matrix
            cov_flat = np.array(pd.read_csv(pardict["covfile"], delim_whitespace=True, header=None))
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

        chi2data = np.dot(fit_data, np.dot(cov_inv, fit_data))
        invcovdata = np.dot(fit_data, cov_inv)

        return x_data, fit_data, cov, cov_inv, chi2data, invcovdata, fitmask


def create_plot(pardict, fittingdata):

    x_data = fittingdata.data["x_data"]
    fit_data = fittingdata.data["fit_data"]
    cov = fittingdata.data["cov"]

    if pardict["do_corr"]:
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
        plt.xlim(0.0, pardict["xfit_max"] * 1.05)
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
        plt.xlim(0.0, pardict["xfit_max"] * 1.05)
        plt.xlabel(r"$k\,(h\,\mathrm{Mpc}^{-1})$", fontsize=22)
        plt.ylabel(r"$kP(k)\,(h^{-3}\,\mathrm{Mpc}^3)$", fontsize=22, labelpad=5)
    plt.tick_params(width=1.3)
    plt.tick_params("both", length=10, which="major")
    plt.tick_params("both", length=5, which="minor")
    for axis in ["top", "left", "bottom", "right"]:
        plt.gca().spines[axis].set_linewidth(1.3)
    for tick in plt.gca().xaxis.get_ticklabels():
        tick.set_fontsize(14)
    for tick in plt.gca().yaxis.get_ticklabels():
        tick.set_fontsize(14)
    plt.gca().set_autoscale_on(False)
    plt.ion()

    return plt


def update_plot(pardict, fittingdata, P_model, plt):

    x_data = fittingdata.data["x_data"]

    if pardict["do_corr"]:
        plt10 = plt.errorbar(
            x_data,
            x_data ** 2 * P_model[: len(x_data)],
            marker="None",
            color="r",
            linestyle="-",
            markeredgewidth=1.3,
            zorder=0,
        )
        plt11 = plt.errorbar(
            x_data,
            x_data ** 2 * P_model[len(x_data) :],
            marker="None",
            color="b",
            linestyle="-",
            markeredgewidth=1.3,
            zorder=0,
        )
    else:
        plt10 = plt.errorbar(
            x_data,
            x_data * P_model[: len(x_data)],
            marker="None",
            color="r",
            linestyle="-",
            markeredgewidth=1.3,
            zorder=0,
        )
        plt11 = plt.errorbar(
            x_data,
            x_data * P_model[len(x_data) :],
            marker="None",
            color="b",
            linestyle="-",
            markeredgewidth=1.3,
            zorder=0,
        )

    plt.pause(0.005)
    if plt10 is not None:
        plt10.remove()
    if plt11 is not None:
        plt11.remove()


def format_pardict(pardict):

    pardict["do_corr"] = int(pardict["do_corr"])
    pardict["do_marg"] = int(pardict["do_marg"])
    pardict["taylor_order"] = int(pardict["taylor_order"])
    pardict["xfit_min"] = float(pardict["xfit_min"])
    pardict["xfit_max"] = float(pardict["xfit_max"])
    pardict["order"] = int(pardict["order"])
    pardict["template_order"] = int(pardict["template_order"])

    return pardict


def do_optimization(func, start, birdmodel, fittingdata, plt):

    from scipy.optimize import basinhopping

    result = basinhopping(
        func,
        start,
        niter_success=10,
        niter=100,
        stepsize=0.1,
        minimizer_kwargs={
            "args": (birdmodel, fittingdata, plt, 0),
            "method": "Nelder-Mead",
            "tol": 1.0e-3,
            "options": {"maxiter": 40000},
        },
    )
    print("#-------------- Best-fit----------------")
    print(result)

    return result


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
