import os
import numpy as np
import scipy as sp
from scipy.interpolate import splrep, splev
import sys
import emcee
from scipy.linalg import lapack
import pandas as pd
from configobj import ConfigObj
import matplotlib.pyplot as plt

sys.path.append("../")
from pybird import pybird
from tbird.Grid import run_camb
from tbird.computederivs import get_template_grids


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


def lnpost(params, pardict, print_flag, bird, DA_fid, H_fid):

    # This returns the posterior distribution which is given by the log prior plus the log likelihood
    prior = lnprior(params, pardict)
    if not np.isfinite(prior):
        return -np.inf
    like = lnlike(params, pardict, print_flag, bird, DA_fid, H_fid)
    return prior + like


def lnprior(params, pardict):

    # Here we define the prior for all the parameters. We'll ignore the constants as they
    # cancel out when subtracting the log posteriors
    if pardict["do_marg"]:
        alpha_perp, alpha_par, fval, b1, c2, c4 = params
    else:
        alpha_perp, alpha_par, fval, b1, c2, b3, c4, cct, cr1, cr2, ce1, cemono, cequad = params

    # Flat prior for alpha_perp
    if 0.8 < alpha_perp < 1.2:
        alpha_perp_prior = 1.0 / 0.4
    else:
        return -np.inf

    # Flat prior for alpha_par
    if 0.8 < alpha_par < 1.2:
        alpha_par_prior = 1.0 / 0.4
    else:
        return -np.inf

    # Flat prior for f
    if 0.0 < fval < 2.0:
        fval = 1.0 / 2.0
    else:
        return -np.inf

    # Flat prior for b1
    if 0.0 < b1 < 3.0:
        b1_prior = 1.0 / 3.0
    else:
        return -np.inf

    # Flat prior for c2
    if -4.0 <= c2 < 4.0:
        c2_prior = 1.0 / 8.0
    else:
        return -np.inf

    # Gaussian prior for c4
    c4_prior = -0.5 * 0.25 * c4 ** 2

    if pardict["do_marg"]:

        return c4_prior

    else:
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

        return c4_prior + b3_prior + cct_prior + cr1_prior + cr2_prior + ce1_prior + cemono_prior + cequad_prior


def lnlike(params, pardict, print_flag, bird, DA_fid, H_fid):

    if pardict["do_marg"]:
        alpha_perp, alpha_par, fval, b1, c2, c4 = params
    else:
        alpha_perp, alpha_par, fval, b1, c2, b3, c4, cct, cr1, cr2, ce1, cemono, cequad = params

    # Modify the bird by the input values
    Plin = np.swapaxes(lininterp([fval])[0], axis1=1, axis2=2)
    Ploop = np.swapaxes(loopinterp([fval])[0], axis1=1, axis2=2)

    kfull = Plin[:, 0, :]
    bird.P11l = Plin[:, 1:, :]
    bird.Ploopl = Ploop[:, 1:13, :]
    bird.Pctl = Ploop[:, 13:, :]
    bird.DA = DA_fid * alpha_perp
    bird.H = H_fid / alpha_par

    # Apply the AP shift caused by the two alpha parameters
    projection.AP(bird)
    # projection.Window(bird)

    if pardict["do_marg"]:
        bs = np.array([b1, (c2 + c4) / np.sqrt(2.0), 0.0, (c2 - c4) / np.sqrt(2.0), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        if pardict["do_corr"]:
            Pi = get_Xi_for_marg(bird, b1)
        else:
            Pi = get_Pi_for_marg(bird, b1, shot_noise)
    else:
        bs = np.array([b1, (c2 + c4) / np.sqrt(2.0), b3, (c2 - c4) / np.sqrt(2.0), cct, cr1, cr2, ce1, cemono, cequad])

    if pardict["do_corr"]:
        P_model = computePS(bs, bird)
    else:
        P_model = computePS(bs, bird)

    # Plotting. Only really for debugging, remove for production runs
    if plot_flag:
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

    if pardict["do_marg"]:

        Covbi = np.dot(Pi, np.dot(cov_inv, Pi.T)) + priormat
        Cinvbi = np.linalg.inv(Covbi)
        vectorbi = np.dot(P_model, np.dot(cov_inv, Pi.T)) - np.dot(invcovdata, Pi.T)
        chi2nomar = np.dot(P_model, np.dot(cov_inv, P_model)) - 2.0 * np.dot(invcovdata, P_model) + chi2data
        chi2mar = -np.dot(vectorbi, np.dot(Cinvbi, vectorbi)) + np.log(np.linalg.det(Covbi))
        chi_squared = chi2nomar + chi2mar
        if print_flag and (np.random.rand() < 0.1):
            print(params, chi2mar, chi2nomar, chi_squared)

    else:

        # Compute the chi_squared
        chi_squared = 0.0
        for i in range(2 * len(x_data)):
            chi_squared += (P_model[i] - fit_data[i]) * np.sum(cov_inv[i, 0:] * (P_model - fit_data))

        if print_flag and (np.random.rand() < 0.1):
            print(params, chi_squared)

    return -0.5 * chi_squared


def computePS(cvals, bird):
    plin0, plin2 = bird.P11l
    ploop0, ploop2 = bird.Ploopl
    pct0, pct2 = bird.Pctl
    b1, b2, b3, b4, cct, cr1, cr2, ce1, cemono, cequad = cvals

    # the columns of the Ploop data files.
    cloop = np.array([1, b1, b2, b3, b4, b1 * b1, b1 * b2, b1 * b3, b1 * b4, b2 * b2, b2 * b4, b4 * b4])
    cct = np.array(
        [
            b1 * cct / k_nl ** 2,
            b1 * cr1 / k_m ** 2,
            b1 * cr2 / k_m ** 2,
            cct / k_nl ** 2,
            cr1 / k_m ** 2,
            cr2 / k_m ** 2,
        ]
    )

    P0 = np.dot(cloop, ploop0) + np.dot(cct, pct0) + plin0[0] + b1 * plin0[1] + b1 * b1 * plin0[2]
    P2 = np.dot(cloop, ploop2) + np.dot(cct, pct2) + plin2[0] + b1 * plin2[1] + b1 * b1 * plin2[2]

    P0 = sp.interpolate.splev(x_data, sp.interpolate.splrep(bird.co.k, P0)) + ce1 + cemono * x_data ** 2 / k_m ** 2
    P2 = sp.interpolate.splev(x_data, sp.interpolate.splrep(bird.co.k, P2)) + cequad * x_data ** 2 / k_m ** 2

    return np.concatenate([P0, P2])


def get_Pi_for_marg(bird, b1, shot_noise):

    ploop0, ploop2 = bird.Ploopl

    Onel0 = np.array([np.ones(len(x_data)), np.zeros(len(x_data))])  # shot-noise mono
    kl0 = np.array([x_data, np.zeros(len(x_data))])  # k^2 mono
    kl2 = np.array([np.zeros(len(x_data)), x_data])  # k^2 quad

    Pb3 = np.array(
        [
            splev(x_data, splrep(bird.co.k, ploop0[3] + b1 * ploop0[7])),
            splev(x_data, splrep(bird.co.k, ploop2[3] + b1 * ploop2[7])),
        ]
    )
    Pcct = np.array(
        [
            splev(x_data, splrep(bird.co.k, ploop0[15] + b1 * ploop0[12])),
            splev(x_data, splrep(bird.co.k, ploop2[15] + b1 * ploop2[12])),
        ]
    )
    Pcr1 = np.array(
        [
            splev(x_data, splrep(bird.co.k, ploop0[16] + b1 * ploop0[13])),
            splev(x_data, splrep(bird.co.k, ploop2[16] + b1 * ploop2[13])),
        ]
    )
    Pcr2 = np.array(
        [
            splev(x_data, splrep(bird.co.k, ploop0[17] + b1 * ploop0[14])),
            splev(x_data, splrep(bird.co.k, ploop2[17] + b1 * ploop2[14])),
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


def get_Xi_for_marg(bird, b1):

    Cloop0, Cloop2 = bird.Cloopl

    Cb3 = np.array(
        [
            splev(x_data, splrep(bird.co.k, Cloop0[3] + b1 * Cloop0[7])),
            splev(x_data, splrep(bird.co.k, Cloop2[3] + b1 * Cloop2[7])),
        ]
    )
    Ccct = np.array(
        [
            splev(x_data, splrep(bird.co.k, Cloop0[15] + b1 * Cloop0[12])),
            splev(x_data, splrep(bird.co.k, Cloop2[15] + b1 * Cloop2[12])),
        ]
    )
    Ccr1 = np.array(
        [
            splev(x_data, splrep(bird.co.k, Cloop0[16] + b1 * Cloop0[13])),
            splev(x_data, splrep(bird.co.k, Cloop2[16] + b1 * Cloop2[13])),
        ]
    )
    Ccr2 = np.array(
        [
            splev(x_data, splrep(bird.co.k, Cloop0[17] + b1 * Cloop0[14])),
            splev(x_data, splrep(bird.co.k, Cloop2[17] + b1 * Cloop2[14])),
        ]
    )

    Xi = np.array(
        [Cb3, 2.0 * Ccct / k_nl ** 2, 2.0 * Ccr1 / k_m ** 2, 2.0 * Ccr2 / k_m ** 2,]  # *b3  # *cct  # *cr1  # *cr2
    )

    Xi = Xi.reshape((Xi.shape[0], -1))

    return Xi


if __name__ == "__main__":

    # Code to generate a power spectrum template at fixed cosmology using pybird, then fit the AP parameters and fsigma8
    # First read in the config file
    configfile = sys.argv[1]
    print_flag = sys.argv[2]
    plot_flag = sys.argv[2]
    pardict = ConfigObj(configfile)

    shot_noise = 309.210197  # Taken from the header of the data power spectrum file.

    pardict["do_corr"] = int(pardict["do_corr"])
    pardict["do_marg"] = int(pardict["do_marg"])
    pardict["taylor_order"] = int(pardict["taylor_order"])
    pardict["xfit_min"] = float(pardict["xfit_min"])
    pardict["xfit_max"] = float(pardict["xfit_max"])

    # Read in the data
    if pardict["do_corr"]:
        print(pardict["datafile"])
        data = np.array(pd.read_csv(pardict["datafile"], delim_whitespace=True, header=None))
        x_data = data[:, 0]
        fitmask = (np.where(np.logical_and(x_data >= pardict["xfit_min"], x_data <= pardict["xfit_max"]))[0]).astype(
            int
        )
        x_data = x_data[fitmask]
        fit_data = np.concatenate([data[fitmask, 1], data[fitmask, 2]])
        print(x_data, fitmask)

        # Read in, reshape and mask the covariance matrix
        cov_flat = np.array(pd.read_csv(pardict["covfile"], delim_whitespace=True, header=None))
        cov_size = np.array([cov_flat[-1, 0] + 1, cov_flat[-1, 1] + 1]).astype(int)
        cov_input = cov_flat[:, 2].reshape(cov_size)
        cov_size = (cov_size / 3).astype(int)
        cov = np.empty((2 * len(x_data), 2 * len(x_data)))
        cov[: len(x_data), : len(x_data)] = cov_input[fitmask[:, None], fitmask[None, :]]
        cov[: len(x_data), len(x_data) :] = cov_input[cov_size[0] + fitmask[:, None], fitmask[None, :]]
        cov[len(x_data) :, : len(x_data)] = cov_input[fitmask[:, None], cov_size[1] + fitmask[None, :]]
        cov[len(x_data) :, len(x_data) :] = cov_input[cov_size[0] + fitmask[:, None], cov_size[1] + fitmask[None, :]]

    else:
        x_data, pk0, pk2 = read_pk(pardict["datafile"], pardict["xfit_min"], pardict["xfit_max"], 1)
        fit_data = np.concatenate([pk0, pk2])

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

    # Set up the model
    common = pybird.Common(Nl=2, kmax=5.0, optiresum=False)
    nonlinear = pybird.NonLinear(load=False, save=False, co=common)
    resum = pybird.Resum(co=common)

    # Get some cosmological values at the grid centre
    kin, Plin, Da, Hz, fN, sigma8 = run_camb(pardict)

    # Set up the window function and projection effects. No window at the moment for the UNIT sims,
    # so we'll create an identity matrix for this. I'm also assuming that the fiducial cosmology
    # used to make the measurements is the same as Grid centre
    kout, nkout = common.k, len(common.k)
    projection = pybird.Projection(kout, DA=Da, H=Hz, window_fourier_name=None, co=common)
    projection.p = kout
    window = np.zeros((2, 2, nkout, nkout))
    window[0, 0, :, :] = np.eye(nkout)
    window[1, 1, :, :] = np.eye(nkout)
    projection.Waldk = window

    # Load in the model components
    truecrd = np.linspace(float(pardict["growth_min"]), float(pardict["growth_max"]), int(pardict["ngrowth"]))
    bird = pybird.Bird(kin, Plin, fN, DA=Da, H=Hz, z=pardict["z_pk"], which="all", co=common)
    lintab, looptab = get_template_grids(pardict, nmult=2, nout=2)
    lininterp = sp.interpolate.interp1d(truecrd, lintab, axis=0)
    loopinterp = sp.interpolate.interp1d(truecrd, looptab, axis=0)

    # Some constants for the EFT model
    k_m, k_nl = 0.7, 0.7
    eft_priors = np.array([2.0, 2.0, 4.0, 4.0, 2.0, 2.0, 2.0])
    priormat = np.diagflat(1.0 / eft_priors ** 2)

    # Plotting (for checking/debugging, should turn off for production runs)
    if plot_flag:
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

    # Optimize to set up the first point for the chain (for emcee we need to give each walker a random value about this point so we just sample the prior)
    do_marg = pardict["do_marg"]
    pardict["do_marg"] = 0
    start = np.array([1.0, 1.0, fN, 1.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    nll = lambda *args: -lnpost(*args)
    result = sp.optimize.basinhopping(
        nll,
        start,
        niter_success=10,
        niter=100,
        stepsize=0.1,
        minimizer_kwargs={
            "args": (pardict, 1, bird, Da, Hz),
            "method": "Nelder-Mead",
            "tol": 1.0e-4,
            "options": {"maxiter": 40000},
        },
    )
    print("#-------------- Best-fit----------------")
    print(result)
    exit()
    result = {"x": start}
    pardict["do_marg"] = do_marg

    # Set up the MCMC
    # How many free parameters and walkers (this is for emcee's method)
    if pardict["do_marg"]:
        nparams = len(result["x"]) - 7
    else:
        nparams = len(result["x"])
    nwalkers = nparams * 8

    if pardict["do_marg"]:
        begin = [
            [
                (0.01 * (np.random.rand() - 0.5) + 1.0) * result["x"][0],
                (0.01 * (np.random.rand() - 0.5) + 1.0) * result["x"][1],
                (0.1 * (np.random.rand() - 0.5) + 1.0) * result["x"][2],
                (0.1 * (np.random.rand() - 0.5) + 1.0) * result["x"][3],
                (0.1 * (np.random.rand() - 0.5) + 1.0) * result["x"][4],
                (0.1 * (np.random.rand() - 0.5) + 1.0) * result["x"][6],
            ]
            for i in range(nwalkers)
        ]
    else:
        begin = [
            [(0.02 * (np.random.rand() - 0.5) + 1.0) * result["x"][j] for j in range(len(result["x"]))]
            for i in range(nwalkers)
        ]

    # RELEASE THE CHAIN!!!
    sampler = emcee.EnsembleSampler(nwalkers, nparams, lnpost, args=[pardict, print_flag, bird, Da, Hz])
    pos, prob, state = sampler.run_mcmc(begin, 1)
    sampler.reset()

    if pardict["do_corr"]:
        chainfile = str("%s_xi_%2d_%3d_template.dat" % (pardict["fitfile"], pardict["xfit_min"], pardict["xfit_max"]))
    else:
        chainfile = str(
            "%s_pk_%3.2d_%3.2d_template.dat" % (pardict["fitfile"], pardict["xfit_min"], pardict["xfit_max"])
        )
    f = open(chainfile, "w")

    # Run and print out the chain for 20000 links
    counter = 0
    for result in sampler.sample(pos, iterations=20000):
        counter += 1
        if (counter % 100) == 0:
            print(counter)
            print("Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction)))
        position = result.coords
        lnprobab = result.log_prob
        for k in range(position.shape[0]):
            f.write("%4d  " % k)
            for m in range(position.shape[1]):
                f.write("%12.6f  " % position[k][m])
            f.write("%12.6f  " % lnprobab[k])
            f.write("\n")
    f.close()
