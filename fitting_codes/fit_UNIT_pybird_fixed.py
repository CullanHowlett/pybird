import numpy as np
import sys
import pandas as pd
from configobj import ConfigObj

sys.path.append("../")
from pybird import pybird
from fitting_codes.fitting_utils import (
    FittingData,
    BirdModel,
    create_plot,
    update_plot,
    format_pardict,
)


def do_optimization(func, start, birdmodel, fittingdata, plt, Plin, Ploop):

    from scipy.optimize import basinhopping

    result = basinhopping(
        func,
        start,
        niter_success=10,
        niter=100,
        stepsize=0.1,
        minimizer_kwargs={
            "args": (birdmodel, fittingdata, plt, Plin, Ploop),
            "method": "Nelder-Mead",
            "tol": 1.0e-3,
            "options": {"maxiter": 40000},
        },
    )
    print("#-------------- Best-fit----------------")
    print(result)

    return result


def do_emcee(func, start, birdmodel, fittingdata, plt, Plin, Ploop):

    import emcee

    # Set up the MCMC
    # How many free parameters and walkers (this is for emcee's method)
    if birdmodel.pardict["do_marg"]:
        if birdmodel.pardict["do_corr"]:
            nparams = len(start) - 4
        else:
            nparams = len(start) - 7
    else:
        nparams = len(start)
    nwalkers = nparams * 8

    if birdmodel.pardict["do_marg"]:
        if fixed_h:
            begin = [
                [
                    (0.01 * (np.random.rand() - 0.5) + 1.0) * start[0],
                    (0.1 * (np.random.rand() - 0.5) + 1.0) * start[1],
                    (0.1 * (np.random.rand() - 0.5) + 1.0) * start[2],
                    (0.1 * (np.random.rand() - 0.5) + 1.0) * start[3],
                    (0.1 * (np.random.rand() - 0.5) + 1.0) * start[5],
                ]
                for i in range(nwalkers)
            ]
        else:
            begin = [
                [
                    (0.01 * (np.random.rand() - 0.5) + 1.0) * start[0],
                    (0.01 * (np.random.rand() - 0.5) + 1.0) * start[1],
                    (0.1 * (np.random.rand() - 0.5) + 1.0) * start[2],
                    (0.1 * (np.random.rand() - 0.5) + 1.0) * start[3],
                    (0.1 * (np.random.rand() - 0.5) + 1.0) * start[4],
                    (0.1 * (np.random.rand() - 0.5) + 1.0) * start[6],
                ]
                for i in range(nwalkers)
            ]
    else:
        begin = [
            [(0.1 * (np.random.rand() - 0.5) + 1.0) * start[j] for j in range(len(start))] for i in range(nwalkers)
        ]

    h_str = "fixedh" if fixed_h else "varyh"
    marg_str = "marg" if pardict["do_marg"] else "all"
    hex_str = "hex" if pardict["do_hex"] else "nohex"
    dat_str = "xi" if pardict["do_corr"] else "pk"
    fmt_str = "%s_%s_%2d_%3d_%s_%s_%s_%s.hdf5" if pardict["do_corr"] else "%s_%s_%3.2lf_%3.2lf_%s_%s_%s_%s.hdf5"

    taylor_strs = ["grid", "1order", "2order", "3order", "4order"]
    chainfile = str(
        fmt_str
        % (
            birdmodel.pardict["fitfile"],
            dat_str,
            birdmodel.pardict["xfit_min"],
            birdmodel.pardict["xfit_max"],
            taylor_strs[pardict["taylor_order"]],
            h_str,
            hex_str,
            marg_str,
        )
    )

    # Set up the backend
    backend = emcee.backends.HDFBackend(chainfile)
    backend.reset(nwalkers, nparams)

    # Initialize the sampler
    sampler = emcee.EnsembleSampler(
        nwalkers, nparams, func, args=[birdmodel, fittingdata, plt, fixed_h], backend=backend
    )

    # Run the sampler for a max of 20000 iterations. We check convergence every 100 steps and stop if
    # the chain is longer than 100 times the estimated autocorrelation time and if this estimate
    # changed by less than 1%. I copied this from the emcee site as it seemed reasonable.
    max_iter = 40000
    index = 0
    old_tau = np.inf
    autocorr = np.empty(max_iter)
    counter = 0
    for sample in sampler.sample(begin, iterations=max_iter, progress=True):

        # Only check convergence every 100 steps
        if sampler.iteration % 100:
            continue

        # Compute the autocorrelation time so far
        # Using tol=0 means that we'll always get an estimate even
        # if it isn't trustworthy
        tau = sampler.get_autocorr_time(tol=0)
        autocorr[index] = np.mean(tau)
        counter += 100
        print("Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction)))
        print("Mean Auto-Correlation time: {0:.3f}".format(autocorr[index]))

        # Check convergence
        converged = np.all(tau * 100 < sampler.iteration)
        converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
        if converged:
            print("Reached Auto-Correlation time goal: %d > 100 x %.3f" % (counter, autocorr[index]))
            break
        old_tau = tau
        index += 1


def lnpost(params, birdmodel, fittingdata, plt, Plin, Ploop):

    # This returns the posterior distribution which is given by the log prior plus the log likelihood
    prior = lnprior(params, birdmodel)
    if not np.isfinite(prior):
        return -np.inf
    like = lnlike(params, birdmodel, fittingdata, plt, Plin, Ploop)
    return prior + like


def lnprior(params, birdmodel):

    # Here we define the prior for all the parameters. We'll ignore the constants as they
    # cancel out when subtracting the log posteriors
    if birdmodel.pardict["do_marg"]:
        b1, c2, c4 = params
    else:
        if birdmodel.pardict["do_corr"]:
            b1, c2, b3, c4, cct, cr1, cr2 = params
        else:
            b1, c2, b3, c4, cct, cr1, cr2, ce1, cemono, cequad = params

    # Flat prior for b1
    if b1 < 0.0 or b1 > 3.0:
        return -np.inf

    # Flat prior for c2
    if c2 < -4.0 or c2 > 4.0:
        return -np.inf

    # Gaussian prior for c4
    c4_prior = -0.5 * 0.25 * c4 ** 2

    if birdmodel.pardict["do_marg"]:

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

        if birdmodel.pardict["do_corr"]:

            return c4_prior + b3_prior + cct_prior + cr1_prior + cr2_prior

        else:

            # Gaussian prior for ce1 of width 2 centred on 0
            ce1_prior = -0.5 * 0.25 * ce1 ** 2

            # Gaussian prior for cemono of width 2 centred on 0
            cemono_prior = -0.5 * 0.25 * cemono ** 2

            # Gaussian prior for cequad of width 2 centred on 0
            cequad_prior = -0.5 * 0.25 * cequad ** 2

            return c4_prior + b3_prior + cct_prior + cr1_prior + cr2_prior + ce1_prior + cemono_prior + cequad_prior


def lnlike(params, birdmodel, fittingdata, plt, Plin, Ploop):

    if birdmodel.pardict["do_marg"]:
        b2 = (params[-2] + params[-1]) / np.sqrt(2.0)
        b4 = (params[-2] - params[-1]) / np.sqrt(2.0)
        if birdmodel.pardict["do_corr"]:
            bs = [params[-3], b2, 0.0, b4, 0.0, 0.0, 0.0]
        else:
            bs = [params[-3], b2, 0.0, b4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    else:
        if birdmodel.pardict["do_corr"]:
            b2 = (params[-6] + params[-4]) / np.sqrt(2.0)
            b4 = (params[-6] - params[-4]) / np.sqrt(2.0)
            bs = [params[-7], b2, params[-5], b4, params[-3], params[-2], params[-1]]
        else:
            b2 = (params[-9] + params[-7]) / np.sqrt(2.0)
            b4 = (params[-9] - params[-7]) / np.sqrt(2.0)
            bs = [
                params[-10],
                b2,
                params[-8],
                b4,
                params[-6],
                params[-5],
                params[-4],
                params[-3] * fittingdata.data["shot_noise"],
                params[-2] * fittingdata.data["shot_noise"],
                params[-1] * fittingdata.data["shot_noise"],
            ]

    # Get the bird model
    P_model, P_model_interp = birdmodel.compute_model(bs, Plin, Ploop, fittingdata.data["x_data"])
    Pi = birdmodel.get_Pi_for_marg(Ploop, bs[0], fittingdata.data["shot_noise"], fittingdata.data["x_data"])

    chi_squared = birdmodel.compute_chi2(P_model_interp, Pi, fittingdata.data)

    if plt is not None:
        update_plot(pardict, birdmodel.kin, P_model, plt)
        if np.random.rand() < 0.1:
            print(params, chi_squared)

    return -0.5 * chi_squared


if __name__ == "__main__":

    # Code to generate a power spectrum template at fixed cosmology using pybird then fit only the bias parameters
    configfile = sys.argv[1]
    plot_flag = int(sys.argv[2])
    pardict = ConfigObj(configfile)

    # Just converts strings in pardicts to numbers in int/float etc.
    pardict = format_pardict(pardict)

    # Set up the data
    shot_noise = 309.210197  # Taken from the header of the data power spectrum file.
    fittingdata = FittingData(pardict, shot_noise=shot_noise)

    # Set up the BirdModel by reading in the linear power spectrum
    pardict["do_hex"] = 1
    birdmodel = BirdModel(pardict, template=False, direct=True)
    Pin = np.array(
        pd.read_csv(
            "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_HandShake/Pk_Planck15_Table4.txt",
            delim_whitespace=True,
            header=None,
            skiprows=0,
        )
    )
    Om = float(pardict["omega_cdm"]) + float(pardict["omega_b"]) / float(pardict["h"]) ** 2
    birdmodel.Pmod *= (pybird.DgN(Om, 1.0 / (1.0 + float(pardict["z_pk"]))) / pybird.DgN(Om, 1.0)) ** 2

    import matplotlib.pyplot as plt

    fig = plt.figure(0)
    ax = fig.add_axes([0.13, 0.13, 0.85, 0.85])
    ax.plot(Pin[:, 0], Pin[:, 1], color="r")
    ax.plot(birdmodel.kmod, birdmodel.Pmod, color="b")
    ax.set_xscale("log")
    ax.set_yscale("log")
    plt.show()

    bird = pybird.Bird(
        birdmodel.kmod,
        birdmodel.Pmod,
        birdmodel.fN,
        DA=birdmodel.Da,
        H=birdmodel.Hz,
        z=pardict["z_pk"],
        which="all",
        co=birdmodel.common,
    )
    birdmodel.nonlinear.PsCf(bird)
    bird.setPsCfl()
    birdmodel.resum.PsCf(bird)
    birdmodel.projection.AP(bird)
    birdmodel.projection.kdata(bird)
    if pardict["do_corr"]:
        Plin, Ploop = bird.formatTaylorCf(sdata=birdmodel.kin)
    else:
        Plin, Ploop = bird.formatTaylorPs(kdata=birdmodel.kin)
    Plin = np.swapaxes(Plin.reshape((birdmodel.Nl, Plin.shape[-2] // birdmodel.Nl, Plin.shape[-1])), axis1=1, axis2=2)[
        :, 1:, :
    ]
    Ploop = np.swapaxes(
        Ploop.reshape((birdmodel.Nl, Ploop.shape[-2] // birdmodel.Nl, Ploop.shape[-1])), axis1=1, axis2=2
    )[:, 1:, :]
    pardict["do_hex"] = 0
    birdmodel.Nl = 2

    # Plotting (for checking/debugging, should turn off for production runs)
    plt = None
    if plot_flag:
        plt = create_plot(pardict, fittingdata)

    if pardict["do_corr"]:
        start = np.array([1.3, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    else:
        start = np.array([1.3, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    # Does an optimization
    result = do_optimization(lambda *args: -lnpost(*args), start, birdmodel, fittingdata, plt, Plin, Ploop)

    # Does an MCMC
    # do_emcee(lnpost, start, birdmodel, fittingdata, plt)

    # Does an MCMC with fixed h
    # if pardict["do_corr"]:
    #    start = np.array([birdmodel.valueref[0], omstart, 1.3, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    # else:
    #    start = np.array([birdmodel.valueref[0], omstart, 1.3, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    # do_emcee(lnpost, start, birdmodel, fittingdata, plt, fixed_h=True)
