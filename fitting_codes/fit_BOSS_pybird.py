import numpy as np
import sys
from configobj import ConfigObj
from multiprocessing import Pool

sys.path.append("../")
from fitting_codes.fitting_utils import (
    FittingData,
    BirdModel,
    create_plot,
    update_plot,
    format_pardict,
    do_optimization,
    get_Planck,
)


def do_emcee(func, start):

    import emcee

    # Set up the MCMC
    # How many free parameters and walkers (this is for emcee's method)
    nparams = len(start)
    nwalkers = nparams * 4

    begin = [[(0.01 * (np.random.rand() - 0.5) + 1.0) * start[j] for j in range(len(start))] for i in range(nwalkers)]

    marg_str = "marg" if pardict["do_marg"] else "all"
    hex_str = "hex" if pardict["do_hex"] else "nohex"
    dat_str = "xi" if pardict["do_corr"] else "pk"
    fmt_str = "%s_%s_%2dhex%2d_%s_%s_%s.hdf5" if pardict["do_corr"] else "%s_%s_%3.2lfhex%3.2lf_%s_%s_%s_planck.hdf5"
    fitlim = birdmodels[0].pardict["xfit_min"][0] if pardict["do_corr"] else birdmodels[0].pardict["xfit_max"][0]
    fitlimhex = birdmodels[0].pardict["xfit_min"][2] if pardict["do_corr"] else birdmodels[0].pardict["xfit_max"][2]

    taylor_strs = ["grid", "1order", "2order", "3order", "4order"]
    chainfile = str(
        fmt_str
        % (
            birdmodels[0].pardict["fitfile"],
            dat_str,
            fitlim,
            fitlimhex,
            taylor_strs[pardict["taylor_order"]],
            hex_str,
            marg_str,
        )
    )
    print(chainfile)

    # Set up the backend
    backend = emcee.backends.HDFBackend(chainfile)
    backend.reset(nwalkers, nparams)

    with Pool() as pool:

        # Initialize the sampler
        sampler = emcee.EnsembleSampler(nwalkers, nparams, func, pool=pool, backend=backend, vectorize=True)

        # Run the sampler for a max of 20000 iterations. We check convergence every 100 steps and stop if
        # the chain is longer than 100 times the estimated autocorrelation time and if this estimate
        # changed by less than 1%. I copied this from the emcee site as it seemed reasonable.
        max_iter = 20000
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
            converged = np.all(tau * 50 < sampler.iteration)
            converged &= np.all(np.abs(old_tau - tau) / tau < 0.05)
            if converged:
                print("Reached Auto-Correlation time goal: %d > 100 x %.3f" % (counter, autocorr[index]))
                break
            old_tau = tau
            index += 1


def do_zeus(func, start):

    import zeus

    # Set up the MCMC
    # How many free parameters and walkers (this is for emcee's method)
    nparams = len(start)
    nwalkers = nparams * 4

    begin = [[(0.01 * (np.random.rand() - 0.5) + 1.0) * start[j] for j in range(len(start))] for i in range(nwalkers)]

    marg_str = "marg" if pardict["do_marg"] else "all"
    hex_str = "hex" if pardict["do_hex"] else "nohex"
    dat_str = "xi" if pardict["do_corr"] else "pk"
    fmt_str = "%s_%s_%2dhex%2d_%s_%s_%s.hdf5" if pardict["do_corr"] else "%s_%s_%3.2lfhex%3.2lf_%s_%s_%s_planck.hdf5"
    fitlim = birdmodels[0].pardict["xfit_min"][0] if pardict["do_corr"] else birdmodels[0].pardict["xfit_max"][0]
    fitlimhex = birdmodels[0].pardict["xfit_min"][2] if pardict["do_corr"] else birdmodels[0].pardict["xfit_max"][2]

    taylor_strs = ["grid", "1order", "2order", "3order", "4order"]
    chainfile = str(
        fmt_str
        % (
            birdmodels[0].pardict["fitfile"],
            dat_str,
            fitlim,
            fitlimhex,
            taylor_strs[pardict["taylor_order"]],
            hex_str,
            marg_str,
        )
    )
    print(chainfile)

    # Set up the backend
    with Pool() as pool:

        # Initialize the sampler
        sampler = zeus.EnsembleSampler(nwalkers, nparams, func, pool=pool, vectorize=True)

        old_tau = np.inf
        niter = 0
        converged = 0
        while ~converged:
            sampler.run_mcmc(begin, nsteps=20)
            tau = zeus.AutoCorrTime(sampler.get_chain(discard=0.5))
            converged = np.all(10 * tau < niter)
            converged &= np.all(np.abs(old_tau - tau) / tau < 0.05)
            old_tau = tau
            begin = None
            niter += 1000
            print("Niterations/Max Iterations: ", niter, "/", 5000)
            print("Integrated ACT/Min Convergence Iterations: ", tau, "/", np.amax(10 * tau))
            if niter >= 5000:
                break

        # Remove burn-in and and save the samples
        tau = zeus.AutoCorrTime(sampler.get_chain(discard=0.5))
        burnin = int(2 * np.max(tau))
        samples = sampler.get_chain(discard=burnin, flat=True).T


def lnpost(params):

    if params.ndim == 1:
        params = params.reshape((-1, len(params)))
    params = params.T

    # This returns the posterior distribution which is given by the log prior plus the log likelihood
    prior = lnprior(params, birdmodels)
    index = np.where(~np.isinf(prior))[0]
    like = np.zeros(np.shape(params[1]))
    if len(index) > 0:
        like[index] = lnlike(params[:, index], birdmodels, fittingdata, plt)

    return prior + like


def lnprior(params, birdmodels):

    # Here we define the prior for all the parameters. We'll ignore the constants as they
    # cancel out when subtracting the log posteriors

    ln10As, h, omega_cdm, omega_b, omega_k = params[:5]
    # ln10As, h, omega_cdm, omega_b, omega_k = birdmodels[0].valueref[:, None]
    # omega_k = [0.0]
    # ln10As, h, omega_cdm = params[:3]
    # omega_b = birdmodel.valueref[3] / birdmodel.valueref[2] * omega_cdm

    lower_bounds = birdmodels[0].valueref - birdmodels[0].pardict["order"] * birdmodels[0].delta
    upper_bounds = birdmodels[0].valueref + birdmodels[0].pardict["order"] * birdmodels[0].delta

    priors = np.zeros(np.shape(params[1]))

    # Flat priors for cosmological parameters
    for i, param in enumerate([ln10As, h, omega_cdm, omega_b, omega_k]):
        priors += np.where(np.logical_or(param < lower_bounds[i], param > upper_bounds[i]), -np.inf, 0.0)

    # BBN (D/H) inspired prior on omega_b
    omega_b_prior = -0.5 * (omega_b - 0.02230) ** 2 / 0.00037 ** 2
    # omega_b_prior = 0.0
    priors += omega_b_prior

    # Planck prior
    # diff = params[:4] - birdmodel.valueref
    # Planck_prior = -0.5 * diff @ planck_icov @ diff
    priors += np.zeros(np.shape(params[1]))

    nz = len(birdmodels[0].pardict["z_pk"])
    for i in range(nz):
        if birdmodels[0].pardict["do_marg"]:
            # b1, c2, c4 = params[-3 * (nz - i) : -3 * (nz - i - 1)] if i != nz - 1 else params[-3 * (nz - i) :]
            b1, c2 = params[-2 * (nz - i) : -2 * (nz - i - 1)] if i != nz - 1 else params[-2 * (nz - i) :]
        else:
            b1, c2, b3, c4, cct, cr1, cr2, ce1, cemono, cequad, bnlo = (
                params[-11 * (nz - i) : -11 * (nz - i - 1)] if i != nz - 1 else params[-11 * (nz - i) :]
            )

        # Flat prior for b1
        priors += np.where(np.logical_or(b1 < 0.0, b1 > 3.0), -np.inf, 0.0)

        # Flat prior for c2
        priors += np.where(np.logical_or(c2 < -4.0, c2 > 4.0), -np.inf, 0.0)

        # Gaussian prior for c4
        # priors += -0.5 * 0.25 * c4 ** 2

        if not birdmodels[0].pardict["do_marg"]:

            # Gaussian prior for b3 of width 2 centred on 0
            priors += -0.5 * 0.25 * b3 ** 2

            # Gaussian prior for cct of width 2 centred on 0
            priors += -0.5 * 0.25 * cct ** 2

            # Gaussian prior for cr1 of width 4 centred on 0
            priors += -0.5 * 0.0625 * cr1 ** 2

            # Gaussian prior for cr1 of width 4 centred on 0
            priors += -0.5 * 0.0625 * cr2 ** 2

            # Gaussian prior for ce1 of width 2 centred on 0
            priors += -0.5 * 0.25 * ce1 ** 2

            # Gaussian prior for cemono of width 2 centred on 0
            priors += -0.5 * 0.25 * cemono ** 2

            # Gaussian prior for cequad of width 2 centred on 0
            priors += -0.5 * 0.25 * cequad ** 2

            # Gaussian prior for bnlo of width 2 centred on 0
            priors += -0.5 * 0.25 * bnlo ** 2

    return priors


def lnlike(params, birdmodels, fittingdata, plt):

    # Get the bird model
    ln10As, h, omega_cdm, omega_b, omega_k = params[:5]
    # ln10As, h, omega_cdm, omega_b, omega_k = birdmodels[0].valueref[:, None]
    # omega_k = [0.0]
    # omega_b = birdmodel.valueref[3] / birdmodel.valueref[2] * omega_cdm

    Picount = 0
    P_models, Plins, Ploops = [], [], []
    nmarg = len(birdmodels[0].eft_priors)
    nz = len(birdmodels[0].pardict["z_pk"])
    Pi_full = np.zeros((nz * len(birdmodels[0].eft_priors), len(fittingdata.data["fit_data"]), len(ln10As)))
    for i in range(nz):
        if birdmodels[0].pardict["do_marg"]:
            # counter = -3 * (nz - i)
            # b2 = (params[counter + 1] + params[counter + 2]) / np.sqrt(2.0)
            # b4 = (params[counter + 1] - params[counter + 2]) / np.sqrt(2.0)
            counter = -2 * (nz - i)
            b2 = (params[counter + 1]) / np.sqrt(2.0)
            b4 = (params[counter + 1]) / np.sqrt(2.0)
            margb = np.zeros(np.shape(params)[1])
            bs = np.array(
                [
                    params[counter],
                    b2,
                    margb,
                    b4,
                    margb,
                    margb,
                    margb,
                    margb,
                    margb,
                    margb,
                    # margb,
                ]
            )
        else:
            counter = -11 * (nz - i)
            b2 = (params[counter + 1] + params[counter + 3]) / np.sqrt(2.0)
            b4 = (params[counter + 1] - params[counter + 3]) / np.sqrt(2.0)
            bs = np.array(
                [
                    params[counter],
                    b2,
                    params[counter + 2],
                    b4,
                    params[counter + 4],
                    params[counter + 5],
                    params[counter + 6],
                    params[counter + 7] * float(fittingdata.data["shot_noise"][i]),
                    params[counter + 8] * float(fittingdata.data["shot_noise"][i]),
                    params[counter + 9] * float(fittingdata.data["shot_noise"][i]),
                    # params[counter + 10],
                ]
            )

        Plin, Ploop = birdmodels[i].compute_pk(np.array([ln10As, h, omega_cdm, omega_b, omega_k]))
        P_model, P_model_interp = birdmodels[i].compute_model(bs, Plin, Ploop, fittingdata.data["x_data"][i])
        Pi = birdmodels[i].get_Pi_for_marg(
            Ploop, bs[0], float(fittingdata.data["shot_noise"][i]), fittingdata.data["x_data"][i]
        )
        Plins.append(Plin)
        Ploops.append(Ploop)
        P_models.append(P_model_interp)
        Pi_full[i * nmarg : (i + 1) * nmarg, Picount : Picount + fittingdata.data["ndata"][i]] = Pi
        Picount += fittingdata.data["ndata"][i]

    P_model = np.concatenate(P_models)
    # chi_squared = birdmodels[0].compute_chi2(P_model, Pi_full, fittingdata.data)

    # Now get the best-fit values for parameters we don't care about
    P_models = []
    bs_analytic = birdmodels[0].compute_bestfit_analytic(Pi_full[:, :, 0], fittingdata.data, P_model[:, 0])
    pardict["do_marg"] = 0
    for i in range(nz):
        # counter = -3 * (nz - i)
        # b2 = (params[counter + 1, 0] + params[counter + 2, 0]) / np.sqrt(2.0)
        # b4 = (params[counter + 1, 0] - params[counter + 2, 0]) / np.sqrt(2.0)
        counter = -2 * (nz - i)
        b2 = (params[counter + 1, 0]) / np.sqrt(2.0)
        b4 = (params[counter + 1, 0]) / np.sqrt(2.0)
        bs = np.array(
            [
                params[counter, 0],
                b2,
                bs_analytic[7 * i],
                b4,
                bs_analytic[7 * i + 1],
                bs_analytic[7 * i + 2],
                bs_analytic[7 * i + 3],
                bs_analytic[7 * i + 4] * float(fittingdata.data["shot_noise"][i]),
                bs_analytic[7 * i + 5] * float(fittingdata.data["shot_noise"][i]),
                bs_analytic[7 * i + 6] * float(fittingdata.data["shot_noise"][i]),
                # bs_analytic[8 * i + 7],
            ]
        )[:, None]
        P_model, P_model_interp = birdmodels[i].compute_model(
            bs, Plins[i][:, :, :, 0, None], Ploops[i][:, :, :, 0, None], fittingdata.data["x_data"][i]
        )
        P_models.append(P_model_interp[:, 0])
    chi_squared_print = birdmodels[0].compute_chi2(np.concatenate(P_models), Pi_full[:, :, 0], fittingdata.data)
    pardict["do_marg"] = 1

    if plt is not None:
        update_plot(pardict, fittingdata.data["x_data"][plot_flag - 1], P_models[plot_flag - 1], plt)
        if np.random.rand() < 1.0:
            print(params[:, 0], chi_squared_print, len(fittingdata.data["fit_data"]))

    """if plt is not None:
        P_models = []
        chi_squared_print = chi_squared[0]
        if birdmodels[0].pardict["do_marg"]:
            bs_analytic = birdmodels[0].compute_bestfit_analytic(Pi_full[:, :, 0], fittingdata.data, P_model[:, 0])
            pardict["do_marg"] = 0
            for i in range(nz):
                # counter = -3 * (nz - i)
                # b2 = (params[counter + 1, 0] + params[counter + 2, 0]) / np.sqrt(2.0)
                # b4 = (params[counter + 1, 0] - params[counter + 2, 0]) / np.sqrt(2.0)
                counter = -2 * (nz - i)
                b2 = (params[counter + 1, 0]) / np.sqrt(2.0)
                b4 = (params[counter + 1, 0]) / np.sqrt(2.0)
                bs = np.array(
                    [
                        params[counter, 0],
                        b2,
                        bs_analytic[8 * i],
                        b4,
                        bs_analytic[8 * i + 1],
                        bs_analytic[8 * i + 2],
                        bs_analytic[8 * i + 3],
                        bs_analytic[8 * i + 4] * float(fittingdata.data["shot_noise"][i]),
                        bs_analytic[8 * i + 5] * float(fittingdata.data["shot_noise"][i]),
                        bs_analytic[8 * i + 6] * float(fittingdata.data["shot_noise"][i]),
                        bs_analytic[8 * i + 7],
                    ]
                )[:, None]
                P_model, P_model_interp = birdmodels[i].compute_model(
                    bs, Plins[i][:, :, :, 0, None], Ploops[i][:, :, :, 0, None], fittingdata.data["x_data"][i]
                )
                P_models.append(P_model_interp[:, 0])
            chi_squared_print = birdmodels[0].compute_chi2(np.concatenate(P_models), Pi_full[:, :, 0], fittingdata.data)
            pardict["do_marg"] = 1
        update_plot(pardict, fittingdata.data["x_data"][plot_flag - 1], P_models[plot_flag - 1], plt)
        if np.random.rand() < 1.0:
            print(params[:, 0], bs_analytic, chi_squared, chi_squared_print, len(fittingdata.data["fit_data"]))"""

    return -0.5 * chi_squared_print


if __name__ == "__main__":

    # Code to generate a power spectrum template at fixed cosmology using pybird, then fit the AP parameters and fsigma8
    # First read in the config file
    configfile = sys.argv[1]
    plot_flag = int(sys.argv[2])
    pardict = ConfigObj(configfile)

    # Just converts strings in pardicts to numbers in int/float etc.
    pardict = format_pardict(pardict)

    # Set up the data
    fittingdata = FittingData(pardict)

    # Set up the BirdModels
    birdmodels = []
    for i in range(len(pardict["z_pk"])):
        # birdmodels.append(BirdModel(pardict, direct=True, redindex=i, window=fittingdata.data["windows"][i]))
        birdmodels.append(BirdModel(pardict, redindex=i))

    # Read in and create a Planck prior covariance matrix
    Planck_file = "/Volumes/Work/UQ/CAMB/COM_CosmoParams_fullGrid_R3.01/base/plikHM_TTTEEE_lowl_lowE_lensing/base_plikHM_TTTEEE_lowl_lowE_lensing"
    planck_mean, planck_cov, planck_icov = get_Planck(Planck_file, 4)

    # Plotting (for checking/debugging, should turn off for production runs)
    plt = None
    if plot_flag:
        plt = create_plot(pardict, fittingdata, plotindex=plot_flag - 1)

    start = [[2.6, birdmodels[0].valueref[1], birdmodels[0].valueref[2], birdmodels[0].valueref[3]], [-0.10]]
    if birdmodels[0].pardict["do_marg"]:
        for i in range(len(pardict["z_pk"])):
            start.append([1.3, 0.5])
    else:
        for i in range(len(pardict["z_pk"])):
            start.append([1.3, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    start = np.concatenate(start)

    # Does an optimization
    """lower_bounds = birdmodels[0].valueref - birdmodels[0].pardict["order"] * birdmodels[0].delta
    upper_bounds = birdmodels[0].valueref + birdmodels[0].pardict["order"] * birdmodels[0].delta
    start = (
        (lower_bounds[0], upper_bounds[0]),
        (lower_bounds[1], upper_bounds[1]),
        (lower_bounds[2], upper_bounds[2]),
        (lower_bounds[3], upper_bounds[3]),
        (lower_bounds[4], upper_bounds[4]),
        (0.0, 3.0),
        (-4.0, 4.0),
        (0.0, 3.0),
        (-4.0, 4.0),
        (0.0, 3.0),
        (-4.0, 4.0),
        (0.0, 3.0),
        (-4.0, 4.0),
    )"""
    result = do_optimization(lambda *args: -lnpost(*args), start)

    # Does an MCMC
    # do_emcee(lnpost, start)
