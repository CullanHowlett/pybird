import numpy as np
import sys
from configobj import ConfigObj

sys.path.append("../")
from fitting_codes.fitting_utils import (
    FittingData,
    BirdModel,
    create_plot,
    update_plot,
    format_pardict,
    do_optimization,
)


def do_emcee(func, start, birdmodel, fittingdata, plt):

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
            [(0.02 * (np.random.rand() - 0.5) + 1.0) * start[j] for j in range(len(start))] for i in range(nwalkers)
        ]

    # RELEASE THE CHAIN!!!
    sampler = emcee.EnsembleSampler(nwalkers, nparams, func, args=[birdmodel, fittingdata, plt])
    pos, prob, state = sampler.run_mcmc(begin, 1)
    sampler.reset()

    marg_str = "marg" if pardict["do_marg"] else "all"
    hex_str = "hex" if pardict["do_hex"] else "nohex"
    dat_str = "xi" if pardict["do_corr"] else "pk"
    fmt_str = "%s_%s_%2d_%3d_%s_%s_%s.dat" if pardict["do_corr"] else "%s_%s_%3.2lf_%3.2lf_%s_%s_%s.dat"

    taylor_strs = ["grid", "1order", "2order", "3order", "4order"]
    chainfile = str(
        fmt_str
        % (
            birdmodel.pardict["fitfile"],
            dat_str,
            birdmodel.pardict["xfit_min"],
            birdmodel.pardict["xfit_max"],
            taylor_strs[pardict["taylor_order"]],
            hex_str,
            marg_str,
        )
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


def lnpost(params, birdmodel, fittingdata, plt):

    # This returns the posterior distribution which is given by the log prior plus the log likelihood
    prior = lnprior(params, birdmodel)
    if not np.isfinite(prior):
        return -np.inf
    like = lnlike(params, birdmodel, fittingdata, plt)
    return prior + like


def lnprior(params, birdmodel):

    # Here we define the prior for all the parameters. We'll ignore the constants as they
    # cancel out when subtracting the log posteriors
    if birdmodel.pardict["do_marg"]:
        b1, c2, c4 = params[-3:]
    else:
        if birdmodel.pardict["do_corr"]:
            b1, c2, b3, c4, cct, cr1, cr2 = params[-7:]
        else:
            b1, c2, b3, c4, cct, cr1, cr2, ce1, cemono, cequad = params[-10:]

    lower_bounds = birdmodel.valueref - birdmodel.pardict["template_order"] * birdmodel.delta
    upper_bounds = birdmodel.valueref + birdmodel.pardict["template_order"] * birdmodel.delta

    # Flat priors for alpha_perp, alpha_par and fsigma8
    if np.any(np.less(params[:3], lower_bounds)) or np.any(np.greater(params[:3], upper_bounds)):
        return -np.inf

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

        # Gaussian prior for ce1 of width 1 centred on 0
        ce1_prior = -0.5 * 0.25 * ce1 ** 2

        # Gaussian prior for cemono of width 2 centred on 0
        cemono_prior = -0.5 * 0.25 * cemono ** 2

        # Gaussian prior for cequad of width 2 centred on 0
        cequad_prior = -0.5 * 0.25 * cequad ** 2

        return c4_prior + b3_prior + cct_prior + cr1_prior + cr2_prior + ce1_prior + cemono_prior + cequad_prior


def lnlike(params, birdmodel, fittingdata, plt):

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
    Plin, Ploop = birdmodel.compute_pk(params[:3])
    P_model = birdmodel.compute_model(bs, Plin, Ploop, fittingdata.data["x_data"])
    Pi = birdmodel.get_Pi_for_marg(Ploop, bs[0], fittingdata.data["shot_noise"], fittingdata.data["x_data"])

    chi_squared = birdmodel.compute_chi2(P_model, Pi, fittingdata.data)

    if plt is not None:
        update_plot(pardict, fittingdata, P_model, plt)
        if np.random.rand() < 0.1:
            print(params, chi_squared)

    return -0.5 * chi_squared


if __name__ == "__main__":

    # Code to generate a power spectrum template at fixed cosmology using pybird, then fit the AP parameters and fsigma8
    # First read in the config file
    configfile = sys.argv[1]
    plot_flag = int(sys.argv[2])
    pardict = ConfigObj(configfile)

    # Just converts strings in pardicts to numbers in int/float etc.
    pardict = format_pardict(pardict)

    # Set up the data
    shot_noise = 309.210197  # Taken from the header of the data power spectrum file.
    fittingdata = FittingData(pardict, shot_noise=shot_noise)

    # Set up the BirdModel
    birdmodel = BirdModel(pardict, template=True)

    # Plotting (for checking/debugging, should turn off for production runs)
    plt = None
    if plot_flag:
        plt = create_plot(pardict, fittingdata)

    # Does an optimization
    start = np.array([1.0, 1.0, birdmodel.fN, 1.3, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    result = do_optimization(lambda *args: -lnpost(*args), start, birdmodel, fittingdata, plt)

    # Does an MCMC
    do_emcee(lnpost, start, birdmodel, fittingdata, plt)
