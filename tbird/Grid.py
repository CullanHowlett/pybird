import numpy as np
import copy
import camb
from classy import Class
from scipy.special import hyp2f1


def smooth_hinton2017(ks, pk, degree=13, sigma=1, weight=0.5):
    """ Smooth power spectrum based on Hinton 2017 polynomial method """
    log_ks = np.log(ks)
    log_pk = np.log(pk)
    index = np.argmax(pk)
    maxk2 = log_ks[index]
    gauss = np.exp(-0.5 * np.power(((log_ks - maxk2) / sigma), 2))
    w = np.ones(pk.size) - weight * gauss
    z = np.polyfit(log_ks, log_pk, degree, w=w)
    p = np.poly1d(z)
    polyval = p(log_ks)
    pk_smoothed = np.exp(polyval)
    return pk_smoothed


def grid_properties(pardict):
    """Computes some useful properties of the grid given the parameters read from the input file

    Parameters
    ----------
    pardict: dict
        A dictionary of parameters read from the config file

    Returns
    -------
    valueref: np.array
        An array of the central values of the grid
    delta: np.array
        An array containing the grid cell widths
    flattenedgrid: np.array
        The number of grid cells from the center for each coordinate, flattened
    truecrd: list of np.array
        A list containing 1D numpy arrays for the values of the cosmological parameters along each grid axis
    """

    order = float(pardict["order"])
    valueref = np.array([float(pardict[k]) for k in pardict["freepar"]])
    delta = np.fabs(np.array(pardict["dx"], dtype=np.float) * valueref)
    squarecrd = [np.arange(-order, order + 1) for l in pardict["freepar"]]
    truecrd = [valueref[l] + delta[l] * np.arange(-order, order + 1) for l in range(len(pardict["freepar"]))]
    squaregrid = np.array(np.meshgrid(*squarecrd, indexing="ij"))
    flattenedgrid = squaregrid.reshape([len(pardict["freepar"]), -1]).T

    return valueref, delta, flattenedgrid, truecrd


def grid_properties_template(pardict, fN, sigma8):
    """Computes some useful properties of the grid given the parameters read from the input file

    Parameters
    ----------
    pardict: dict
        A dictionary of parameters read from the config file

    Returns
    -------
    valueref: np.array
        An array of the central values of the grid
    delta: np.array
        An array containing the grid cell widths
    flattenedgrid: np.array
        The number of grid cells from the center for each coordinate, flattened
    truecrd: list of np.array
        A list containing 1D numpy arrays for the values of the cosmological parameters along each grid axis
    """

    order = float(pardict["template_order"])
    valueref = np.array([1.0, 1.0, fN, sigma8])
    delta = np.array(pardict["template_dx"], dtype=np.float) * valueref
    squarecrd = [np.arange(-order, order + 1) for l in range(4)]
    truecrd = [valueref[l] + delta[l] * np.arange(-order, order + 1) for l in range(4)]
    squaregrid = np.array(np.meshgrid(*squarecrd, indexing="ij"))
    flattenedgrid = squaregrid.reshape([4, -1]).T

    return valueref, delta, flattenedgrid, truecrd


def grid_properties_template_hybrid(pardict, fsigma8, omegamh2):
    """Computes some useful properties of the grid given the parameters read from the input file

    Parameters
    ----------
    pardict: dict
        A dictionary of parameters read from the config file

    Returns
    -------
    valueref: np.array
        An array of the central values of the grid
    delta: np.array
        An array containing the grid cell widths
    flattenedgrid: np.array
        The number of grid cells from the center for each coordinate, flattened
    truecrd: list of np.array
        A list containing 1D numpy arrays for the values of the cosmological parameters along each grid axis
    """

    order = float(pardict["template_order"])
    valueref = np.array([1.0, 1.0, fsigma8, omegamh2])
    delta = np.array(pardict["template_dx"], dtype=np.float) * valueref
    squarecrd = [np.arange(-order, order + 1) for l in range(4)]
    truecrd = [valueref[l] + delta[l] * np.arange(-order, order + 1) for l in range(4)]
    squaregrid = np.array(np.meshgrid(*squarecrd, indexing="ij"))
    flattenedgrid = squaregrid.reshape([4, -1]).T

    return valueref, delta, flattenedgrid, truecrd


def run_camb(pardict, redindex=0):
    """Runs an instance of CAMB given the cosmological parameters in pardict

    Parameters
    ----------
    pardict: dict
        A dictionary of parameters read from the config file

    Returns
    -------
    kin: np.array
        the k-values of the CAMB linear power spectrum
    Plin: np.array
        The linear power spectrum
    Da: float
        The angular diameter distance to the value of z_pk in the config file, without the factor c/H_0
    H: float
        The Hubble parameter at z_pk, without the factor H_0
    fN: float
        The scale-independent growth rate at z_pk
    """

    parlinear = copy.deepcopy(pardict)

    # Set the CAMB parameters
    pars = camb.CAMBparams()
    if "A_s" not in parlinear.keys():
        if "ln10^{10}A_s" in parlinear.keys():
            parlinear["A_s"] = np.exp(float(parlinear["ln10^{10}A_s"])) / 1.0e10
        else:
            print("Error: Neither ln10^{10}A_s nor A_s given in config file")
            exit()
    if "H0" not in parlinear.keys():
        if "h" in parlinear.keys():
            parlinear["H0"] = 100.0 * float(parlinear["h"])
        else:
            print("Error: Neither H0 nor h given in config file")
            exit()
    if "w0_fld" in parlinear.keys():
        pars.set_dark_energy(w=float(parlinear["w0_fld"]), dark_energy_model="fluid")
    pars.InitPower.set_params(As=float(parlinear["A_s"]), ns=float(parlinear["n_s"]))
    pars.set_matter_power(
        redshifts=[float(parlinear["z_pk"][redindex]), 0.0], kmax=float(parlinear["P_k_max_h/Mpc"]), nonlinear=False
    )
    pars.set_cosmology(
        H0=float(parlinear["H0"]),
        omch2=float(parlinear["omega_cdm"]),
        ombh2=float(parlinear["omega_b"]),
        omk=float(parlinear["Omega_k"]),
        tau=float(parlinear["tau_reio"]),
        mnu=float(parlinear["Sum_mnu"]),
        neutrino_hierarchy=parlinear["nu_hierarchy"],
    )
    pars.NonLinear = camb.model.NonLinear_none

    # Run CAMB
    results = camb.get_results(pars)

    # Get the power spectrum
    kin, _, Plin = results.get_matter_power_spectrum(
        minkh=2.0e-5,
        maxkh=float(parlinear["P_k_max_h/Mpc"]),
        npoints=200,
    )

    # Get some derived quantities
    Omega_m = results.get_Omega("cdm") + results.get_Omega("baryon") + results.get_Omega("nu")
    Da = results.angular_diameter_distance(float(parlinear["z_pk"][redindex])) * float(parlinear["H0"]) / 299792.458
    H = results.hubble_parameter(float(parlinear["z_pk"][redindex])) / float(parlinear["H0"])
    fsigma8 = results.get_fsigma8()[0]
    sigma8 = results.get_sigma8()[0]
    sigma12 = results.get_sigmaR(12.0, hubble_units=False)[0]
    r_d = results.get_derived_params()["rdrag"]

    return kin, Plin[-1], Omega_m, Da, H, fsigma8 / sigma8, sigma8, sigma12, r_d


def run_class(pardict, redindex=0):
    """Runs an instance of CAMB given the cosmological parameters in pardict

    Parameters
    ----------
    pardict: dict
        A dictionary of parameters read from the config file

    Returns
    -------
    kin: np.array
        the k-values of the CAMB linear power spectrum
    Plin: np.array
        The linear power spectrum
    Da: float
        The angular diameter distance to the value of z_pk in the config file, without the factor c/H_0
    H: float
        The Hubble parameter at z_pk, without the factor H_0
    fN: float
        The scale-independent growth rate at z_pk
    """

    parlinear = copy.deepcopy(pardict)
    if int(parlinear["N_ncdm"] == 2):
        parlinear["m_ncdm"] = parlinear["m_ncdm"][0] + "," + parlinear["m_ncdm"][1]
    if int(parlinear["N_ncdm"] == 3):
        parlinear["m_ncdm"] = parlinear["m_ncdm"][0] + "," + parlinear["m_ncdm"][1] + "," + parlinear["m_ncdm"][2]

    # Set the CLASS parameters
    M = Class()
    if "A_s" not in parlinear.keys():
        if "ln10^{10}A_s" in parlinear.keys():
            parlinear["A_s"] = np.exp(float(parlinear["ln10^{10}A_s"])) / 1.0e10
        else:
            print("Error: Neither ln10^{10}A_s nor A_s given in config file")
            exit()
    if "H0" not in parlinear.keys():
        if "h" in parlinear.keys():
            parlinear["H0"] = 100.0 * float(parlinear["h"])
        else:
            print("Error: Neither H0 nor h given in config file")
            exit()
    M.set(
        {
            "A_s": float(parlinear["A_s"]),
            "n_s": float(parlinear["n_s"]),
            "H0": float(parlinear["H0"]),
            "omega_b": float(parlinear["omega_b"]),
            "omega_cdm": float(parlinear["omega_cdm"]),
            "N_ur": float(parlinear["N_ur"]),
            "N_ncdm": int(parlinear["N_ncdm"]),
            "m_ncdm": parlinear["m_ncdm"],
            "Omega_k": float(parlinear["Omega_k"]),
        }
    )
    M.set(
        {
            "output": "mPk",
            "P_k_max_1/Mpc": float(parlinear["P_k_max_h/Mpc"]),
            "z_max_pk": float(parlinear["z_pk"][redindex]),
        }
    )
    M.compute()

    kin = np.logspace(np.log10(2.0e-5), np.log10(float(parlinear["P_k_max_h/Mpc"])), 200)
    Plin = np.array([M.pk_cb_lin(ki * M.h(), float(parlinear["z_pk"][redindex])) * M.h() ** 3 for ki in kin])
    # Plin = np.array([M.pk_lin(ki * M.h(), 0.0) * M.h() ** 3 for ki in kin])
    # Plin *= (M.scale_independent_growth_factor(float(parlinear["z_pk"])) / M.scale_independent_growth_factor(0.0)) ** 2

    # Get some derived quantities
    Omega_m = M.Om_m(0.0)
    a_z = 1.0 / (1.0 + float(parlinear["z_pk"][redindex]))
    growth_z = a_z * hyp2f1(1.0 / 3.0, 1, 11.0 / 6.0, -(a_z ** 3) / Omega_m * (1.0 - Omega_m))
    growth_0 = hyp2f1(1.0 / 3.0, 1, 11.0 / 6.0, -1.0 / Omega_m * (1.0 - Omega_m))

    # print(growth_z / growth_0)

    # np.savetxt(
    #    "/Volumes/Work/UQ/DESI/MockChallenge/Pre_recon_Stage2/pkmodel_UNIT_cosmo_matter.dat",
    #    np.c_[kin, Plin],
    #    header="k    P_lin",
    # )

    # Plin *= (growth_z / growth_0) ** 2

    Da = M.angular_distance(float(parlinear["z_pk"][redindex])) * M.Hubble(0.0)
    H = M.Hubble(float(parlinear["z_pk"][redindex])) / M.Hubble(0.0)
    f = M.scale_independent_growth_factor_f(float(parlinear["z_pk"][redindex]))
    sigma8 = M.sigma(8.0 / M.h(), float(parlinear["z_pk"][redindex]))
    sigma8_0 = M.sigma(8.0 / M.h(), 0.0)
    sigma12 = M.sigma(12.0, float(parlinear["z_pk"][redindex]))
    r_d = M.rs_drag()

    # print(Omega_m, Da, H, f, sigma8, sigma12, r_d, r_d*float(parlinear["H0"])/100.0)

    return kin, Plin, Omega_m, Da, H, f, sigma8, sigma8_0, sigma12, r_d


if __name__ == "__main__":

    import sys

    sys.path.append("../")
    import matplotlib.pyplot as plt
    from configobj import ConfigObj
    from scipy.interpolate import splev, splrep

    # Read in the config file, job number and total number of jobs
    configfile = sys.argv[1]
    pardict = ConfigObj(configfile)

    # Get some cosmological values at the grid centre
    kin_camb, Plin_camb, Om_camb, Da_camb, Hz_camb, fN_camb, sigma8_camb, sigma12_camb, r_d_camb = run_camb(pardict)
    kin_class, Plin_class, Om_class, Da_class, Hz_class, fN_class, sigma8_class, sigma12_class, r_d_class = run_class(
        pardict
    )

    print(Om_camb, Om_class, 100.0 * (Om_camb / Om_class - 1.0))
    print(Da_camb, Da_class, 100.0 * (Da_camb / Da_class - 1.0))
    print(Hz_camb, Hz_class, 100.0 * (Hz_camb / Hz_class - 1.0))
    print(fN_camb, fN_class, 100.0 * (fN_camb / fN_class - 1.0))
    print(sigma8_camb, sigma8_class, 100.0 * (sigma8_camb / sigma8_class - 1.0))
    print(sigma12_camb, sigma12_class, 100.0 * (sigma12_camb / sigma12_class - 1.0))
    print(r_d_camb, r_d_class, 100.0 * (r_d_camb / r_d_class - 1.0))

    fig = plt.figure(0)
    ax = fig.add_axes([0.13, 0.13, 0.85, 0.85])
    ax.plot(kin_camb, Plin_camb, color="r")
    ax.plot(kin_class, Plin_class, color="b")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1.0e-5, 1.1 * float(pardict["P_k_max_h/Mpc"]))
    plt.show()

    fig = plt.figure(1)
    ax = fig.add_axes([0.13, 0.13, 0.85, 0.85])
    ax.plot(kin_camb, 100.0 * (Plin_camb / splev(kin_camb, splrep(kin_class, Plin_class)) - 1.0), color="r")
    ax.plot(kin_class, 100.0 * (splev(kin_class, splrep(kin_camb, Plin_camb)) / Plin_class - 1.0), color="b")

    plt.show()
