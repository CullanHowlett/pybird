import numpy as np
import copy
import camb


def grid_properties(pardict):
    """ Computes some useful properties of the grid given the parameters read from the input file

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
    delta = np.array(pardict["dx"], dtype=np.float) * valueref
    squarecrd = [np.arange(-order, order + 1) for l in pardict["freepar"]]
    truecrd = [valueref[l] + delta[l] * np.arange(-order, order + 1) for l in range(len(pardict["freepar"]))]
    squaregrid = np.array(np.meshgrid(*squarecrd, indexing="ij"))
    flattenedgrid = squaregrid.reshape([len(pardict["freepar"]), -1]).T

    return valueref, delta, flattenedgrid, truecrd


def run_camb(pardict, background_only=False):
    """ Runs an instance of CAMB given the cosmological parameters in pardict

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
        redshifts=[float(parlinear["z_pk"]), 0.0001], kmax=float(parlinear["P_k_max_h/Mpc"]),
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
    if background_only:
        results = camb.get_background(pars, no_thermo=True)
    else:
        results = camb.get_results(pars)

        # Get the power spectrum
        kin, _, Plin = results.get_matter_power_spectrum(
            minkh=2.0e-5, maxkh=float(parlinear["P_k_max_h/Mpc"]), npoints=2000
        )

    # Get some derived quantities
    Da = results.angular_diameter_distance(float(parlinear["z_pk"])) * float(parlinear["H0"]) / 299792.458
    H = results.hubble_parameter(float(parlinear["z_pk"])) / float(parlinear["H0"])
    fsigma8 = results.get_fsigma8()[0]
    sigma8 = results.get_sigma8()[0]

    if background_only:
        return Da, H
    else:
        return kin, Plin[-1], Da, H, fsigma8 / sigma8, sigma8
