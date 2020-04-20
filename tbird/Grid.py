import numpy as np
import os
import plinear
import copy
from scipy.optimize import fsolve
import configobj as cfg
import camb

configfile = "/Volumes/Work/UQ/DESI/cBIRD/input_files/tbird_UNIT.ini"
gridname = "camb-z0p9873-A_s-h-omega_cdm-omega_b"
freepar = ["ln10^{10}A_s", "h", "omega_cdm", "omega_b"]
dx = np.array([0.1, 0.015, 0.005, 0.001])     # This is the percentage deviation!
order = 4  # For the moment I keep this same for everything.
center = order + 1 # Here we use a smaller grid, then padded with zeros

parref = cfg.ConfigObj(configfile)
redshift = float(parref["z_pk"])
valueref = np.array([float(parref[k]) for k in freepar])
delta = dx
squarecrd = [np.arange(-order, order + 1) for l in freepar]  # list -i_n, ..., +i_n where to evaluate each freepar
truecrd = [valueref[l] + delta[l] * np.arange(-order, order + 1) for l in range(len(freepar))]  # list -i_n, ..., +i_n where to evaluate each freepar
squaregrid = np.array(np.meshgrid(*squarecrd, indexing='ij'))
flattenedgrid = squaregrid.reshape([len(freepar), -1]).T
truegrid = np.array(np.meshgrid(*truecrd, indexing='ij'))

#psdatadir = os.path.join("input", "DataSims") 

def get_masses(sum_masses, hierarchy='NH'):
    # a function returning the three masses given the Delta m^2, the total mass, and the hierarchy (e.g. 'IN' or 'IH')
    # Values are in the latest PDG
    # any string containing letter 'n' will be considered as refering to normal hierarchy
    if 'n' in hierarchy.lower():
        # Normal hierarchy massive neutrinos. Calculates the individual
        # neutrino masses from M_tot_NH and deletes M_tot_NH
        delta_m_squared_21 = 7.37e-5
        delta_m_squared_31 = 2.56e-3
        def m1_func(m1, M_tot):
            return M_tot**2 - (m1 + np.sqrt(m1**2 + delta_m_squared_21) + np.sqrt(m1**2 + delta_m_squared_31))**2
        m1, opt_output, success, output_message = fsolve(
            m1_func, sum_masses/3., (sum_masses), full_output=True, xtol=1e-04, maxfev=500)
        m1 = m1[0]
        m2 = (delta_m_squared_21 + m1**2.)**0.5
        m3 = (delta_m_squared_31 + m1**2.)**0.5
        return m1, m2, m3
    else:
        return None


# First step: compute PS. Tested, good
def CompPterms(pardict, kmin=0.001, kmax=0.31):
    """Given a parameter dictionary, a kmin (can be None) and a kmax (can be None),
    returns Plin, Ploop concatenated for multipoles, shape (lenk * 3, columns).
    The zeroth column are the k"""
    parlinear = copy.deepcopy(pardict)
    # print("As", parlinear["A_s"])
    #if "ln10^{10}A_s" in parlinear.keys():
    #    print("we have the log here")
    #if float(parlinear['N_ncdm']) > 0:
    #    m1, m2, m3 = get_masses(float(parlinear['Sum_mnu']))
    #    parlinear['m_ncdm'] = str(m1) + ', ' + str(m2) + ', ' + str(m3)
    this_plin = plinear.LinearPower(parlinear, np.logspace(np.log10(1.e-7), np.log10(2.01), 150))
    this_plin.compute()
    kin, Plin = np.loadtxt(os.path.join(parlinear["PathToOutput"], "class_pk.dat"), unpack=True)
    z = this_plin.redshift
    Om_m = this_plin.Omega_m
    return kin, Plin, z, Om_m


# First step: compute PS. Tested, good
def CompPterms_camb(pardict):

    parlinear = copy.deepcopy(pardict)
    pars = camb.CAMBparams()
    if "A_s" not in parlinear.keys():
        if "ln10^{10}A_s" in parlinear.keys():
            parlinear["A_s"] = np.exp(float(parlinear["ln10^{10}A_s"]))/1.0e10
    if "H0" not in parlinear.keys():
        parlinear["H0"] = 100.0*float(parlinear["h"])
    if "w0_fld" in parlinear.keys():
        pars.set_dark_energy(w=float(parlinear["w0_fld"]), dark_energy_model="fluid")
    pars.InitPower.set_params(As=float(parlinear["A_s"]), ns=float(parlinear["n_s"]))
    pars.set_matter_power(redshifts=[float(parlinear["z_pk"]), 0.0001], kmax=float(parlinear["P_k_max_h/Mpc"]))
    pars.set_cosmology(H0=float(parlinear["H0"]), omch2=float(parlinear["omega_cdm"]), ombh2=float(parlinear["omega_b"]), 
            omk=float(parlinear["Omega_k"]), tau=float(parlinear["tau_reio"]), mnu=float(parlinear["Sum_mnu"]), neutrino_hierarchy="degenerate")
    pars.NonLinear = camb.model.NonLinear_none
    results = camb.get_results(pars)
    params = results.get_derived_params()
    kin, z, Plin = results.get_matter_power_spectrum(minkh=2.0e-5, maxkh=float(parlinear["P_k_max_h/Mpc"]), npoints=2000)

    Om_m = results.get_Omega('cdm')+results.get_Omega('baryon')+results.get_Omega('nu')
    Da = results.angular_diameter_distance(z[-1])*parlinear["H0"]/299792.458
    H = results.hubble_parameter(z[-1])/parlinear["H0"]
    fN = (results.get_fsigma8()/results.get_sigma8())[0]     # Assumes scale-independent. Could compute scale-dependent using finite differencing of the power spectra, but would then need to incorporate this into pybird.

    return kin, Plin[-1], z[-1], Om_m, Da, H, fN
