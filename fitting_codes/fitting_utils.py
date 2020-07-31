# This file contains useful routines that should work regardless of whether you are fitting
# with fixed or varying template, and for any number of cosmological parameters

import os
import copy
import numpy as np
import scipy as sp
from scipy.interpolate import splrep, splev
import sys
from scipy.linalg import lapack, cholesky
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append("../")
from pybird import pybird
from tbird.Grid import grid_properties, grid_properties_template, run_camb, run_class
from tbird.computederivs import get_grids, get_template_grids, get_PSTaylor, get_ParamsTaylor

# Wrapper around the pybird data and model evaluation
class BirdModel:
    def __init__(self, pardict, template=False, direct=False):

        self.pardict = pardict
        self.Nl = 3 if pardict["do_hex"] else 2
        self.template = template
        self.direct = direct

        # Some constants for the EFT model
        self.k_m, self.k_nl = 0.7, 0.7
        self.eft_priors = np.array([2.0, 2.0, 4.0, 4.0, 2.0, 2.0, 2.0])
        self.priormat = np.diagflat(1.0 / self.eft_priors ** 2)

        # Get some values at the grid centre
        if pardict["code"] == "CAMB":
            self.kmod, self.Pmod, self.Om, self.Da, self.Hz, self.fN, self.sigma8, self.sigma12, self.r_d = run_camb(
                pardict
            )
        else:
            self.kmod, self.Pmod, self.Om, self.Da, self.Hz, self.fN, self.sigma8, self.sigma12, self.r_d = run_class(
                pardict
            )

        if self.template:
            self.valueref, self.delta, self.flattenedgrid, self.truecrd = grid_properties_template(
                pardict, self.fN, self.sigma8
            )
        else:
            self.valueref, self.delta, self.flattenedgrid, self.truecrd = grid_properties(pardict)

        # Prepare the model
        if self.direct:
            print("Direct not currently supported :(")
            exit()
            if self.template:
                self.correlator, self.bird = self.setup_pybird()
                self.kin = self.correlator.projection.xout
            else:
                self.correlator, self.bird = self.setup_pybird()
                self.kin = self.correlator.projection.xout
        else:
            self.kin, self.paramsmod, self.linmod, self.loopmod = self.load_model()

    def setup_pybird(self):

        from pybird_dev.pybird import Correlator
        from pybird_dev.bird import Bird

        Nl = 3 if self.pardict["do_hex"] else 2
        optiresum = True if self.pardict["do_corr"] else False
        output = "bCf" if self.pardict["do_corr"] else "bPk"
        z_pk = float(self.pardict["z_pk"])
        correlator = Correlator()

        # Set up pybird
        correlator.set(
            {
                "output": output,
                "multipole": Nl,
                "z": z_pk,
                "optiresum": optiresum,
                "with_bias": False,
                "with_exact_time": False,
                "with_time": False,
                "kmax": 0.5,
                "with_AP": True,
                "DA_AP": self.Da,
                "H_AP": self.Hz,
            }
        )

        correlator.read_cosmo({"k11": self.kmod, "P11": self.Pmod, "z": z_pk, "Omega0_m": self.Om})

        bird = Bird(
            correlator.cosmo,
            with_bias=correlator.config["with_bias"],
            with_stoch=correlator.config["with_stoch"],
            with_nlo_bias=correlator.config["with_nlo_bias"],
            with_assembly_bias=correlator.config["with_assembly_bias"],
            co=correlator.co,
        )
        correlator.nonlinear.PsCf(bird)
        bird.setPsCfl()

        return correlator, bird

    def load_model(self):

        # Load in the model components
        gridname = self.pardict["code"].lower() + "-" + self.pardict["gridname"]
        if self.pardict["taylor_order"]:
            if self.template:
                paramsmod = None
                if self.pardict["do_corr"]:
                    linmod = np.load(
                        os.path.join(self.pardict["outgrid"], "DerClin_%s_template.npy" % gridname), allow_pickle=True,
                    )
                    loopmod = np.load(
                        os.path.join(self.pardict["outgrid"], "DerCloop_%s_template.npy" % gridname), allow_pickle=True,
                    )
                else:
                    linmod = np.load(
                        os.path.join(self.pardict["outgrid"], "DerPlin_%s_template.npy" % gridname), allow_pickle=True,
                    )
                    loopmod = np.load(
                        os.path.join(self.pardict["outgrid"], "DerPloop_%s_template.npy" % gridname), allow_pickle=True,
                    )
            else:
                paramsmod = np.load(
                    os.path.join(self.pardict["outgrid"], "DerParams_%s.npy" % gridname), allow_pickle=True,
                )
                if self.pardict["do_corr"]:
                    linmod = np.load(
                        os.path.join(self.pardict["outgrid"], "DerClin_%s.npy" % gridname), allow_pickle=True,
                    )
                    loopmod = np.load(
                        os.path.join(self.pardict["outgrid"], "DerCloop_%s.npy" % gridname), allow_pickle=True,
                    )
                else:
                    linmod = np.load(
                        os.path.join(self.pardict["outgrid"], "DerPlin_%s.npy" % gridname), allow_pickle=True,
                    )
                    loopmod = np.load(
                        os.path.join(self.pardict["outgrid"], "DerPloop_%s.npy" % gridname), allow_pickle=True,
                    )
            kin = linmod[0][0, :, 0]
        else:
            if self.template:
                lintab, looptab = get_template_grids(self.pardict, pad=False, cf=self.pardict["do_corr"])
                paramsmod = None
                kin = lintab[..., 0, :, 0][(0,) * 4]
            else:
                paramstab, lintab, looptab = get_grids(self.pardict, pad=False, cf=self.pardict["do_corr"])
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

    def compute_model_direct(self, coords, bs, x_data):

        bias = {"b1": bs[0], "b2": bs[1], "b3": bs[2], "b4": bs[3], "cct": bs[4], "cr1": bs[5], "cr2": bs[6]}
        ce1, cemono, cequad = bs[-3:]
        self.bird.f = coords[2]

        if self.pardict["do_corr"]:
            self.correlator.resum.PsCf(self.bird)
            self.bird.setreduceCflb(bias)
            self.correlator.projection.AP(bird=self.bird, q=coords[:2])
            plin, ploop = self.bird.formatTaylorCf()
            if self.pardict["do_hex"]:
                P0, P2, P4 = self.bird.fullCf
            else:
                P0, P2 = self.bird.fullCf
        else:
            self.correlator.resum.Ps(self.bird)
            self.bird.setreducePslb(bias)
            self.correlator.projection.AP(bird=self.bird, q=coords[:2])
            plin, ploop = self.bird.formatTaylorPs()
            if self.pardict["do_hex"]:
                P0, P2, P4 = self.bird.fullPs
            else:
                P0, P2 = self.bird.fullPs

        P0_interp = sp.interpolate.splev(x_data[0], sp.interpolate.splrep(self.kin, P0))
        P2_interp = sp.interpolate.splev(x_data[1], sp.interpolate.splrep(self.kin, P2))
        if self.pardict["do_hex"]:
            P4_interp = sp.interpolate.splev(x_data[2], sp.interpolate.splrep(self.kin, P4))

        if self.pardict["do_corr"]:
            C0 = np.exp(-self.k_m * x_data[0]) * self.k_m ** 2 / (4.0 * np.pi * x_data[0])
            C1 = -self.k_m ** 2 * np.exp(-self.k_m * x_data[0]) / (4.0 * np.pi * x_data[0] ** 2)
            C2 = (
                np.exp(-self.k_m * x_data[1])
                * (3.0 + 3.0 * self.k_m * x_data[1] + self.k_m ** 2 * x_data[1] ** 2)
                / (4.0 * np.pi * x_data[1] ** 3)
            )

            P0_interp += ce1 * C0 + cemono * C1
            P2_interp += cequad * C2
        else:
            P0_interp += ce1 + cemono * x_data[0] ** 2 / self.k_m ** 2
            P2_interp += cequad * x_data[1] ** 2 / self.k_m ** 2

        if self.pardict["do_hex"]:
            P_model = np.concatenate([P0, P2, P4])
            P_model_interp = np.concatenate([P0_interp, P2_interp, P4_interp])
        else:
            P_model = np.concatenate([P0, P2])
            P_model_interp = np.concatenate([P0_interp, P2_interp])

        ploop = ploop.reshape((3, ploop.shape[0] // 3, ploop.shape[1]))
        ploop = np.swapaxes(ploop, axis1=1, axis2=2)[:, 1:, :]

        return P_model, P_model_interp, ploop

    def compute_model(self, cvals, plin, ploop, x_data):

        plin0, plin2, plin4 = plin
        ploop0, ploop2, ploop4 = ploop

        if self.pardict["do_corr"]:
            b1, b2, b3, b4, cct, cr1, cr2, ce1, cemono, cequad = cvals
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
        if self.pardict["do_hex"]:
            P4 = np.dot(cvals, ploop4) + plin4[0] + b1 * plin4[1] + b1 * b1 * plin4[2]

        P0_interp = sp.interpolate.splev(x_data[0], sp.interpolate.splrep(self.kin, P0))
        P2_interp = sp.interpolate.splev(x_data[1], sp.interpolate.splrep(self.kin, P2))
        if self.pardict["do_hex"]:
            P4_interp = sp.interpolate.splev(x_data[2], sp.interpolate.splrep(self.kin, P4))

        if self.pardict["do_corr"]:
            C0 = np.exp(-self.k_m * x_data[0]) * self.k_m ** 2 / (4.0 * np.pi * x_data[0])
            C1 = -self.k_m ** 2 * np.exp(-self.k_m * x_data[0]) / (4.0 * np.pi * x_data[0] ** 2)
            C2 = (
                np.exp(-self.k_m * x_data[1])
                * (3.0 + 3.0 * self.k_m * x_data[1] + self.k_m ** 2 * x_data[1] ** 2)
                / (4.0 * np.pi * x_data[1] ** 3)
            )

            P0_interp += ce1 * C0 + cemono * C1
            P2_interp += cequad * C2
        else:
            P0_interp += ce1 + cemono * x_data[0] ** 2 / self.k_m ** 2
            P2_interp += cequad * x_data[1] ** 2 / self.k_m ** 2

        if self.pardict["do_hex"]:
            P_model = np.concatenate([P0, P2, P4])
            P_model_interp = np.concatenate([P0_interp, P2_interp, P4_interp])
        else:
            P_model = np.concatenate([P0, P2])
            P_model_interp = np.concatenate([P0_interp, P2_interp])

        return P_model, P_model_interp

    def compute_chi2(self, P_model, Pi, data):

        if self.pardict["do_marg"]:

            Covbi = np.dot(Pi, np.dot(data["cov_inv"], Pi.T))
            Covbi += self.priormat
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
            for i in range(len(data["fit_data"])):
                chi_squared += (P_model[i] - data["fit_data"][i]) * np.sum(
                    data["cov_inv"][i, 0:] * (P_model - data["fit_data"])
                )

        return chi_squared

    # Ignore names, works for both power spectrum and correlation function
    def get_Pi_for_marg(self, ploop, b1, shot_noise, x_data):

        if self.pardict["do_marg"]:

            ploop0, ploop2, ploop4 = ploop

            Pb3 = np.concatenate(
                [
                    splev(x_data[0], splrep(self.kin, ploop0[3] + b1 * ploop0[7])),
                    splev(x_data[1], splrep(self.kin, ploop2[3] + b1 * ploop2[7])),
                ]
            )
            Pcct = np.concatenate(
                [
                    splev(x_data[0], splrep(self.kin, ploop0[15] + b1 * ploop0[12])),
                    splev(x_data[1], splrep(self.kin, ploop2[15] + b1 * ploop2[12])),
                ]
            )
            Pcr1 = np.concatenate(
                [
                    splev(x_data[0], splrep(self.kin, ploop0[16] + b1 * ploop0[13])),
                    splev(x_data[1], splrep(self.kin, ploop2[16] + b1 * ploop2[13])),
                ]
            )
            Pcr2 = np.concatenate(
                [
                    splev(x_data[0], splrep(self.kin, ploop0[17] + b1 * ploop0[14])),
                    splev(x_data[1], splrep(self.kin, ploop2[17] + b1 * ploop2[14])),
                ]
            )

            if self.pardict["do_hex"]:

                Pb3 = np.concatenate([Pb3, splev(x_data[2], splrep(self.kin, ploop4[3] + b1 * ploop4[7]))])
                Pcct = np.concatenate([Pcct, splev(x_data[2], splrep(self.kin, ploop4[15] + b1 * ploop4[12]))])
                Pcr1 = np.concatenate([Pcr1, splev(x_data[2], splrep(self.kin, ploop4[16] + b1 * ploop4[13]))])
                Pcr2 = np.concatenate([Pcr2, splev(x_data[2], splrep(self.kin, ploop4[17] + b1 * ploop4[14]))])

            if self.pardict["do_corr"]:

                C0 = np.concatenate(
                    [np.exp(-self.k_m * x_data[0]) / (4.0 * np.pi * x_data[0]), np.zeros(len(x_data[1]))]
                )  # shot-noise mono
                C1 = np.concatenate(
                    [-np.exp(-self.k_m * x_data[0]) / (4.0 * np.pi * x_data[0] ** 2), np.zeros(len(x_data[1]))]
                )  # k^2 mono
                C2 = np.concatenate(
                    [
                        np.zeros(len(x_data[0])),
                        np.exp(-self.k_m * x_data[1])
                        * (3.0 + 3.0 * self.k_m * x_data[1] + self.k_m ** 2 * x_data[1] ** 2)
                        / (4.0 * np.pi * x_data[1] ** 3),
                    ]
                )  # k^2 quad

                if self.pardict["do_hex"]:
                    C0 = np.concatenate([C0, np.zeros(len(x_data[2]))])  # shot-noise mono
                    C1 = np.concatenate([C1, np.zeros(len(x_data[2]))])  # k^2 mono
                    C2 = np.concatenate([C2, np.zeros(len(x_data[2]))])  # k^2 quad

                Pi = np.array(
                    [
                        Pb3,  # *b3
                        2.0 * Pcct / self.k_nl ** 2,  # *cct
                        2.0 * Pcr1 / self.k_m ** 2,  # *cr1
                        2.0 * Pcr2 / self.k_m ** 2,  # *cr2
                        C0 * self.k_m ** 2 * shot_noise,  # ce1
                        C1 * self.k_m ** 2 * shot_noise,  # cemono
                        C2 * shot_noise,  # cequad
                    ]
                )

            else:

                Onel0 = np.concatenate([np.ones(len(x_data[0])), np.zeros(len(x_data[1]))])  # shot-noise mono
                kl0 = np.concatenate([x_data[0], np.zeros(len(x_data[1]))])  # k^2 mono
                kl2 = np.concatenate([np.zeros(len(x_data[0])), x_data[1]])  # k^2 quad

                if self.pardict["do_hex"]:
                    Onel0 = np.concatenate([Onel0, np.zeros(len(x_data[2]))])  # shot-noise mono
                    kl0 = np.concatenate([kl0, np.zeros(len(x_data[2]))])  # k^2 mono
                    kl2 = np.concatenate([kl2, np.zeros(len(x_data[2]))])  # k^2 quad

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

        else:

            Pi = None

        return Pi

    def compute_bestfit_analytic(self, Pi, data):

        Covbi = np.dot(Pi, np.dot(data["cov_inv"], Pi.T))
        Covbi += self.priormat
        Cinvbi = np.linalg.inv(Covbi)
        vectorbi = Pi @ data["cov_inv"] @ data["fit_data"]

        return Cinvbi @ vectorbi

    def get_components(self, coords, cvals, shotnoise=None):

        plin, ploop = self.compute_pk(coords)

        if self.pardict["do_hex"]:
            plin0, plin2 = plin
            ploop0, ploop2 = ploop
        else:
            plin0, plin2, plin4 = plin
            ploop0, ploop2, ploop4 = ploop

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
        if self.pardict["do_hex"]:
            P4lin = plin4[0] + b1 * plin4[1] + b1 * b1 * plin4[2]
            P4loop = np.dot(cloop, ploop4[:12, :])
            P4ct = np.dot(cvalsct, ploop4[12:, :])
            Plin = [P0lin, P2lin, P4lin]
            Ploop = [P0loop, P2loop, P4loop]
            Pct = [P0ct, P2ct, P4ct]
        else:
            Plin = [P0lin, P2lin]
            Ploop = [P0loop, P2loop]
            Pct = [P0ct, P2ct]

        if self.pardict["do_corr"]:
            P0st, P2st, P4st = None, None, None
        else:
            P0st = ce1 * shotnoise + cemono * shotnoise * self.kin ** 2 / self.k_m ** 2
            P2st = cequad * shotnoise * self.kin ** 2 / self.k_m ** 2
            P4st = np.zeros(len(self.kin))
        if self.pardict["do_hex"]:
            Pst = [P0st, P2st, P4st]
        else:
            Pst = [P0st, P2st]

        return Plin, Ploop, Pct, Pst


# Holds all the data in a convenient dictionary
class FittingData:
    def __init__(self, pardict, shot_noise=0.0):

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

        # Check covariance matrix is symmetric and positive-definite by trying to do a cholesky decomposition
        diff = np.abs((self.data["cov"] - self.data["cov"].T) / self.data["cov"])
        if not (np.logical_or(diff <= 1.0e-6, np.isnan(diff))).all():
            print(diff)
            print("Error: Covariance matrix not symmetric!")
            exit(0)
        try:
            cholesky(self.data["cov"])
        except:
            print("Error: Covariance matrix not positive-definite!")
            exit(0)

    def read_pk(self, inputfile, step_size):

        dataframe = pd.read_csv(
            inputfile,
            comment="#",
            skiprows=10,
            delim_whitespace=True,
            names=["k", "pk0", "pk1", "pk2", "pk3", "pk4", "nk"],
        )
        k = dataframe["k"].values
        if step_size == 1:
            k_rebinned = k
            pk0_rebinned = dataframe["pk0"].values
            pk2_rebinned = dataframe["pk2"].values
            pk4_rebinned = dataframe["pk4"].values
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
                dataframe["pk4"].values = np.concatenate(
                    (dataframe["pk4"].values, [dataframe["pk4"].values[-1]] * to_add)
                )
                weight = np.concatenate((weight, [0] * to_add))
            k = k.reshape((-1, step_size))
            pk0 = (dataframe["pk0"].values).reshape((-1, step_size))
            pk2 = (dataframe["pk2"].values).reshape((-1, step_size))
            pk4 = (dataframe["pk4"].values).reshape((-1, step_size))
            weight = weight.reshape((-1, step_size))
            # Take the average of every group of step_size rows to rebin
            k_rebinned = np.average(k, axis=1)
            pk0_rebinned = np.average(pk0, axis=1, weights=weight)
            pk2_rebinned = np.average(pk2, axis=1, weights=weight)
            pk4_rebinned = np.average(pk4, axis=1, weights=weight)

        return np.vstack([k_rebinned, pk0_rebinned, pk2_rebinned, pk4_rebinned]).T

    def read_data(self, pardict):

        # Read in the data
        print(pardict["datafile"])
        if pardict["do_corr"]:
            data = np.array(pd.read_csv(pardict["datafile"], delim_whitespace=True, header=None))
        else:
            data = self.read_pk(pardict["datafile"], 1)

        x_data = data[:, 0]
        fitmask = [
            (np.where(np.logical_and(x_data >= pardict["xfit_min"][0], x_data <= pardict["xfit_max"][0]))[0]).astype(
                int
            ),
            (np.where(np.logical_and(x_data >= pardict["xfit_min"][1], x_data <= pardict["xfit_max"][1]))[0]).astype(
                int
            ),
            (np.where(np.logical_and(x_data >= pardict["xfit_min"][2], x_data <= pardict["xfit_max"][2]))[0]).astype(
                int
            ),
        ]
        x_data = [data[fitmask[0], 0], data[fitmask[1], 0], data[fitmask[2], 0]]
        if pardict["do_hex"]:
            fit_data = np.concatenate([data[fitmask[0], 1], data[fitmask[1], 2], data[fitmask[2], 3]])
        else:
            fit_data = np.concatenate([data[fitmask[0], 1], data[fitmask[1], 2]])

        # Read in, reshape and mask the covariance matrix
        cov_flat = np.array(pd.read_csv(pardict["covfile"], delim_whitespace=True, header=None))
        nin = len(data[:, 0])
        cov_input = cov_flat[:, 2].reshape((3 * nin, 3 * nin))
        nx0, nx2 = len(x_data[0]), len(x_data[1])
        nx4 = len(x_data[2]) if pardict["do_hex"] else 0
        mask0, mask2, mask4 = fitmask[0][:, None], fitmask[1][:, None], fitmask[2][:, None]
        cov = np.zeros((nx0 + nx2 + nx4, nx0 + nx2 + nx4))
        cov[:nx0, :nx0] = cov_input[mask0, mask0.T]
        cov[:nx0, nx0 : nx0 + nx2] = cov_input[mask0, nin + mask2.T]
        cov[nx0 : nx0 + nx2, :nx0] = cov_input[nin + mask2, mask0.T]
        cov[nx0 : nx0 + nx2, nx0 : nx0 + nx2] = cov_input[nin + mask0, nin + mask2.T]
        if pardict["do_hex"]:
            cov[:nx0, nx0 + nx2 :] = cov_input[mask0, 2 * nin + mask4.T]
            cov[nx0 + nx2 :, :nx0] = cov_input[2 * nin + mask4, mask0.T]
            cov[nx0 : nx0 + nx2, nx0 + nx2 :] = cov_input[nin + mask2, 2 * nin + mask4.T]
            cov[nx0 + nx2 :, nx0 : nx0 + nx2] = cov_input[2 * nin + mask4, nin + mask2.T]
            cov[nx0 + nx2 :, nx0 + nx2 :] = cov_input[2 * nin + mask4, 2 * nin + mask4.T]

        # Invert the covariance matrix
        identity = np.eye(nx0 + nx2 + nx4)
        cov_lu, pivots, cov_inv, info = lapack.dgesv(cov, identity)

        chi2data = np.dot(fit_data, np.dot(cov_inv, fit_data))
        invcovdata = np.dot(fit_data, cov_inv)

        return x_data, fit_data, cov, cov_inv, chi2data, invcovdata, fitmask


def create_plot(pardict, fittingdata):

    if pardict["do_hex"]:
        x_data = fittingdata.data["x_data"]
        nx0, nx2, nx4 = len(x_data[0]), len(x_data[1]), len(x_data[2])
    else:
        print(fittingdata.data["x_data"][:2])
        x_data = fittingdata.data["x_data"][:2]
        nx0, nx2, nx4 = len(x_data[0]), len(x_data[1]), 0
    fit_data = fittingdata.data["fit_data"]
    cov = fittingdata.data["cov"]

    plt_data = (
        np.concatenate(x_data) ** 2 * fit_data if pardict["do_corr"] else np.concatenate(x_data) ** 1.5 * fit_data
    )
    if pardict["do_corr"]:
        plt_err = np.concatenate(x_data) ** 2 * np.sqrt(cov[np.diag_indices(nx0 + nx2 + nx4)])
    else:
        plt_err = np.concatenate(x_data) ** 1.5 * np.sqrt(cov[np.diag_indices(nx0 + nx2 + nx4)])

    """plt.errorbar(
        x_data[0],
        plt_data[:nx0],
        yerr=plt_err[:nx0],
        marker="o",
        markerfacecolor="r",
        markeredgecolor="k",
        color="r",
        linestyle="None",
        markeredgewidth=1.3,
        zorder=5,
    )
    plt.errorbar(
        x_data[1],
        plt_data[nx0 : nx0 + nx2],
        yerr=plt_err[nx0 : nx0 + nx2],
        marker="o",
        markerfacecolor="b",
        markeredgecolor="k",
        color="b",
        linestyle="None",
        markeredgewidth=1.3,
        zorder=5,
    )
    if pardict["do_hex"]:
        plt.errorbar(
            x_data[2],
            plt_data[nx0 + nx2 :],
            yerr=plt_err[nx0 + nx2 :],
            marker="o",
            markerfacecolor="g",
            markeredgecolor="k",
            color="g",
            linestyle="None",
            markeredgewidth=1.3,
            zorder=5,
        )"""

    plt.xlim(0.0, np.amax(pardict["xfit_max"]) * 1.05)
    plt.ylim(-2.0, 2.0)

    if pardict["do_corr"]:
        plt.xlabel(r"$s\,(h^{-1}\,\mathrm{Mpc})$", fontsize=16)
        plt.ylabel(r"$s^{2}\xi(s)$", fontsize=16, labelpad=5)
    else:
        plt.xlabel(r"$k\,(h\,\mathrm{Mpc}^{-1})$", fontsize=16)
        plt.ylabel(r"$k^{3/2}P(k)\,(h^{-3/2}\,\mathrm{Mpc}^{3/2})$", fontsize=16, labelpad=5)
    plt.tick_params(width=1.3)
    plt.tick_params("both", length=10, which="major")
    plt.tick_params("both", length=5, which="minor")
    for axis in ["top", "left", "bottom", "right"]:
        plt.gca().spines[axis].set_linewidth(1.3)
    for tick in plt.gca().xaxis.get_ticklabels():
        tick.set_fontsize(14)
    for tick in plt.gca().yaxis.get_ticklabels():
        tick.set_fontsize(14)
    plt.tight_layout()
    plt.gca().set_autoscale_on(False)
    plt.ion()

    return plt


def update_plot(pardict, fittingdata, x_data, P_model, plt, keep=False):

    if pardict["do_hex"]:
        nx0, nx2, nx4 = len(x_data[0]), len(x_data[1]), len(x_data[2])
    else:
        x_data = x_data[:2]
        nx0, nx2, nx4 = len(x_data[0]), len(x_data[1]), 0
    plt_data = (
        (P_model - fittingdata.data["fit_data"]) / np.sqrt(fittingdata.data["cov"][np.diag_indices(nx0 + nx2 + nx4)])
        if pardict["do_corr"]
        else (P_model - fittingdata.data["fit_data"])
        / np.sqrt(fittingdata.data["cov"][np.diag_indices(nx0 + nx2 + nx4)])
    )

    plt10 = plt.errorbar(
        x_data[0], plt_data[:nx0], marker="None", color="r", linestyle="-", markeredgewidth=1.3, zorder=0,
    )
    plt11 = plt.errorbar(
        x_data[1], plt_data[nx0 : nx0 + nx2], marker="None", color="b", linestyle="-", markeredgewidth=1.3, zorder=0,
    )
    if pardict["do_hex"]:
        plt12 = plt.errorbar(
            x_data[2], plt_data[nx0 + nx2 :], marker="None", color="g", linestyle="-", markeredgewidth=1.3, zorder=0,
        )

    if keep:
        plt.ioff()
        plt.show()
    if not keep:
        plt.pause(0.005)
        if plt10 is not None:
            plt10.remove()
        if plt11 is not None:
            plt11.remove()
        if pardict["do_hex"]:
            if plt12 is not None:
                plt12.remove()


def format_pardict(pardict):

    pardict["do_corr"] = int(pardict["do_corr"])
    pardict["do_marg"] = int(pardict["do_marg"])
    pardict["do_hex"] = int(pardict["do_hex"])
    pardict["taylor_order"] = int(pardict["taylor_order"])
    pardict["xfit_min"] = np.array(pardict["xfit_min"]).astype(float)
    pardict["xfit_max"] = np.array(pardict["xfit_max"]).astype(float)
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
        stepsize=0.05,
        minimizer_kwargs={
            "args": (birdmodel, fittingdata, plt),
            "method": "Nelder-Mead",
            "tol": 1.0e-4,
            "options": {"maxiter": 40000, "xatol": 1.0e-4, "fatol": 1.0e-4},
        },
    )
    print("#-------------- Best-fit----------------")
    print(result)

    return result


def read_chain(chainfile, burnlimitlow=5000, burnlimitup=None):

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


def read_chain_backend(chainfile):

    import copy
    import emcee

    reader = emcee.backends.HDFBackend(chainfile)

    tau = reader.get_autocorr_time()
    burnin = int(2 * np.max(tau))
    samples = reader.get_chain(discard=burnin, flat=True)
    log_prob_samples = reader.get_log_prob(discard=burnin, flat=True)
    bestid = np.argmax(log_prob_samples)

    return samples, copy.copy(samples[bestid]), log_prob_samples
