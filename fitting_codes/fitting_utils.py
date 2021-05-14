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
from tbird.Grid import grid_properties, run_camb, run_class
from tbird.computederivs import get_grids, get_PSTaylor, get_ParamsTaylor

# Wrapper around the pybird data and model evaluation
class BirdModel:
    def __init__(self, pardict, redindex=0, template=False, direct=False, window=None):

        self.redindex = redindex
        self.pardict = pardict
        self.Nl = 3 if pardict["do_hex"] else 2
        self.template = template
        self.direct = direct
        self.window = window

        # Some constants for the EFT model
        self.k_m, self.k_nl = 0.7, 0.7
        self.eft_priors = np.array([2.0, 2.0, 4.0, 4.0, 2.0, 2.0, 2.0, 2.0])

        # Get some values at the grid centre
        if pardict["code"] == "CAMB":
            (
                self.kmod,
                self.Pmod,
                self.Om,
                self.Da,
                self.Hz,
                self.fN,
                self.sigma8,
                self.sigma8_0,
                self.sigma12,
                self.r_d,
            ) = run_camb(pardict)
            self.omega_nu = float(self.pardict["Sum_mnu"]) / 93.14
        else:
            (
                self.kmod,
                self.Pmod,
                self.Om,
                self.Da,
                self.Hz,
                self.fN,
                self.sigma8,
                self.sigma8_0,
                self.sigma12,
                self.r_d,
            ) = run_class(pardict)
            self.omega_nu = float(self.pardict["m_ncdm"]) / 93.14

        # Prepare the model
        if self.direct:
            self.valueref = np.array([float(pardict[k]) for k in pardict["freepar"]])
            self.delta = np.array(pardict["dx"], dtype=np.float) * self.valueref
            if self.template:
                self.correlator = self.setup_pybird()
                self.correlator.compute(
                    {
                        "k11": self.kmod,
                        "P11": self.Pmod,
                        "z": float(self.pardict["z_pk"][self.redindex]),
                        "Omega0_m": self.Om,
                        "f": self.fN,
                        "DA": self.Da,
                        "H": self.Hz,
                    }
                )
                self.linmod, self.loopmod = None, None
                self.kin = self.correlator.co.k
            else:
                self.correlator = self.setup_pybird()
                self.kin = self.correlator.co.k
        else:
            self.valueref, self.delta, self.flattenedgrid, self.truecrd = grid_properties(pardict)
            self.kin, self.paramsmod, self.linmod, self.loopmod = self.load_model()
            if self.template:
                self.correlator = self.setup_pybird()
                self.correlator.compute(
                    {
                        "k11": self.kmod,
                        "P11": self.Pmod,
                        "z": float(self.pardict["z_pk"][self.redindex]),
                        "Omega0_m": self.Om,
                        "f": self.fN,
                        "DA": self.Da,
                        "H": self.Hz,
                    }
                )

    def setup_pybird(self):

        from pybird_dev.pybird import Correlator

        Nl = 3 if self.pardict["do_hex"] else 2
        optiresum = True if self.pardict["do_corr"] else False
        output = "bCf" if self.pardict["do_corr"] else "bPk"
        kmax = None if self.pardict["do_corr"] else 0.6
        correlator = Correlator()

        # Set up pybird
        correlator.set(
            {
                "output": output,
                "multipole": Nl,
                "z": float(self.pardict["z_pk"][self.redindex]),
                "optiresum": optiresum,
                "with_bias": False,
                "with_nlo_bias": True,
                "with_time": not (self.template),
                "with_AP": True,
                "kmax": kmax,
                "DA_AP": self.Da,
                "H_AP": self.Hz,
                "with_window": False if self.window is None else True,
                "windowPk": self.window,
                "windowCf": self.window + ".dat",
            }
        )

        return correlator

    def load_model(self):

        # Load in the model components
        outgrids = np.loadtxt(self.pardict["outgrid"], dtype=str)
        gridnames = np.loadtxt(self.pardict["gridname"], dtype=str)
        gridname = self.pardict["code"].lower() + "-" + gridnames[self.redindex]
        if self.pardict["taylor_order"]:
            paramsmod = np.load(
                os.path.join(outgrids[self.redindex], "DerParams_%s.npy" % gridname),
                allow_pickle=True,
            )
            if self.template:
                paramsmod = None
                if self.pardict["do_corr"]:
                    linmod = np.load(
                        os.path.join(outgrids[self.redindex], "DerClin_%s_noAP.npy" % gridname),
                        allow_pickle=True,
                    )
                    loopmod = np.load(
                        os.path.join(outgrids[self.redindex], "DerCloop_%s_noAP.npy" % gridname),
                        allow_pickle=True,
                    )
                else:
                    linmod = np.load(
                        os.path.join(outgrids[self.redindex], "DerPlin_%s.npy" % gridname),
                        allow_pickle=True,
                    )
                    loopmod = np.load(
                        os.path.join(outgrids[self.redindex], "DerPloop_%s_noAP.npy" % gridname),
                        allow_pickle=True,
                    )
            else:
                if self.pardict["do_corr"]:
                    linmod = np.load(
                        os.path.join(outgrids[self.redindex], "DerClin_%s.npy" % gridname),
                        allow_pickle=True,
                    )
                    loopmod = np.load(
                        os.path.join(outgrids[self.redindex], "DerCloop_%s.npy" % gridname),
                        allow_pickle=True,
                    )
                else:
                    linmod = np.load(
                        os.path.join(outgrids[self.redindex], "DerPlin_%s.npy" % gridname),
                        allow_pickle=True,
                    )
                    loopmod = np.load(
                        os.path.join(outgrids[self.redindex], "DerPloop_%s.npy" % gridname),
                        allow_pickle=True,
                    )
            kin = linmod[0][0, :, 0]
        else:
            paramstab, lintab, looptab, lintab_noAP, looptab_noAP = get_grids(
                self.pardict, outgrids[self.redindex], gridname, pad=False, cf=self.pardict["do_corr"]
            )
            paramsmod = sp.interpolate.RegularGridInterpolator(self.truecrd, paramstab)
            kin = lintab[..., 0, :, 0][(0,) * len(self.pardict["freepar"])]
            if self.template:
                linmod = sp.interpolate.RegularGridInterpolator(self.truecrd, lintab_noAP)
                loopmod = sp.interpolate.RegularGridInterpolator(self.truecrd, looptab_noAP)
            else:
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

        if self.direct:
            Plins, Ploops = [], []
            for i in range(np.shape(coords)[1]):
                Plin, Ploop = self.compute_model_direct(coords[:, i])
                Plins.append(Plin)
                Ploops.append(Ploop)
            Plin = np.transpose(np.array(Plins), axes=[1, 2, 3, 0])
            Ploop = np.transpose(np.array(Ploops), axes=[1, 3, 2, 0])
        else:
            if self.pardict["taylor_order"]:
                dtheta = coords - self.valueref[:, None]
                Plin = get_PSTaylor(dtheta, self.linmod, self.pardict["taylor_order"])
                Ploop = get_PSTaylor(dtheta, self.loopmod, self.pardict["taylor_order"])
            else:
                Plin = self.linmod(coords.T)
                Ploop = self.loopmod(coords.T)
            Plin = np.transpose(Plin, axes=[1, 3, 2, 0])[:, 1:, :, :]
            Ploop = np.transpose(Ploop, axes=[1, 2, 3, 0])[:, :, 1:, :]

        return Plin, Ploop

    def compute_model_direct(self, coords):

        parameters = copy.deepcopy(self.pardict)

        for k, var in enumerate(self.pardict["freepar"]):
            parameters[var] = coords[k]
        if parameters["code"] == "CAMB":
            kin, Pin, Om, Da, Hz, fN, sigma8, sigma8_0, sigma12, r_d = run_camb(parameters)
        else:
            kin, Pin, Om, Da, Hz, fN, sigma8, sigma8_0, sigma12, r_d = run_class(parameters)

        # Get non-linear power spectrum from pybird
        self.correlator.compute(
            {
                "k11": kin,
                "P11": Pin,
                "z": float(self.pardict["z_pk"][self.redindex]),
                "Omega0_m": Om,
                "f": fN,
                "DA": Da,
                "H": Hz,
            }
        )
        Plin, Ploop = (
            self.correlator.bird.formatTaylorCf() if self.pardict["do_corr"] else self.correlator.bird.formatTaylorPs()
        )

        Plin = np.swapaxes(Plin.reshape((self.Nl, Plin.shape[-2] // self.Nl, Plin.shape[-1])), axis1=1, axis2=2)[
            :, 1:, :
        ]
        Ploop = np.swapaxes(Ploop.reshape((self.Nl, Ploop.shape[-2] // self.Nl, Ploop.shape[-1])), axis1=1, axis2=2)[
            :, 1:, :
        ]

        return Plin, Ploop

    def modify_template(self, params):
        # Modify the template power spectrum by scaling by f and then reapplying the AP effect.
        alpha_perp, alpha_par, fsigma8 = params
        self.correlator.bird.f = fsigma8 / self.sigma8

        P11l_AP, Pctl_AP, Ploopl_AP, Pnlol_AP = self.correlator.projection.AP(
            bird=self.correlator.bird, q=[alpha_perp, alpha_par], overwrite=False
        )
        Plin, Ploop = self.correlator.bird.formatTaylorPs(Ps=[P11l_AP, Ploopl_AP, Pctl_AP, Pnlol_AP])

        sum_cols = [0, 1, 4, 7, 9, 10, 12, 15, 17, 18, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
        Ploop = np.array([np.sum(Ploop[:, sum_cols[i] : sum_cols[i + 1]], axis=1) for i in range(20)]).T

        Plin = np.swapaxes(Plin.reshape((3, Plin.shape[-2] // 3, Plin.shape[-1])), axis1=1, axis2=2)[:, 1:, :]
        Ploop = np.swapaxes(Ploop.reshape((3, Ploop.shape[-2] // 3, Ploop.shape[-1])), axis1=1, axis2=2)[:, 1:, :]

        return Plin, Ploop

    def compute_hybrid(self, params):

        omega_rat = self.valueref[3] / self.valueref[2]
        omega_cdm = (params[3] - self.omega_nu) / (1.0 + omega_rat)
        omega_b = omega_rat * omega_cdm

        coords = [self.valueref[0], self.valueref[1], omega_cdm, omega_b]
        Plin, Ploop = self.compute_pk(coords)

        self.correlator.bird.P11l = np.einsum(
            "n,lnk->lnk", 1.0 / np.array([1.0, 2.0 * self.fN, self.fN ** 2]), Plin[:, ::-1, :]
        )
        self.correlator.bird.Ploopl = np.einsum(
            "n,lnk->lnk",
            1.0
            / np.array(
                [
                    self.fN ** 2,
                    self.fN ** 3,
                    self.fN ** 4,
                    self.fN,
                    self.fN ** 2,
                    self.fN ** 3,
                    self.fN,
                    self.fN ** 2,
                    self.fN,
                    self.fN,
                    self.fN ** 2,
                    1.0,
                    self.fN,
                    self.fN ** 2,
                    1.0,
                    self.fN,
                    1.0,
                    1.0,
                    self.fN,
                    1.0,
                    1.0,
                    1.0,
                ]
            ),
            Ploop[:, :22, :],
        )
        self.correlator.bird.Pctl = np.einsum(
            "n,lnk->lnk",
            1.0 / np.array([2.0, 2.0, 2.0, 2.0 * self.fN, 2.0 * self.fN, 2.0 * self.fN]),
            Ploop[:, 22:28],
        )
        if self.correlator.bird.with_nlo_bias:
            self.correlator.bird.Pnlol[:, 0, :] = Ploop[:, 28]

        return self.modify_template(params[:3])

    def compute_model(self, cvals, plin, ploop, x_data):

        if (self.direct or self.template) and not self.pardict["do_hex"]:
            plin0, plin2 = plin
            ploop0, ploop2 = ploop
        else:
            plin0, plin2, plin4 = plin
            ploop0, ploop2, ploop4 = ploop

        b1, b2, b3, b4, cct, cr1, cr2, ce1, cemono, cequad, bnlo = cvals

        # the columns of the Ploop data files.
        cvals = np.array(
            [
                np.ones(np.shape(b1)),
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
                2.0 * b1 * cct / self.k_nl ** 2,
                2.0 * b1 * cr1 / self.k_m ** 2,
                2.0 * b1 * cr2 / self.k_m ** 2,
                2.0 * cct / self.k_nl ** 2,
                2.0 * cr1 / self.k_m ** 2,
                2.0 * cr2 / self.k_m ** 2,
                2.0 * b1 ** 2 * bnlo / self.k_m ** 4,
            ]
        )

        P0 = np.sum(cvals * ploop0, axis=1) + plin0[0] + b1 * plin0[1] + b1 * b1 * plin0[2]
        P2 = np.sum(cvals * ploop2, axis=1) + plin2[0] + b1 * plin2[1] + b1 * b1 * plin2[2]
        if self.pardict["do_hex"]:
            P4 = np.sum(cvals * ploop4, axis=1) + plin4[0] + b1 * plin4[1] + b1 * b1 * plin4[2]

        P0_interp = [sp.interpolate.splev(x_data[0], sp.interpolate.splrep(self.kin, P0[:, i])) for i in range(len(b1))]
        P2_interp = [sp.interpolate.splev(x_data[1], sp.interpolate.splrep(self.kin, P2[:, i])) for i in range(len(b1))]
        if self.pardict["do_hex"]:
            P4_interp = [
                sp.interpolate.splev(x_data[2], sp.interpolate.splrep(self.kin, P4[:, i])) for i in range(len(b1))
            ]

        if self.pardict["do_corr"]:
            C0 = np.exp(-self.k_m * x_data[0]) * self.k_m ** 2 / (4.0 * np.pi * x_data[0])
            C1 = -self.k_m ** 2 * np.exp(-self.k_m * x_data[0]) / (4.0 * np.pi * x_data[0] ** 2)
            C2 = (
                np.exp(-self.k_m * x_data[1])
                * (3.0 + 3.0 * self.k_m * x_data[1] + self.k_m ** 2 * x_data[1] ** 2)
                / (4.0 * np.pi * x_data[1] ** 3)
            )

            P0_interp += np.outer(ce1, C0) + np.outer(cemono, C1)
            P2_interp += np.outer(cequad, C2)
        else:
            P0_interp += ce1[:, None] + np.outer(cemono, x_data[0] ** 2 / self.k_m ** 2)
            P2_interp += np.outer(cequad, x_data[1] ** 2 / self.k_m ** 2)

        if self.pardict["do_hex"]:
            P_model = np.concatenate([P0, P2, P4], axis=1)
            P_model_interp = np.concatenate([P0_interp, P2_interp, P4_interp], axis=1)
        else:
            P_model = np.concatenate([P0, P2], axis=1)
            P_model_interp = np.concatenate([P0_interp, P2_interp], axis=1)

        return P_model.T, P_model_interp.T

    def compute_chi2(self, P_model, Pi, data):

        if self.pardict["do_marg"]:

            Pi = np.transpose(Pi, axes=(2, 0, 1))
            Pimult = np.dot(Pi, data["cov_inv"])
            Covbi = np.einsum("dpk,dqk->dpq", Pimult, Pi)
            Covbi += np.diag(1.0 / np.tile(self.eft_priors, len(data["x_data"])))
            vectorbi = np.einsum("dpk,kd->dp", Pimult, P_model) - np.dot(Pi, data["invcovdata"])
            chi2nomar = (
                np.einsum("kd,kd->d", P_model, np.dot(data["cov_inv"], P_model))
                - 2.0 * np.dot(data["invcovdata"], P_model)
                + data["chi2data"]
            )
            chi2mar = -np.einsum("dp,dp->d", vectorbi, np.linalg.solve(Covbi, vectorbi)) + np.log(np.linalg.det(Covbi))
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

            if (self.direct or self.template) and not self.pardict["do_hex"]:
                ploop0, ploop2 = ploop
            else:
                ploop0, ploop2, ploop4 = ploop

            Pb3 = np.concatenate(
                np.swapaxes(
                    [
                        [
                            splev(x_data[0], splrep(self.kin, ploop0[:, 3, i] + b * ploop0[:, 7, i]))
                            for i, b in enumerate(b1)
                        ],
                        [
                            splev(x_data[1], splrep(self.kin, ploop2[:, 3, i] + b * ploop2[:, 7, i]))
                            for i, b in enumerate(b1)
                        ],
                    ],
                    axis1=1,
                    axis2=2,
                )
            )
            Pcct = np.concatenate(
                np.swapaxes(
                    [
                        [
                            splev(x_data[0], splrep(self.kin, ploop0[:, 15, i] + b * ploop0[:, 12, i]))
                            for i, b in enumerate(b1)
                        ],
                        [
                            splev(x_data[1], splrep(self.kin, ploop2[:, 15, i] + b * ploop2[:, 12, i]))
                            for i, b in enumerate(b1)
                        ],
                    ],
                    axis1=1,
                    axis2=2,
                )
            )
            Pcr1 = np.concatenate(
                np.swapaxes(
                    [
                        [
                            splev(x_data[0], splrep(self.kin, ploop0[:, 16, i] + b * ploop0[:, 13, i]))
                            for i, b in enumerate(b1)
                        ],
                        [
                            splev(x_data[1], splrep(self.kin, ploop2[:, 16, i] + b * ploop2[:, 13, i]))
                            for i, b in enumerate(b1)
                        ],
                    ],
                    axis1=1,
                    axis2=2,
                )
            )
            Pcr2 = np.concatenate(
                np.swapaxes(
                    [
                        [
                            splev(x_data[0], splrep(self.kin, ploop0[:, 17, i] + b * ploop0[:, 14, i]))
                            for i, b in enumerate(b1)
                        ],
                        [
                            splev(x_data[1], splrep(self.kin, ploop2[:, 17, i] + b * ploop2[:, 14, i]))
                            for i, b in enumerate(b1)
                        ],
                    ],
                    axis1=1,
                    axis2=2,
                )
            )
            Pnlo = np.concatenate(
                np.swapaxes(
                    [
                        [splev(x_data[0], splrep(self.kin, b ** 2 * ploop0[:, 18, i])) for i, b in enumerate(b1)],
                        [splev(x_data[1], splrep(self.kin, b ** 2 * ploop2[:, 18, i])) for i, b in enumerate(b1)],
                    ],
                    axis1=1,
                    axis2=2,
                )
            )

            if self.pardict["do_hex"]:

                Pb3 = np.concatenate(
                    [
                        Pb3,
                        np.array(
                            [
                                splev(x_data[0], splrep(self.kin, ploop4[:, 3, i] + b * ploop4[:, 7, i]))
                                for i, b in enumerate(b1)
                            ]
                        ).T,
                    ]
                )
                Pcct = np.concatenate(
                    [
                        Pcct,
                        np.array(
                            [
                                splev(x_data[0], splrep(self.kin, ploop4[:, 15, i] + b * ploop4[:, 12, i]))
                                for i, b in enumerate(b1)
                            ]
                        ).T,
                    ]
                )
                Pcr1 = np.concatenate(
                    [
                        Pcr1,
                        np.array(
                            [
                                splev(x_data[0], splrep(self.kin, ploop4[:, 16, i] + b * ploop4[:, 13, i]))
                                for i, b in enumerate(b1)
                            ]
                        ).T,
                    ]
                )
                Pcr2 = np.concatenate(
                    [
                        Pcr2,
                        np.array(
                            [
                                splev(x_data[0], splrep(self.kin, ploop4[:, 17, i] + b * ploop4[:, 14, i]))
                                for i, b in enumerate(b1)
                            ]
                        ).T,
                    ]
                )
                Pnlo = np.concatenate(
                    [
                        Pnlo,
                        np.array(
                            [splev(x_data[0], splrep(self.kin, b ** 2 * ploop4[:, 18, i])) for i, b in enumerate(b1)]
                        ).T,
                    ]
                )

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
                        np.tile(C0, (len(b1), 1)).T * self.k_m ** 2 * shot_noise,  # ce1
                        np.tile(C1, (len(b1), 1)).T * self.k_m ** 2 * shot_noise,  # cemono
                        np.tile(C2, (len(b1), 1)).T * shot_noise,  # cequad
                        2.0 * Pnlo / self.k_m ** 4,  # bnlo
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
                        np.tile(Onel0, (len(b1), 1)).T * shot_noise,  # *ce1
                        np.tile(kl0 ** 2, (len(b1), 1)).T / self.k_m ** 2 * shot_noise,  # *cemono
                        np.tile(kl2 ** 2, (len(b1), 1)).T / self.k_m ** 2 * shot_noise,  # *cequad
                        2.0 * Pnlo / self.k_m ** 4,  # bnlo
                    ]
                )

        else:

            Pi = None

        return Pi

    def compute_bestfit_analytic(self, Pi, data, model):

        Covbi = np.dot(Pi, np.dot(data["cov_inv"], Pi.T))
        Covbi += np.diag(1.0 / np.tile(self.eft_priors, len(data["x_data"])))
        Cinvbi = np.linalg.inv(Covbi)
        vectorbi = Pi @ data["cov_inv"] @ (data["fit_data"] - model)

        return Cinvbi @ vectorbi

    def get_components(self, coords, cvals):

        if self.direct:
            plin, ploop = self.compute_model_direct(coords)
        else:
            plin, ploop = self.compute_pk(coords)

        plin0, plin2, plin4 = plin
        ploop0, ploop2, ploop4 = ploop

        b1, b2, b3, b4, cct, cr1, cr2, ce1, cemono, cequad, bnlo = cvals

        # the columns of the Ploop data files.
        cloop = np.array([1, b1, b2, b3, b4, b1 * b1, b1 * b2, b1 * b3, b1 * b4, b2 * b2, b2 * b4, b4 * b4])
        cvalsct = np.array(
            [
                2.0 * b1 * cct / self.k_nl ** 2,
                2.0 * b1 * cr1 / self.k_m ** 2,
                2.0 * b1 * cr2 / self.k_m ** 2,
                2.0 * cct / self.k_nl ** 2,
                2.0 * cr1 / self.k_m ** 2,
                2.0 * cr2 / self.k_m ** 2,
            ]
        )
        cnlo = 2.0 * b1 ** 2 * bnlo / self.k_m ** 4

        P0lin = plin0[0] + b1 * plin0[1] + b1 * b1 * plin0[2]
        P2lin = plin2[0] + b1 * plin2[1] + b1 * b1 * plin2[2]
        P0loop = np.dot(cloop, ploop0[:12, :])
        P2loop = np.dot(cloop, ploop2[:12, :])
        P0ct = np.dot(cvalsct, ploop0[12:-1, :])
        P2ct = np.dot(cvalsct, ploop2[12:-1, :])
        P0nlo = cnlo * ploop0[-1, :]
        P2nlo = cnlo * ploop2[-1, :]
        if self.pardict["do_hex"]:
            P4lin = plin4[0] + b1 * plin4[1] + b1 * b1 * plin4[2]
            P4loop = np.dot(cloop, ploop4[:12, :])
            P4ct = np.dot(cvalsct, ploop4[12:-1, :])
            P4nlo = cnlo * ploop4[-1, :]
            Plin = [P0lin, P2lin, P4lin]
            Ploop = [P0loop + P0nlo, P2loop + P2nlo, P4loop + P4nlo]
            Pct = [P0ct, P2ct, P4ct]
        else:
            Plin = [P0lin, P2lin]
            Ploop = [P0loop + P0nlo, P2loop + P2nlo]
            Pct = [P0ct, P2ct]

        if self.pardict["do_corr"]:
            C0 = np.exp(-self.k_m * self.kin) * self.k_m ** 2 / (4.0 * np.pi * self.kin)
            C1 = -self.k_m ** 2 * np.exp(-self.k_m * self.kin) / (4.0 * np.pi * self.kin ** 2)
            C2 = (
                np.exp(-self.k_m * self.kin)
                * (3.0 + 3.0 * self.k_m * self.kin + self.k_m ** 2 * self.kin ** 2)
                / (4.0 * np.pi * self.kin ** 3)
            )
            P0st = ce1 * C0 + cemono * C1
            P2st = cequad * C2
            P4st = np.zeros(len(self.kin))
        else:
            P0st = ce1 + cemono * self.kin ** 2 / self.k_m ** 2
            P2st = cequad * self.kin ** 2 / self.k_m ** 2
            P4st = np.zeros(len(self.kin))
        if self.pardict["do_hex"]:
            Pst = [P0st, P2st, P4st]
        else:
            Pst = [P0st, P2st]

        return Plin, Ploop, Pct, Pst


# Holds all the data in a convenient dictionary
class FittingData:
    def __init__(self, pardict):

        x_data, fit_data, cov, cov_inv, chi2data, invcovdata, fitmask = self.read_data(pardict)
        winnames = np.loadtxt(pardict["winfile"], dtype=str)
        print(winnames)

        self.data = {
            "x_data": x_data,
            "fit_data": fit_data,
            "cov": cov,
            "cov_inv": cov_inv,
            "chi2data": chi2data,
            "invcovdata": invcovdata,
            "fitmask": fitmask,
            "shot_noise": pardict["shot_noise"],
            "windows": winnames,
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

    def read_pk(self, inputfile, step_size, skiprows):

        dataframe = pd.read_csv(
            inputfile,
            comment="#",
            skiprows=skiprows,
            delim_whitespace=True,
            names=["k", "pk0", "pk2", "pk4", "nk"],
            header=None,
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

        # Updated. Now reads files for every redshift bin and stores them consecutively. Also
        # deals with NGC+SGC data, concatenating it for every redshift bin.

        # Read in the data
        datafiles = np.loadtxt(pardict["datafile"], ndmin=1, dtype=str)
        nz = len(pardict["z_pk"])
        print(datafiles, nz)
        all_xdata = []
        all_fitmask = []
        all_fit_data = []
        for i in range(nz):
            x_data, fitmask, fit_data = self.get_some_data(pardict, datafiles[i])
            all_xdata.append(x_data)
            all_fitmask.append(fitmask)
            all_fit_data.append(fit_data)
        fitmask = np.concatenate(all_fitmask)
        fit_data = np.concatenate(all_fit_data)

        """# Read in, reshape and mask the covariance matrix
        cov_flat = np.array(pd.read_csv(pardict["covfile"], delim_whitespace=True, header=None))
        nin = len(x_data)
        cov_input = cov_flat[:, 2].reshape((3 * nin, 3 * nin))
        nx0, nx2 = len(x_data[0]), len(x_data[1])
        nx4 = len(x_data[2]) if pardict["do_hex"] else 0
        mask0, mask2, mask4 = fitmask[0][:, None], fitmask[1][:, None], fitmask[2][:, None]
        cov = np.zeros((nx0 + nx2 + nx4, nx0 + nx2 + nx4))
        cov[:nx0, :nx0] = cov_input[mask0, mask0.T]
        cov[:nx0, nx0 : nx0 + nx2] = cov_input[mask0, nin + mask2.T]
        cov[nx0 : nx0 + nx2, :nx0] = cov_input[nin + mask2, mask0.T]
        cov[nx0 : nx0 + nx2, nx0 : nx0 + nx2] = cov_input[nin + mask2, nin + mask2.T]
        if pardict["do_hex"]:
            cov[:nx0, nx0 + nx2 :] = cov_input[mask0, 2 * nin + mask4.T]
            cov[nx0 + nx2 :, :nx0] = cov_input[2 * nin + mask4, mask0.T]
            cov[nx0 : nx0 + nx2, nx0 + nx2 :] = cov_input[nin + mask2, 2 * nin + mask4.T]
            cov[nx0 + nx2 :, nx0 : nx0 + nx2] = cov_input[2 * nin + mask4, nin + mask2.T]
            cov[nx0 + nx2 :, nx0 + nx2 :] = cov_input[2 * nin + mask4, 2 * nin + mask4.T]"""

        # Read in, reshape and mask the covariance matrix
        cov_input = np.array(pd.read_csv(pardict["covfile"], delim_whitespace=True, header=None))
        cov = np.delete(np.delete(cov_input, ~fitmask, axis=0), ~fitmask, axis=1)

        # Invert the covariance matrix
        cov_lu, pivots, cov_inv, info = lapack.dgesv(cov, np.eye(len(cov)))

        chi2data = np.dot(fit_data, np.dot(cov_inv, fit_data))
        invcovdata = np.dot(fit_data, cov_inv)

        return all_xdata, fit_data, cov, cov_inv, chi2data, invcovdata, fitmask

    def get_some_data(self, pardict, datafile):

        print(datafile)
        if pardict["do_corr"]:
            data = np.array(pd.read_csv("datafile", delim_whitespace=True, header=None))
        else:
            data = self.read_pk(datafile, 1, 0)

        x_data = data[:, 0]
        """fitmask = [
            (np.where(np.logical_and(x_data >= pardict["xfit_min"][0], x_data <= pardict["xfit_max"][0]))[0]).astype(
                int
            ),
            (np.where(np.logical_and(x_data >= pardict["xfit_min"][1], x_data <= pardict["xfit_max"][1]))[0]).astype(
                int
            ),
            (np.where(np.logical_and(x_data >= pardict["xfit_min"][2], x_data <= pardict["xfit_max"][2]))[0]).astype(
                int
            ),
        ]"""
        ell = 3 if pardict["do_hex"] else 2
        fitmask = np.array(
            [np.logical_and(x_data >= pardict["xfit_min"][i], x_data <= pardict["xfit_max"][i]) for i in range(ell)]
        )
        x_data = np.array([data[fitmask[i], 0] for i in range(ell)])
        fit_data = np.concatenate([data[fitmask[i], i + 1] for i in range(ell)])

        return x_data, np.concatenate(fitmask), fit_data


def create_plot(pardict, fittingdata, plotindex=0):

    if pardict["do_hex"]:
        x_data = fittingdata.data["x_data"][plotindex]
        nx0, nx2, nx4 = len(x_data[0]), len(x_data[1]), len(x_data[2])
    else:
        x_data = fittingdata.data["x_data"][plotindex][:2]
        nx0, nx2, nx4 = len(x_data[0]), len(x_data[1]), 0
    fit_data = fittingdata.data["fit_data"]
    cov = fittingdata.data["cov"]

    ndata = nx0 + nx2 + nx4
    plt_data = (
        np.concatenate(x_data) ** 2 * fit_data[plotindex * ndata : (plotindex + 1) * ndata]
        if pardict["do_corr"]
        else np.concatenate(x_data) ** 1.0 * fit_data[plotindex * ndata : (plotindex + 1) * ndata]
    )
    if pardict["do_corr"]:
        plt_err = np.concatenate(x_data) ** 2 * np.sqrt(np.diag(cov)[plotindex * ndata : (plotindex + 1) * ndata])
    else:
        plt_err = np.concatenate(x_data) ** 1.0 * np.sqrt(np.diag(cov)[plotindex * ndata : (plotindex + 1) * ndata])

    plt.errorbar(
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
        )

    plt.xlim(np.amin(pardict["xfit_min"]) * 0.95, np.amax(pardict["xfit_max"]) * 1.05)
    if pardict["do_corr"]:
        plt.xlabel(r"$s\,(h^{-1}\,\mathrm{Mpc})$", fontsize=16)
        plt.ylabel(r"$s^{2}\xi(s)$", fontsize=16, labelpad=5)
    else:
        plt.xlabel(r"$k\,(h\,\mathrm{Mpc}^{-1})$", fontsize=16)
        plt.ylabel(r"$kP(k)\,(h^{-2}\,\mathrm{Mpc}^{2})$", fontsize=16, labelpad=5)
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


def update_plot(pardict, x_data, P_model, plt, keep=False, plot_index=0):

    if pardict["do_hex"]:
        nx0, nx2, nx4 = len(x_data[0]), len(x_data[1]), len(x_data[2])
    else:
        x_data = x_data[:2]
        nx0, nx2, nx4 = len(x_data[0]), len(x_data[1]), 0
    plt_data = np.concatenate(x_data) ** 2 * P_model if pardict["do_corr"] else np.concatenate(x_data) ** 1.0 * P_model

    plt10 = plt.errorbar(
        x_data[0],
        plt_data[:nx0],
        marker="None",
        color="r",
        linestyle="-",
        markeredgewidth=1.3,
        zorder=0,
    )
    plt11 = plt.errorbar(
        x_data[1],
        plt_data[nx0 : nx0 + nx2],
        marker="None",
        color="b",
        linestyle="-",
        markeredgewidth=1.3,
        zorder=0,
    )
    if pardict["do_hex"]:
        plt12 = plt.errorbar(
            x_data[2],
            plt_data[nx0 + nx2 :],
            marker="None",
            color="g",
            linestyle="-",
            markeredgewidth=1.3,
            zorder=0,
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


def update_plot_components(pardict, kin, P_components, plt, keep=False, comp_list=(True, True, True, True)):

    ls = [":", "-.", "--", "-"]
    labels = ["Linear", "Linear+Loop", "Linear+Loop+Counter", "Linear+Loop+Counter+Stoch"]
    kinfac = kin ** 2 if pardict["do_corr"] else kin ** 1.0

    part_comp = [np.zeros(len(kin)), np.zeros(len(kin)), np.zeros(len(kin))]
    for (line, comp, add, label) in zip(ls, P_components, comp_list, labels):
        for i, c in enumerate(comp):
            part_comp[i] += c
        if add:
            plt10 = plt.errorbar(
                kin,
                kinfac * part_comp[0],
                marker="None",
                color="r",
                linestyle=line,
                markeredgewidth=1.3,
                zorder=0,
                label=label,
            )
            plt11 = plt.errorbar(
                kin,
                kinfac * part_comp[1],
                marker="None",
                color="b",
                linestyle=line,
                markeredgewidth=1.3,
                zorder=0,
            )
            if pardict["do_hex"]:
                plt12 = plt.errorbar(
                    kin,
                    kinfac * part_comp[2],
                    marker="None",
                    color="g",
                    linestyle=line,
                    markeredgewidth=1.3,
                    zorder=0,
                )
    plt.legend()

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
    pardict["z_pk"] = np.array(pardict["z_pk"], dtype=float)
    print(np.shape(pardict["z_pk"]))
    if not any(np.shape(pardict["z_pk"])):
        pardict["z_pk"] = [float(pardict["z_pk"])]

    return pardict


def do_optimization(func, start):

    from scipy.optimize import basinhopping

    result = basinhopping(
        func,
        start,
        niter_success=10,
        niter=100,
        stepsize=0.01,
        minimizer_kwargs={
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


def get_Planck(filename, nfiles, usecols=(6, 29, 3, 2), raw=False):

    weights = []
    chain = []
    for i in range(1, nfiles):
        strin = str("%s_%d.txt" % (filename, i))
        data = np.array(pd.read_csv(strin, delim_whitespace=True, header=None))
        weights.append(data[:, 0])
        chain.append(data[:, list(usecols)])
    weights = np.concatenate(weights)
    chain = np.concatenate(chain)

    # Need to convert H0 to h if requested
    index = np.where(np.array(usecols) == 29)[0]
    if len(index) > 0:
        chain[:, index] /= 100.0

    if raw:
        return weights, chain
    else:
        cov = np.cov(chain, rowvar=False, aweights=weights)
        return np.average(chain, axis=0, weights=weights), cov, np.linalg.inv(cov)
