import numpy as np
import os
import time
import sys
from itertools import combinations
import findiff
from configobj import ConfigObj

sys.path.append("../")
from tbird.Grid import grid_properties


def get_template_grids(parref, nmult=2, nout=2):
    # order_i is the number of points away from the origin for parameter i
    # The len(freepar) sub-arrays are the outputs of a meshgrid, which I feed to findiff
    outgrid = parref["outgrid"]
    name = parref["gridname"]

    plin = np.load(os.path.join(outgrid, "TablePlin_template_%s.npy" % name))
    plin = plin.reshape((plin.shape[0], nmult, plin.shape[-2] // nmult, plin.shape[-1]))
    ploop = np.load(os.path.join(outgrid, "TablePloop_template_%s.npy" % name))
    ploop = ploop.reshape((ploop.shape[0], nmult, ploop.shape[-2] // nmult, ploop.shape[-1]))

    # The output is not concatenated for multipoles
    return plin[..., :nout, :, :], ploop[..., :nout, :, :]


def get_grids(parref, nmult=2, nout=2, pad=True, read_params=True):
    # order_i is the number of points away from the origin for parameter i
    # The len(freepar) sub-arrays are the outputs of a meshgrid, which I feed to findiff
    outgrid = parref["outgrid"]
    name = parref["gridname"]

    # Coordinates have shape (len(freepar), 2 * order_1 + 1, ..., 2 * order_n + 1)
    shapecrd = np.concatenate([[len(parref["freepar"])], np.full(len(parref["freepar"]), 2 * int(parref["order"]) + 1)])
    padshape = [(1, 1)] * (len(shapecrd) - 1)

    # grids need to be reshaped and padded at both ends along the freepar directions
    params = np.load(os.path.join(outgrid, "TableParams_%s.npy" % name))
    params = params.reshape((*shapecrd[1:], params.shape[-1]))
    if pad:
        params = np.pad(params, padshape + [(0, 0)], "constant", constant_values=0)

    plin = np.load(os.path.join(outgrid, "TablePlin_%s.npy" % name))
    plin = plin.reshape((*shapecrd[1:], nmult, plin.shape[-2] // nmult, plin.shape[-1]))
    if pad:
        plin = np.pad(plin, padshape + [(0, 0)] * 3, "constant", constant_values=0)

    ploop = np.load(os.path.join(outgrid, "TablePloop_%s.npy" % name))
    ploop = ploop.reshape((*shapecrd[1:], nmult, ploop.shape[-2] // nmult, ploop.shape[-1]))
    if pad:
        ploop = np.pad(ploop, padshape + [(0, 0)] * 3, "constant", constant_values=0)

    # The output is not concatenated for multipoles
    return params, plin[..., :nout, :, :], ploop[..., :nout, :, :]


def get_pder_lin(parref, pi, dx, filename):
    """ Calculates the derivative aroud the Grid.valueref points. Do this only once.
    gridshape is 2 * order + 1, times the number of free parameters
    pi is of shape gridshape, n multipoles, k length, P columns (zeroth being k's)"""
    # Findiff syntax is Findiff((axis, delta of uniform grid along the axis, order of derivative, accuracy))
    t0 = time.time()
    lenpar = len(parref["freepar"])
    idx = int(parref["order"]) + 1

    p0 = pi[idx, idx, idx, idx, ...]
    t1 = time.time()
    print("Done p0 in %s sec" % str(t1 - t0))

    dpdx = np.array([findiff.FinDiff((i, dx[i], 1), acc=4)(pi)[idx, idx, idx, idx, ...] for i in range(lenpar)])
    t0 = time.time()
    print("Done dpdx in %s sec" % str(t0 - t1))

    # Second derivatives
    d2pdx2 = np.array([findiff.FinDiff((i, dx[i], 2), acc=2)(pi)[idx, idx, idx, idx, ...] for i in range(lenpar)])
    t1 = time.time()
    print("Done d2pdx2 in %s sec" % str(t1 - t0))

    d2pdxdy = np.array(
        [
            [i, j, findiff.FinDiff((i, dx[i], 1), (j, dx[j], 1), acc=2)(pi)[idx, idx, idx, idx, ...]]
            for (i, j) in combinations(range(lenpar), 2)
        ]
    )
    t0 = time.time()
    print("Done d2pdxdy in %s sec" % str(t0 - t1))

    # Third derivatives: we only need it for A_s, so I do this by hand
    d3pdx3 = np.array([findiff.FinDiff((i, dx[i], 3))(pi)[idx, idx, idx, idx, ...] for i in range(lenpar)])
    t1 = time.time()
    print("Done d3pdx3 in %s sec" % str(t1 - t0))

    d3pdx2dy = np.array(
        [
            [i, j, findiff.FinDiff((i, dx[i], 2), (j, dx[j], 1))(pi)[idx, idx, idx, idx, ...]]
            for (i, j) in combinations(range(lenpar), 2)
        ]
    )
    t0 = time.time()
    print("Done d3pdx2dy in %s sec" % str(t0 - t1))

    d3pdxdy2 = np.array(
        [
            [i, j, findiff.FinDiff((i, dx[i], 1), (j, dx[j], 2))(pi)[idx, idx, idx, idx, ...]]
            for (i, j) in combinations(range(lenpar), 2)
        ]
    )
    t1 = time.time()
    print("Done d3pdxdy2 in %s sec" % str(t1 - t0))

    d3pdxdydz = np.array(
        [
            [i, j, k, findiff.FinDiff((i, dx[i], 1), (j, dx[j], 1), (k, dx[k], 1))(pi)[idx, idx, idx, idx, ...]]
            for (i, j, k) in combinations(range(lenpar), 3)
        ]
    )
    t0 = time.time()
    print("Done d3pdxdydz in %s sec" % str(t0 - t1))

    d4pdx4 = np.array([findiff.FinDiff((i, dx[i], 4))(pi)[idx, idx, idx, idx, ...] for i in range(lenpar)])
    t1 = time.time()
    print("Done d4pdx4 in %s sec" % str(t1 - t0))

    d4pdx3dy = np.array(
        [
            [i, j, findiff.FinDiff((i, dx[i], 3), (j, dx[j], 1))(pi)[idx, idx, idx, idx, ...]]
            for (i, j) in combinations(range(lenpar), 2)
        ]
    )
    t0 = time.time()
    print("Done d4pdx3dy in %s sec" % str(t0 - t1))

    d4pdxdy3 = np.array(
        [
            [i, j, findiff.FinDiff((i, dx[i], 1), (j, dx[j], 3))(pi)[idx, idx, idx, idx, ...]]
            for (i, j) in combinations(range(lenpar), 2)
        ]
    )
    t1 = time.time()
    print("Done d4pdxdy3 in %s sec" % str(t1 - t0))

    d4pdx2dydz = np.array(
        [
            [i, j, k, findiff.FinDiff((i, dx[i], 2), (j, dx[j], 1), (k, dx[k], 1))(pi)[idx, idx, idx, idx, ...]]
            for (i, j, k) in combinations(range(lenpar), 3)
        ]
    )
    t0 = time.time()
    print("Done d4pdx2dydz in %s sec" % str(t0 - t1))

    d4pdxdy2dz = np.array(
        [
            [i, j, k, findiff.FinDiff((i, dx[i], 1), (j, dx[j], 2), (k, dx[k], 1))(pi)[idx, idx, idx, idx, ...]]
            for (i, j, k) in combinations(range(lenpar), 3)
        ]
    )
    t1 = time.time()
    print("Done d4pdxdy2dz in %s sec" % str(t1 - t0))

    d4pdxdydz2 = np.array(
        [
            [i, j, k, findiff.FinDiff((i, dx[i], 1), (j, dx[j], 1), (k, dx[k], 2))(pi)[idx, idx, idx, idx, ...]]
            for (i, j, k) in combinations(range(lenpar), 3)
        ]
    )
    t0 = time.time()
    print("Done d4pdxdydz2 in %s sec" % str(t0 - t1))

    d4pdxdydzdzm = np.array(
        [
            [
                i,
                j,
                k,
                m,
                findiff.FinDiff((i, dx[i], 1), (j, dx[j], 1), (k, dx[k], 1), (m, dx[m], 1))(pi)[
                    idx, idx, idx, idx, ...
                ],
            ]
            for (i, j, k, m) in combinations(range(lenpar), 4)
        ]
    )
    t1 = time.time()
    print("Done d4pdxdydzdm in %s sec" % str(t1 - t0))

    allder = (
        p0,
        dpdx,
        d2pdx2,
        d2pdxdy,
        d3pdx3,
        d3pdx2dy,
        d3pdxdy2,
        d3pdxdydz,
        d4pdx4,
        d4pdx3dy,
        d4pdxdy3,
        d4pdx2dydz,
        d4pdxdy2dz,
        d4pdxdydz2,
        d4pdxdydzdzm,
    )
    np.save(filename, allder)
    return allder


if __name__ == "__main__":

    # Read in the config file and total number of jobs
    configfile = sys.argv[1]
    pardict = ConfigObj(configfile)

    # Get the grid properties
    valueref, delta, flattenedgrid, truecrd = grid_properties(pardict)

    print("Let's start!")
    t0 = time.time()
    paramsgrid, plingrid, ploopgrid = get_grids(pardict)
    print("Got grids in %s seconds" % str(time.time() - t0))
    print("Calculate derivatives of params")
    get_pder_lin(pardict, paramsgrid, delta, os.path.join(pardict["outgrid"], "DerParams_%s.npy" % pardict["gridname"]))
    exit()
    print("Calculate derivatives of linear PS")
    get_pder_lin(pardict, plingrid, delta, os.path.join(pardict["outgrid"], "DerPlin_%s.npy" % pardict["gridname"]))
    print("Calculate derivatives of loop PS")
    get_pder_lin(pardict, ploopgrid, delta, os.path.join(pardict["outgrid"], "DerPloop_%s.npy" % pardict["gridname"]))
