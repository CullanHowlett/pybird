import numpy as np
from pyDOE import lhs


def create_lhc(ndims=5, nsamp=10000):

    lhc = lhs(ndims, samples=nsamp, criterion="maximin")
    print(lhc)
    np.save(str("lhc_%dD_10000.npy" % ndims), lhc)


if __name__ == "__main__":
    create_lhc()
