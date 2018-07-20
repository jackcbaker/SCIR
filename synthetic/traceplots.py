import numpy as np
import pandas as pd
from model import Dirichlet, SparseDirichlet
from samplers.cir import CIR
from samplers.rsgld import RSGLD
from samplers.direct import Direct


def sparse_comparison():
    model = SparseDirichlet()
    n = 10
    score_curr = 10000.0
    output_curr = np.zeros(10 ** 3)
    for h in [1.0, 0.5, 1e-1, 5e-2, 1e-2, 5e-3, 1e-3]:
        np.random.seed(11)
        print "CIR\t{0}".format(h)
        sampler = CIR(model, h, n, verbose = False, M = 10 ** 3) 
        if sampler.score[-1] < score_curr:
            score_curr = sampler.score[-1]
            output_curr = sampler.output
    print "Final Score: {0}".format(score_curr)
    np.savetxt("data/trace/cir-sparse-trace.dat", output_curr)
    score_curr = 10000.0
    output_curr = np.zeros(10 ** 3)
    for h in [1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6, 1e-6]:
        np.random.seed(11)
        print "RSGLD\t{0}".format(h)
        sampler = RSGLD(model, h, n, verbose = False, M = 10 ** 3) 
        if sampler.score[-1] < score_curr:
            score_curr = sampler.score[-1]
            output_curr = sampler.output
    np.savetxt("data/trace/rsgld-sparse-trace.dat", output_curr)
    print "Final Score: {0}".format(score_curr)
    print "Exact Sampler"
    np.random.seed(11)
    sampler = Direct(model, verbose = False, M = 10 ** 3) 
    np.savetxt("data/trace/exact-sparse-trace.dat", sampler.output)


if __name__ == '__main__':
    sparse_comparison()
