import random
import numpy as np
import argparse
from samplers.cir import CIR
from model.dpmodel import DPPARAFAC
from model.model import PARAFAC

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha0', type=float, default=.5)
    parser.add_argument('--alpha', type=float, default=1.)
    parser.add_argument('--K', type=int, default=20)
    parser.add_argument('--n', type=int, default=1000)
    parser.add_argument('--h_theta', type=float, default=.1)
    parser.add_argument('--h_cir', type=float, default=.1)
    parser.add_argument('--n_iters', type=int, default=10 ** 3)
    parser.add_argument('--tuning', type=int,default=None)
    args = parser.parse_args()

    if args.tuning != None:
        seed = range(5)
        args.seed = seed[args.tuning]

    model = PARAFAC()
    DP = DPPARAFAC(model.d, alpha = args.alpha, alpha0 = args.alpha0)

    sampler = CIR(model, DP, kstar = args.K, h_theta = args.h_theta, 
            h_cir = args.h_cir, minibatch_size = args.n)
    random.seed(args.seed)
    np.random.seed(args.seed)
    sampler.fit(n_iters = args.n_iters, seed = args.seed)

if __name__ == '__main__':
    main()
