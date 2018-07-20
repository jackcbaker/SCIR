import time
import numpy as np
from score import Score
from model import Dirichlet

class CIR:

    def __init__(self, model, h, n, M = 10 ** 4, verbose = True):
        # CIR parameters (all based on input h)
        self.h = np.sqrt(h)
        self.a = np.sqrt(h)
        self.s = 2 * self.a
        self.model = model
        # Parameters for simulation
        self.M = M          # Num iterations
        self.iter = 0
        self.n = n
        # Initial parameters
        self.theta = self.model.start.copy()
        self.omega = self.theta / self.theta.sum()
        # Containers and create scoring object used to assess the inference
        self.times = np.zeros(self.M)
        self.score = np.zeros(self.M)
        self.output = np.zeros((self.M, self.model.d))
        self.verbose = verbose
        self.assess = Score(self)
        self.simulate()

    def simulate(self):
        # Burn-in chain
        for self.iter in xrange(self.M):
            self.cir_step()
        # Main run
        for self.iter in xrange(self.M):
            t = time.time()
            self.cir_step()
            self.times[self.iter] = time.time() - t
            self.score[self.iter] = self.assess.score(self)
            self.output[self.iter,:] = self.omega
            if self.verbose == True:
                print "{0}\t{1}\t{2}".format(self.iter, self.score[self.iter], self.times[self.iter])

    def cir_step(self):
        # Declare mean parameter using subsampling
        b = self.model.subsample(self) + self.model.alpha
        # Create parameters for simulating from chi square propagation distn
        df = 4 * self.a * b / self.s
        c = 2 * self.a / ((1 - np.exp( - self.a * self.h )) * self.s)
        nc_param = 2 * c * self.theta * np.exp( - self.a * self.h )
        self.theta = np.random.noncentral_chisquare(df, nc_param) / (2 * c)
        # Normalise thetas to get dirichlet sample
        self.omega = self.theta / self.theta.sum()


if __name__ == '__main__':
    model = Dirichlet()
    sampler = CIR(model, 0.05, 10, M = 10 ** 3)
    print sampler.score[-1]
