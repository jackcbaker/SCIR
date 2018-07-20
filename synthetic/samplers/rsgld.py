import time
import numpy as np
from score import Score
from model import Dirichlet

class RSGLD:

    def __init__(self, model, h, n, M = 10 ** 3, verbose = True):
        self.h = h
        self.model = model
        self.M = M      # num iters
        self.n = n
        # Declare initial params
        self.theta = self.model.start.copy()
        self.omega = self.theta / self.theta.sum()
        self.iter = 0
        # Containers for output, create scoring object to assess convergence
        self.times = np.zeros(self.M)
        self.score = np.zeros(self.M)
        self.output = np.zeros((self.M, self.model.d))
        self.verbose = verbose
        self.assess = Score(self)
        self.simulate()

    def simulate(self):
        # Burn-in chain
        for self.iter in xrange(self.M):
            self.update()
        # Main run
        for self.iter in xrange(self.M):
            t = time.time()
            self.update()
            self.times[self.iter] = time.time() - t
            self.score[self.iter] = self.assess.score(self)
            self.output[self.iter,:] = self.omega
            if self.verbose == True:
                print "{0}\t{1}\t{2}".format(self.iter, self.score[self.iter], self.times[self.iter])

    def update(self):
        """One iteration of RSGLD by Teh et. al"""
        theta_prev = self.theta.copy()
        self.theta += self.h / 2. * self.naturalgrad() + self.injected_noise(theta_prev)
        self.theta = np.abs(self.theta)
        self.omega = self.theta / self.theta.sum()

    def naturalgrad(self):
        """Calculate natural gradient for Riemannian sampler, use expanded-mean parameterization"""
        return self.theta * self.logpostgrad() + 1

    def logpostgrad(self):
        Nhat = self.model.subsample(self)
        return self.model.alpha - self.theta + Nhat - self.omega * Nhat.sum()

    def injected_noise(self, theta):
        return np.sqrt(theta * self.h) * np.random.normal(size = self.model.d)

if __name__ == '__main__':
    model = Dirichlet()
    RSGLD(model, 1e-2, 10)
