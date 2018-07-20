import os
import numpy as np

class CIR:

    def __init__(self, model, DP, h_theta = 5e-4, h_cir = 1e-3, minibatch_size = 100, kstar = 10):
        self.DP = DP
        self.K = kstar
        self.model = model
        self.kstar = kstar
        self.h_theta = h_theta
        self.h_cir = h_cir
        self.minibatch_size = minibatch_size
        self.correction = self.model.N / float(self.minibatch_size)
        # Remove previous log file
        self.__rm_log()
        self.__initialise()

    def fit(self, n_iters = 10**3, seed = None):
        for self.iter in xrange(n_iters):
            self.mcmc_step()
            score_curr = self.model.score_sampler(self)
            print "{0}\t{1}\t{2}\t{3}\t{4}".format(self.iter, self.DP.alpha,
                    score_curr, self.mnzeros, self.cstar)
            self.__write_log(score_curr, seed)

    def mcmc_step(self):
        # For components defined by Z:
        # Sample stick breaking params
        self.__sample_omega()
        # Sample new component locations
        self.__sample_theta()
        # Sample slice variables and compute u*
        self.__sample_u()
        # Try to sample new alpha if function exists
        self.__sample_alpha()
        # Sample empty stick breaks
        self.__add_omega()
        # Sample empty components
        self.__add_theta()
        # Sample allocations given new components
        self.__sample_z()

    def __initialise(self):
        self.theta_raw = []
        self.theta = []
        # Sample initial theta from prior
        for i in xrange(self.kstar):
            self.DP.G0.random(self)
        # Initial allocations and counts
        self.z = np.random.randint(0, self.kstar, self.model.N)
        self.zstar = self.z.max() + 1
        self.minibatch = np.random.choice(self.model.N, self.minibatch_size, replace = False)
        self.alloc = [(self.z[self.minibatch] == j) for j in xrange(self.zstar)]
        self.mhat = np.array([alloc_j.sum() for alloc_j in self.alloc])
        self.mhat = self.correction * self.mhat
        # Initial slice parameters
        self.u = np.random.rand(self.model.N)
        self.ustar = self.u[self.minibatch].min()
        # Initial omega parameters
        self.gamma = np.zeros((self.zstar, 2))
        self.gamma[:,0] = np.random.gamma(np.ones(self.zstar))
        self.gamma[:,1] = np.random.gamma(self.DP.alpha * np.ones(self.zstar))
        self.v = self.gamma[:,0] / self.gamma.sum(axis = 1)
        self.omega = [self.v[j] * (1. - self.v[:j]).prod() for j in xrange(self.zstar)]
        self.omega = np.array(self.omega)

    def __sample_omega(self):
        M = np.array([self.mhat[j:].sum() for j in xrange(1, self.zstar + 1)])
        self.__simulate_beta(1. + self.mhat, self.DP.alpha + M)
        self.omega = [self.v[j] * (1. - self.v[:j]).prod() for j in xrange(self.zstar)]
        self.omega = np.array(self.omega)
        self.omegastar = 1. - self.omega.sum()
    
    def __sample_theta(self):
        for j in xrange(self.zstar):
            data_j = np.take(self.model.data, np.flatnonzero(self.alloc[j]), 0)
            self.DP.cir(data_j, j, self)

    def __sample_u(self):
        self.u[self.minibatch] = np.random.uniform(high = self.omega[self.z[self.minibatch]])
        self.ustar = self.u[self.minibatch].min()

    def __sample_alpha(self):
        try:
            self.DP.sample_alpha(self)
        except AttributeError:
            pass

    def __add_omega(self):
        gamma_new = []
        v_new = []
        omega_new = []
        while self.omegastar > self.ustar:
            gamma0 = np.random.gamma(1)
            gamma1 = np.random.gamma(self.DP.alpha)
            gamma_new.append([gamma0, gamma1])
            v_j = gamma0 / (gamma0 + gamma1)
            v_new.append(v_j)
            omega_new.append(self.omegastar * v_j)
            self.omegastar *= (1. - v_j)
        try:
            self.gamma = np.concatenate((self.gamma, gamma_new))
        except ValueError:
            pass
        self.v = np.concatenate((self.v, v_new))
        self.omega = np.concatenate((self.omega, omega_new))
        self.cstar = self.omega.size

    def __add_theta(self):
        num_toadd = xrange(len(self.theta), self.cstar)
        for j in num_toadd:
            self.DP.G0.random(self)

    def __sample_z(self):
        for i in self.minibatch:
            x_i = np.take(self.model.data, i, axis = 0)
            wts = []
            active_allocs = []
            for j in xrange(self.cstar):
                if self.omega[j] > self.u[i]:
                    wts.append(self.DP.F.dens(x_i, self.theta[j]))
                    active_allocs.append(j)
            wts = np.array(wts)
            self.z[i] = np.random.choice(active_allocs, p = wts / wts.sum())
        self.zstar = self.z.max() + 1
        self.minibatch = np.random.choice(self.model.N, self.minibatch_size, replace = False)
        self.alloc = [(self.z[self.minibatch] == j) for j in xrange(self.zstar)]
        self.mhat = np.array([alloc_j.sum() for alloc_j in self.alloc], dtype = float)
        self.mhat *= self.correction
        self.mnzeros = (self.mhat > 0).sum()
        self.v = self.v[:self.zstar]
        self.gamma = self.gamma[:self.zstar,:]

    def __simulate_beta(self, a, b):
        self.__simulate_cir(a, 0)
        self.__simulate_cir(b, 1)
        self.v = self.gamma[:,0] / self.gamma.sum(axis = 1)

    def __simulate_cir(self, a, j):
        h = self.h_cir
        nu = 2 * a
        mu = 2 * self.gamma[:,j] * np.exp(-h) / (1 - np.exp(-h))
        self.gamma[:,j] = np.random.noncentral_chisquare(nu, mu) * (1 - np.exp(-h)) / 2.

    def __rm_log(self):
        try:
            os.remove('output/cir/log-{0}-{1}'.format(self.h_cir, self.h_theta))
        except OSError:
            pass

    def __write_log(self, score, seed):
        with open('output/cir/log-{0}'.format(seed), 'a') as outfile:
            outfile.write("{0}\t{1}\t{2}\t{3}\t{4}\n".format(self.iter, self.DP.alpha,
                    score, self.mnzeros, self.cstar))
