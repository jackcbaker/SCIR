from scipy.stats import itemfreq
import numpy as np
from dp import DP


class DPPARAFAC:

    def __init__(self, d, alpha = 1.0, alpha0 = .1):
        # Initial alpha params
        self.alpha = alpha
        self.d = d
        self.G0 = G0Dir(self.d, alpha0)
        self.F = FMult(self.d)
    
    def sample_theta_cond(self, data_k, k, sampler):
        a = self.G0.a * np.ones(self.d)
        for user in data_k:
            for index, count in user.iteritems():
                a[index] += count
        sampler.theta_raw[k] = np.random.gamma(a)
        sampler.theta[k] = sampler.theta_raw[k] / sampler.theta_raw[k].sum()

    def cir(self, data_k, k, sampler):
        a = np.zeros(self.d)
        for user in data_k:
            for index, count in user.iteritems():
                a[index] += count
        a *= sampler.correction
        a += self.G0.a * np.ones(self.d)
        theta_j = sampler.theta_raw[k].copy()
        sampler.theta_raw[k] = self.cir_raw(a, theta_j, sampler.h_theta)
        sampler.theta[k] = sampler.theta_raw[k] / sampler.theta_raw[k].sum()

    def cir_raw(self, a, gamma, h):
        nu = 2 * a.copy()
        mu = 2 * gamma.copy() * np.exp(-h) / (1 - np.exp(-h))
        return np.random.noncentral_chisquare(nu, mu) * (1 - np.exp(-h)) / 2.
    
    def sgld(self, data_k, k, sampler):
        m = np.zeros(self.d)
        for user in data_k:
            for index, count in user.iteritems():
                m[index] += count
        m *= sampler.correction
        theta_j = sampler.theta_raw[k].copy()
        sampler.theta_raw[k] = self.sgld_raw(m, theta_j, sampler)
        sampler.theta[k] = sampler.theta_raw[k] / sampler.theta_raw[k].sum()

    def sgld_raw(self, n, theta, sampler):
        alpha = self.G0.a * np.ones(self.d)
        out = theta.copy()
        out += sampler.h_theta * (alpha - theta + n - theta / theta.sum() * n.sum())
        out += np.sqrt(2. * theta * sampler.h_theta) * np.random.normal(size = self.d)
        return np.abs(out)

    def sample_alpha(self, sampler):
        a = 1. + sampler.zstar
        b = 1. - np.log(1 - sampler.v[:sampler.zstar]).sum()
        self.alpha = np.random.gamma(a, 1 / b)


class G0Dir:

    def __init__(self, d, alpha0):
        self.a = alpha0
        self.d = d

    def dens(self, theta):
        raise ValueError("Not implemented yet")
        return mvnorm.pdf(theta, self.loc, self.scale)

    def random(self, sampler):
        sampler.theta_raw.append(np.random.gamma(self.a * np.ones(self.d)))
        sampler.theta.append(sampler.theta_raw[-1] / sampler.theta_raw[-1].sum())

    def gradlogdens(self, theta):
        raise ValueError("Not implemented yet")
        return - np.matmul(self.scale_inv, (theta - self.loc))


class FMult:

    def __init__(self, d):
        self.d = d

    def dens(self, x, theta):
        return np.prod([theta[ind] ** n for ind, n in x.iteritems()])

    def logdens(self, x, theta):
        return sum(n * np.log(theta[ind]) for ind, n in x.iteritems())

    def random(self, theta, size = None):
        raise ValueError("Not implemented yet")
        return mvnorm.rvs(theta, self.scale, size = size)

    def gradlogdens(self, x, theta):
        raise ValueError("Not implemented yet")
        return np.matmul(self.scale_inv, (x - theta))
