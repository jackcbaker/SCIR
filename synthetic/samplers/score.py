import numpy as np
from scipy.stats import beta, kstest

class Score:
    """
    Assess how well each inference procedure is doing.

    First a Rosenblatt transformation is applied using the true posterior Dirichlet(N + alpha).
    If the variables were realised from the truth then the transformed variables z will be
    independent and uniformly distributed. So to assess convergence we simply measure how
    close each z_j variable is to the uniform using the Kolmogorov-Smirnov statistic,
    and take the average score over all dimensions j.
    """

    def __init__(self, sampler):
        self.model = sampler.model
        self.z = np.zeros((self.model.d - 1, sampler.M))
        self.alpha = self.model.truth.copy()

    def score(self, sampler):
        self.rosenblatt(sampler)
        return self.score_uniform(sampler)

    def rosenblatt(self, sampler):
        """Apply Rosenblatt transformation"""
        for j in xrange(self.model.d - 1):
            # Calculate a1 and a2, the parameters of the corresponding conditional distns
            a1 = self.alpha[j]
            a2 = self.alpha[(j+1):].sum()
            # Normalise to ensure omega is sampled from a Beta
            correction = 1.0 / (1.0 - sampler.omega[:j].sum())
            self.z[j, sampler.iter] = beta.cdf(sampler.omega[j] * correction, a1, a2)

    def score_uniform(self, sampler):
        """
        Test each Rosenblatt transformed variable for uniformity using Kolmogorov-Smirnov.
        Average result to make score
        """
        score = 0.0
        for j in xrange(self.model.d - 1):
            score += kstest(self.z[j, :(sampler.iter + 1)], "uniform").statistic
        return score / float(self.model.d - 1)
