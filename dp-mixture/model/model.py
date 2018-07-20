import numpy as np
import pickle

class PARAFAC:
    
    def __init__(self, seed = 13, N = 10**4):
        # Set default values
        self.load_dataset()
        self.N = len(self.data)
        self.test_size = len(self.test)
        self.d = 294

    def load_dataset(self):
        with open('model/data/train.pkl') as infile:
            self.data = pickle.load(infile)
        with open('model/data/test.pkl') as infile:
            self.test = pickle.load(infile)

    def score_sampler(self, sampler):
        """Score sampler using log predictive density of held out dataset"""
        wts = sampler.omega / sampler.omega.sum()
        score = 0.
        for i, x_i in enumerate(self.test):
            score_curr = 0.
            for k in xrange(sampler.zstar):
                dens = np.prod([sampler.theta[k][ind] ** count for ind, count in x_i.iteritems()])
                score_curr += wts[k] * dens
            score -= np.log(score_curr)
        return score / self.test_size

if __name__ == '__main__':
    DPGaussMNIST()
