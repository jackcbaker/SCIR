import numpy as np

class Dirichlet:
    """This class just stores some useful parameters for the model to keep things consistent"""

    def __init__(self, seed = 13):
        np.random.seed(seed)
        self.d = 10
        self.z = np.random.choice(self.d, size = 10 ** 3)
        self.alpha = 0.1 * np.ones(self.d)
        self.truth = np.array([(self.z == j).sum() for j in xrange(self.d)]) + self.alpha
        self.init = np.random.choice(self.d, size = 10 ** 4)
#        self.start = np.array([(self.init == j).sum() for j in xrange(self.d)]) + self.alpha
        self.start = self.truth.copy()

    def subsample(self, sampler):
        minibatch = np.random.choice(self.z, size = sampler.n)
        Nhat = np.array([(minibatch == j).sum() for j in xrange(self.d)])
        return self.z.size / float(sampler.n) * Nhat


class SparseDirichlet:
    """This class just stores some useful parameters for the model to keep things consistent"""

    def __init__(self, seed = 13):
        np.random.seed(seed)
        self.d = 10
        self.z = np.array([0] * 800 + [1] * 10 ** 2 + [2] * 10 ** 2)
        self.alpha = 0.1 * np.ones(self.d)
        self.truth = np.array([(self.z == j).sum() for j in xrange(self.d)]) + self.alpha
        self.init = np.random.choice(self.d, size = 10 ** 3)
        self.start = np.array([(self.init == j).sum() for j in xrange(self.d)]) + self.alpha

    def subsample(self, sampler):
        minibatch = np.random.choice(self.z, size = sampler.n)
        Nhat = np.array([(minibatch == j).sum() for j in xrange(self.d)])
        return self.z.size / float(sampler.n) * Nhat


if __name__ == '__main__':
    model = SparseDirichlet(seed = 8)
    print model.truth
