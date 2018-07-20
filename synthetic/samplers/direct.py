import time
import numpy as np
from score import Score
from model import Dirichlet

class Direct:

    def __init__(self, model, M = 10 ** 4, verbose = True):
        self.model = model
        # Parameters for simulation
        self.M = M          # Num iterations
        self.iter = 0
        # Initial parameters
        theta = self.model.start.copy()
        self.omega = theta / theta.sum()
        # Containers and create scoring object used to assess the inference
        self.times = np.zeros(self.M)
        self.score = np.zeros(self.M)
        self.output = np.zeros((self.M, self.model.d))
        self.verbose = verbose
        self.assess = Score(self)
        self.simulate()

    def simulate(self):
        for self.iter in xrange(self.M):
            a = self.model.truth.copy()
            self.omega = np.random.dirichlet(a)
        for self.iter in xrange(self.M):
            a = self.model.truth.copy()
            self.omega = np.random.dirichlet(a)
            self.score[self.iter] = self.assess.score(self)
            self.output[self.iter,:] = self.omega
            if self.verbose == True:
                print "{0}\t{1}".format(self.iter, self.score[self.iter])


if __name__ == '__main__':
    model = Dirichlet()
    Direct(model)
