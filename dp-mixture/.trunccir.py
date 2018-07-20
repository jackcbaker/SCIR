from samplers.cirtrunc import CIR
from model.dpmodel import DPPARAFAC
from model.model import PARAFAC

model = PARAFAC()
DP = DPPARAFAC(model.D, alpha = 1.)

sampler = CIR(model, DP, kstar = 100, h_theta = .1, h_cir = .1, minibatch_size = 5000)
sampler.fit()
