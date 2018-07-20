from samplers.sgldtrunc import SGLD
from model.dpmodel import DPPARAFAC
from model.model import PARAFAC

model = PARAFAC()
DP = DPPARAFAC(model.D, alpha = 1.)

sampler = SGLD(model, DP, kstar = 100, h_theta = .01, h_cir = .01, minibatch_size = 5000)
sampler.fit()
