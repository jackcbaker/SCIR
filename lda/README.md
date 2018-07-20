# LDA Simulations

These simulations are based on code by [Patterson and Teh (2013)](http://papers.nips.cc/paper/4883-stochastic-gradient-riemannian-langevin-dynamics-on-the-probability-simplex) which is available on Github. The entry points for these simulations are `cir_run.py` and `rsgld_run.py`. Each script takes a command line argument from 1-5 which specifies the seed to set. The parameters are set to the optimal ones we found in the manuscript.

Before these scripts can be run the code by Patterson and Teh (2013) for the Gibbs sampler needs to be compiled. This is done by going into the `samplers/` directory and running `python setup.py build_ext --inplace`.

The code requires text data to function. We recommend downloading the data dumps from Wikipedia and parsing them using [`wikiextractor`](https://github.com/attardi/wikiextractor).
