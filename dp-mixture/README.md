# Dirichlet Process Mixture Simulations

The entry points for these simulations are `scir_run.py` and `srgld_run.py`. Each script takes a command line argument from 1-5 which specifies the seed to set. The parameters are set to the optimal ones we found in the manuscript.

Before these scripts can be run the data needs to be parsed, currently it is saved in its raw form. Navigate to `model/data` and run `python parse.py`. The data is the [Anonymous Microsoft Web Data Data Set](https://archive.ics.uci.edu/ml/datasets/Anonymous+Microsoft+Web+Data) from the UCI Machine Learning Repository [1].

[1] J. Breese, D. Heckerman., C. Kadie _Empirical Analysis of Predictive Algorithms for Collaborative Filtering_ Proceedings of the Fourteenth Conference on Uncertainty in Artificial Intelligence, Madison, WI, July, 1998.
