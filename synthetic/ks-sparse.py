import numpy as np
import pandas as pd
from ggplot import *
from model import SparseDirichlet
from samplers.cir import CIR
from samplers.rsgld import RSGLD
from samplers.direct import Direct


def grid_comparison():
    for seed in xrange(5, 10):
        model = SparseDirichlet(seed)
        for n in [1, 10, 50, 100, 500]:
            for h in [1.0, 0.5, 1e-1, 5e-2, 1e-2, 5e-3, 1e-3]:
                print "CIR\t{0}\t{1}\t{2}".format(n, h, seed)
                np.random.seed(seed)
                sampler = CIR(model, h, n, verbose = False, M = 10 ** 3) 
                print sampler.score[-1]
                np.save("data/sparse/cir-score-{0}-{1}-{2}".format(h, n, seed), sampler.score)
                np.save("data/sparse/cir-times-{0}-{1}-{2}".format(h, n, seed), sampler.times)
            for h in [1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6, 1e-6]:
                print "RSGLD\t{0}\t{1}\t{2}".format(n, h, seed)
                np.random.seed(seed)
                sampler = RSGLD(model, h, n, verbose = False, M = 10 ** 3) 
                print sampler.score[-1]
                np.save("data/sparse/rsgld-score-{0}-{1}-{2}".format(h, n, seed), sampler.score)
                np.save("data/sparse/rsgld-times-{0}-{1}-{2}".format(h, n, seed), sampler.times)
        np.random.seed(seed)
        print "Direct\t{0}".format(seed)
        sampler = Direct(model, verbose = False, M = 10 ** 3)
        print sampler.score[-1]
        np.save("data/sparse/direct-score-{0}".format(seed), sampler.score)


def plot_n():
    scores, times = load_data()
    methods = ["CIR", "RSGLD", "Exact"]
    plot_frame = {"Method" : np.repeat(methods, 5).tolist(), 
            "Minibatch Size" : [1, 10, 50, 100, 500] * len(methods),
            "KS Distance" : [], "Lower" : [], "Upper" : []}
    for n in [1, 10, 50, 100, 500]:
        current = 1000.0 * np.ones(5)
        for h in scores["CIR"][n].keys():
            temp = scores['CIR'][n][h][-1, :]
            if temp.mean() < current.mean():
                current = temp
        plot_frame["KS Distance"].append(current.mean())
        plot_frame["Lower"].append(current.min())
        plot_frame["Upper"].append(current.max())
    for n in [1, 10, 50, 100, 500]:
        current = 1000.0 * np.ones(5)
        for h in scores["RSGLD"][n].keys():
            temp = scores['RSGLD'][n][h][-1, :]
            if temp.mean() < current.mean():
                current = temp
        plot_frame["KS Distance"].append(current.mean())
        plot_frame["Lower"].append(current.min())
        plot_frame["Upper"].append(current.max())
    for n in [1, 10, 50, 100, 500]:
        plot_frame["KS Distance"].append(scores["Direct"][n][-1, :].mean())
        plot_frame["Lower"].append(scores["Direct"][n][-1, :].min())
        plot_frame["Upper"].append(scores["Direct"][n][-1, :].max())
    plot_frame = pd.DataFrame(plot_frame)
    plot_frame.to_csv('data/sparse_n_processed.csv', index = False)


def load_data():
    scores = {"CIR" : {1 : {}, 10 : {}, 50 : {}, 100 : {}, 500 : {}}, 
            "RSGLD" : {1 : {}, 10 : {}, 50 : {}, 100 : {}, 500 : {}}, "Direct" : {}}
    times = {"CIR" : {1 : {}, 10 : {}, 50 : {}, 100 : {}, 500 : {}}, 
            "RSGLD" : {1 : {}, 10 : {}, 50 : {}, 100 : {}, 500 : {}}}
    for n in [1, 10, 50, 100, 500]:
        for h in [1.0 ,5e-1 ,1e-1, 5e-2, 1e-2, 5e-3, 1e-3]:
            scores['CIR'][n][h] = np.zeros((10 ** 3, 5))
            times['CIR'][n][h] = np.zeros((10 ** 3, 5))
            for i, seed in enumerate(xrange(5, 10)):
                scores['CIR'][n][h][:, i] = np.load(
                        "data/sparse/cir-score-{0}-{1}-{2}.npy".format(h, n, seed))
                times['CIR'][n][h][:, i] = np.load(
                        "data/sparse/cir-times-{0}-{1}-{2}.npy".format(h, n, seed))
        for h in [1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6, 1e-6]:
            scores['RSGLD'][n][h] = np.zeros((10 ** 3, 5))
            times['RSGLD'][n][h] = np.zeros((10 ** 3, 5))
            for i, seed in enumerate(xrange(5, 10)):
                scores['RSGLD'][n][h][:, i] = np.load(
                        "data/sparse/rsgld-score-{0}-{1}-{2}.npy".format(h, n, seed))
                times['RSGLD'][n][h][:, i] = np.load(
                        "data/sparse/rsgld-times-{0}-{1}-{2}.npy".format(h, n, seed))
        scores['Direct'][n] = np.zeros((10 ** 3, 5))
        for i, seed in enumerate(xrange(5, 10)):
            scores['Direct'][n][:, i] = np.load(
                    "data/sparse/direct-score-{0}.npy".format(seed))
    return scores, times


if __name__ == '__main__':
    grid_comparison()
    plot_n()
