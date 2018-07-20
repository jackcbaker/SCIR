from __future__ import division

import random
import argparse
import itertools
import numpy as np
import os
import cPickle as pickle

import processwiki as pw
import samplers.cir as cir

home = os.path.expanduser('~')
np.set_printoptions(precision=3, suppress=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', type=float, default=0.005)
    parser.add_argument('--beta', type=float, default=1.)
    parser.add_argument('--K', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--epsilon', type=float, default=0.5)
    parser.add_argument('--tau0', type=float, default=10)
    parser.add_argument('--kappa', type=float, default=0.33)
    parser.add_argument('--samples_per_update', type=int, default=200)
    parser.add_argument('--num_updates', type=int, default=10 ** 3)
    parser.add_argument('--output_dir', type=str,default='.')
    parser.add_argument('--tuning', type=int,default=None)
    args = parser.parse_args()

    step_size_params = (args.epsilon, args.tau0, args.kappa)
    D = 317008
    # If tuning argument exists tune appropriately
    if args.tuning != None:
        step_size_params = (steps[args.tuning], tau0[args.tuning], \
                args.kappa, args.tuning)
        random.seed(args.tuning)

    vocab = pw.create_vocab('wiki.vocab')
    W = len(vocab)

    # Data is a list of form [(article name, article word cts)]
    data = pw.parse_docs(pw.online_wiki_docs(), vocab)

    # Use first 100 docs as held out test set
    test_data = itertools.islice(data, 1000)
    (test_names, holdout_train_cts, holdout_test_cts) = zip(*list(test_data))
    holdout_train_cts = list(holdout_train_cts)
    holdout_train_cts = list(holdout_train_cts)

    batched_cts = pw.take_every(args.batch_size, data)

    print "Running CIR Sampler, h = %g, samples per update = %d" % (step_size_params[0],
            args.samples_per_update)
    theta0 = np.random.gamma(1,1,(args.K,W))
    ld = cir.CIRSampler(D, args.K, W, args.alpha, args.beta, theta0,
            step_size_params, args.output_dir)
    ld_args = (args.num_updates, args.samples_per_update, batched_cts,
            holdout_train_cts, holdout_test_cts)
    ld.run_online(*ld_args)
    if args.tuning != None:
        np.savetxt('./cluster/tuning/perplex_cir{0}'.format(args.tuning), np.array(ld.ho_log_preds))
    else:
        np.savetxt('output/perplex_cir{0}'.format(args.tuning), np.array(ld.ho_log_preds))


if __name__ == '__main__':
    main()
