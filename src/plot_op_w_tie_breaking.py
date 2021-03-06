import argparse
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import datetime
import pandas as pd

import json
import seaborn as sns
import matplotlib.pyplot as plt
import os
import matplotlib as mpl

from utils.uniquify import uniquify

def hash_method_plot(cross_play_results, args):
    #plots a graph with average off-diagonal cross-play values

    mean = cross_play_results.mean(axis=1)
    dev = cross_play_results.std(axis=1)

    print('Latex code for table of results:')
    resultstring = ""
    for i, m in enumerate(mean):
        resultstring += "& {:.2f} (\\textpm {:.2f}) ".format(m, dev[i])
    print(resultstring)

    x_vals = 2**np.arange(args.n_different_seeds)

    mpl.rcParams.update({
        "font.size": 9
    })
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42

    plt.figure(figsize=(2.9, 2.6))

    ax1 = plt.axes()
    plt.plot(x_vals, mean, '-')
    plt.plot(x_vals, [args.optimum for i in range(cross_play_results.shape[0])], '--')
    plt.fill_between(x_vals, mean - dev, mean + dev, alpha=0.1)

    ax1.set_xscale('log', basex=2)
    ax1.set_xticks(x_vals)
    ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    plt.title('{}'.format(args.title))
    plt.xlabel('Number of seeds used for tie-breaking')
    plt.ylabel('Avg off-diagonal cross-play value')


    plt.tight_layout()

    if args.save:
        filename = 'results/op_with_tie_breaking/evalrun_{}_op_tie_breaking_plot_{}.pdf'.format(args.run, args.env)
        filename = uniquify(filename)
        plt.savefig(filename)


    plt.show()

def load_cross_play_results(args):

    filename = 'results/sacred/' + args.run + '/config.json'
    print('\n\nloading config from ' + filename)
    with open(filename) as json_file:
        config = json.load(json_file)

    args.env = config["env"]
    args.seed = config["seed"]
    args.n_seeds_per_run = config["n_seeds_per_run"]
    args.hash_function_seeds = config["hash_function_seeds"]

    filename = 'results/sacred/' + args.run + '/info.json'
    print('\n\nloading returns from ' + filename)
    returns = []
    with open(filename) as json_file:
        data = json.load(json_file)
        for dict in data['test_return_mean']:
            returns.append(dict['value'])

    args.n_total_runs = len(returns) // args.n_seeds_per_run ** 2

    #here, each "seed per run" is actually an independent run used to evaluate the hash method

    args.n_different_seeds = args.n_total_runs // len(args.hash_function_seeds)

    cross_play_results = np.zeros((args.n_different_seeds, len(args.hash_function_seeds)))

    print('Number of returns: {}'.format(len(returns)))
    print('Number of independent runs that have been considered: {}'.format(args.n_seeds_per_run))
    print('Number of different amounts of seeds: {}'.format(args.n_different_seeds))
    print('Number of hash function seeds: {}'.format(len(args.hash_function_seeds)))

    for n_seeds in range(args.n_different_seeds):
        for hash_seed in range(len(args.hash_function_seeds)):
            start_index = (n_seeds * len(args.hash_function_seeds) + hash_seed) * args.n_seeds_per_run ** 2
            end_index = (n_seeds * len(args.hash_function_seeds) + hash_seed + 1) * args.n_seeds_per_run ** 2
            return_matrix = np.array(returns[start_index: end_index]).reshape(args.n_seeds_per_run, args.n_seeds_per_run)
            off_diag = return_matrix[~np.eye(return_matrix.shape[0], dtype=bool)]

            print('\n\n-----------\nNumber of seeds: {}'.format(2**n_seeds))
            print('Hash seed index: {}'.format(hash_seed))
            print('off_diagonal_mean:')
            print(off_diag.mean())

            print('off_diagonal_std:')
            print(off_diag.std())

            cross_play_results[n_seeds, hash_seed] = off_diag.mean()

    return cross_play_results, args


if __name__ == '__main__':
    print('\n\n===================================')
    print(datetime.datetime.now())
    # Define the parser
    parser = argparse.ArgumentParser(description='Short sample app')

    parser.add_argument('--run', action="store", dest='run', default=None) #sacred directory of the run generated by op_with_tie_breaking.py
    parser.add_argument('--save', action="store", dest='save', default=True) #whether to save results to file
    parser.add_argument('--title', action="store", dest='title', default='') #title for plot
    parser.add_argument('--optimum', action="store", dest='optimum', default='0.5')

    # Now, parse the command line arguments and store the
    # values in the `args` variable
    args = parser.parse_args()

    args.optimum = float(args.optimum)

    cross_play_results, args = load_cross_play_results(args)

    hash_method_plot(cross_play_results, args)

    print('\n\n')


