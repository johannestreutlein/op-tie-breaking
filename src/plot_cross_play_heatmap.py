import argparse
import numpy as np
import datetime
import pandas as pd

import json
import seaborn as sns
import matplotlib.pyplot as plt
import os
from utils.uniquify import uniquify

def plot_cross_play_heatmap(matrices, args):

    grid_kws = {"width_ratios": (.45, .45, .04), "wspace": .3, "hspace": .2}
    f, (ax1, ax2, cbar_ax) = plt.subplots(1, 3, gridspec_kw=grid_kws, figsize=(10, 4))

    ax1.title.set_text('Other-play')
    ax2.title.set_text('Other-play with tie-breaking')

    axes = [ax1, ax2]

    maxim = 0
    minim = 0

    for index, matrix in enumerate(matrices):

        print('return matrix:')
        print(matrix)
        off_diag = matrix[~np.eye(matrix.shape[0], dtype=bool)]
        print('off_diagonal_mean:')
        print(off_diag.mean())
        print('off_diagonal_std:')
        print(off_diag.std())

        maxim = max([maxim, matrix.max()])
        minim = min([minim, matrix.min()])

    for index, matrix in enumerate(matrices):

        heatmap = sns.heatmap(matrix, cmap='viridis', ax=axes[index], cbar=bool(index), cbar_ax=cbar_ax, vmax=maxim, vmin=minim)#linewidth=0.5,ticklabels=True, yticklabels=True
        heatmap.set_xlabel('Player 2 seed')
        heatmap.set_ylabel('Player 1 seed')
        heatmap.tick_params(length=0, labeltop=True, labelbottom=False)

    if args.save:
        filename = 'results/opt_with_tie_breaking/evalrun_{}_index_{}_cross_play_heatmap_{}.pdf'.format(args.run, args.index, args.env)
        filename = uniquify(filename)
        plt.savefig(filename)

    plt.show()

def load_data(args):

    filename = os.path.join('results/sacred', args.run, 'info.json')
    print('\n\nLoading returns from ' + filename)

    returns = []
    with open(filename) as json_file:
        data = json.load(json_file)
        for dict in data['test_return_mean']:
            returns.append(dict['value'])

    filename = os.path.join('results/sacred', args.run, 'config.json')
    print('\n\nLoading config from ' + filename)
    with open(filename) as json_file:
        config = json.load(json_file)

    args.env = config["env"]
    args.n_seeds_per_run = config["n_seeds_per_run"]

    matrices = []
    #pick out a run by its index.
    for index in args.index:
        relevant_returns = returns[index * args.n_seeds_per_run**2: (index + 1) * args.n_seeds_per_run**2]

    #make it into a matrix

        return_matrix = np.array(relevant_returns).reshape(args.n_seeds_per_run, args.n_seeds_per_run)
        matrices.append(return_matrix)

    return matrices, args


if __name__ == '__main__':
    print(datetime.datetime.now())
    # Define the parser
    parser = argparse.ArgumentParser(description='Short sample app')

    # sacred directory of the run of op_with_tie_breaking.py evaluation
    parser.add_argument('--run', action="store", dest='run', default=None)
    parser.add_argument('--save', action="store", dest='save', default=True)

    # the starting index of the run of which the cross-play value should be plotted to a heatmap
    # should be a list of two indices, the first one corresponding to other-play, the other one to op with tie-breaking
    parser.add_argument('--index', action="store", dest='index', default='[0,100]')
    # Now, parse the command line arguments and store the
    # values in the `args` variable
    args = parser.parse_args()

    print("\nargs:")
    print(args)

    args.index = json.loads(args.index)

    #masks = None

    #if args.hash_run is None:
    #    args.hash_run=args.run

    # Individual arguments can be accessed as attributes...

    matrices, args = load_data(args)

    plot_cross_play_heatmap(matrices, args)
