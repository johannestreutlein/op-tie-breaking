import argparse
import matplotlib.pyplot as plt
import numpy as np
import datetime
import pandas as pd

import json
import seaborn as sns
import matplotlib.pyplot as plt
import os

from utils.uniquify import uniquify

def op_tie_breaking_evaluation(returns, hash_lists, args):
    '''This function takes in returns and lists of hashs. It gives the returns as a matrix to function analyze_equiv_classes
    to build up the equivalence classes of policies. Then, it prints the indices for each, including
    percentages of each class, and also the file path for one of each class (to be able to check which class that is by hand)
    . Lastly, it plots a histogram of hash values, overlaying the different classes.

    '''

    chosen_indices_dict = {}
    n_seeds_total = args.n_seeds_per_run
    if n_seeds_total == 0:
        n_seeds_total = int(np.sqrt(len(returns)))

    threshold = args.threshold
    number_of_runs = len(hash_lists[args.hash_function_seeds[args.index]]) // n_seeds_total

    #just care about this one hash seed
    hash_seed = args.hash_function_seeds[args.index]
    hash_list = hash_lists[hash_seed]

    #only one run, doing cross play between everything
    assert number_of_runs == 1
    assert n_seeds_total == int(np.sqrt(len(returns)))
    assert n_seeds_total == len(hash_list)

    print('\n\n=============================================')
    print('-----------seed of hash function: {}---------'.format(hash_seed))

    print('\n-----------number of total seeds: {}---------'.format(n_seeds_total))

    return_matrix = np.array(returns).reshape(n_seeds_total, n_seeds_total)

    #with np.printoptions(threshold=np.inf):
    #    print('\nreturn matrix:')
    #    print(return_matrix)

    print('\nhash_list:')
    print(hash_list)

    equiv_classes = analyze_equiv_classes(return_matrix, threshold)

    hashs_per_class = []

    data = pd.DataFrame(columns=['seed', 'hash', 'equiv_class'], index=range(n_seeds_total))

    for i, equiv_class in enumerate(equiv_classes):

        print('\nSize of class number {}:'.format(i))
        print('total: {}, percent: {:.2%}'.format(len(equiv_class), len(equiv_class)/n_seeds_total))
        hashs_per_class.append(np.array(hash_list)[equiv_class])
        print('\nExample run:')
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print('path: {}'.format(args.model_paths['model_path'].iloc[equiv_class[0]]))
            print('seed: {}'.format(args.model_paths['seed'].iloc[equiv_class[0]]))
        for j in equiv_class:
            data['seed'].iloc[j] = args.model_paths['seed'].iloc[j]
            data['hash'].iloc[j] = hash_list[j]
            data['equiv_class'].iloc[j] = i

    filename = os.path.join('results', 'analysis_of_equiv_classes', 'hash_seeds_per_class_run_{}_hash_run_{}_env_{}.csv'.format(args.run, args.hash_run, args.env))
    filename = uniquify(filename)

    print('\nSaving data to file {}'.format(filename))

    data.to_csv(filename)

    print('\nNow, print histogram\n\n')

    min_hash = np.array(hash_list).min()
    max_hash = np.array(hash_list).max()

    bins = np.linspace(min_hash, max_hash, args.n_bins)

    hashs_per_class = sorted(hashs_per_class, key=lambda x: -len(x))

    for i, hashs in enumerate(hashs_per_class):
        plt.hist(hashs, bins, alpha=0.5, label='Class {}'.format(i+1))

    plt.legend(loc='upper left')
    plt.title('{}'.format(args.title))
    plt.xlabel('Hash value')
    plt.ylabel('Number of policies')


    if args.save:
        filename = os.path.join('results', 'analysis_of_equiv_classes', 'plot_histogram_run_{}_hash_run_{}_env_{}.pdf'.format(args.run, args.hash_run, args.env))
        filename = uniquify(filename)
        plt.savefig(filename)

    plt.show()



def analyze_equiv_classes(return_matrix=None, threshold=0.5):
    '''this function takes in a return matrix, and dynamically builds up
    all the equivalence classes. it then outputs indices for all classes
    '''

    assert return_matrix is not None
    equiv_classes = []

    for policy in range(return_matrix.shape[0]):
        new_class = True
        for equiv_class in equiv_classes:
            avg_cross_play = (return_matrix[policy, equiv_class[0]] + return_matrix[equiv_class[0], policy]) / 2.
            equiv_class_sp = return_matrix[equiv_class[0], equiv_class[0]]

            sp = return_matrix[policy, policy]

            if avg_cross_play > equiv_class_sp - threshold and sp > equiv_class_sp - threshold:
                equiv_class.append(policy)
                new_class = False
                break

        if new_class:
            equiv_classes.append([policy])


    for i, equiv_class in enumerate(equiv_classes):
        print('\n-----looking at class no {}-----'.format(i))

        print('\nindices for class:')
        print(equiv_class)


    return equiv_classes

def load_info(args):
    returns = None

    filename = os.path.join('results', 'sacred', args.run, 'config.json')
    print('\n\nloading config from {}'.format(filename))
    with open(filename) as json_file:
        config = json.load(json_file)

    args.env = config["env"]
    args.seed = config["seed"]
    args.n_seeds_per_run = config["n_seeds_per_run"]

    filename = os.path.join('results', 'sacred', args.hash_run, 'config.json')
    print('\n\nloading config from {}'.format(filename))
    with open(filename) as json_file:
        config_hash = json.load(json_file)

    args.hash_function_seeds = config_hash["hash_function_seeds"]


    filename = os.path.join('results', 'sacred', args.hash_run, 'info.json')
    print('\n\nloading hashs, model paths from ' + filename)

    hash_lists = {}

    with open(filename) as json_file:
        data = json.load(json_file)

    args.model_paths = data['model_paths']

    hash_seed = args.hash_function_seeds[args.index]

    hashs = data['trajectories_hash_{}'.format(hash_seed)]

    print('\n')
    print('hash values seed {}:'.format(hash_seed))
    print(hashs)

    hash_lists[hash_seed] = hashs

    filename = os.path.join('results', 'sacred', args.run, 'info.json')
    print('\n\nloading returns from ' + filename)
    returns = []
    with open(filename) as json_file:
        data = json.load(json_file)
        for dict in data['test_return_mean']:
            returns.append(dict['value'])

        #the models that have been evaluated and that had their hash function calculated have to coincide
        assert args.model_paths == data['model_paths']

    args.model_paths = pd.DataFrame(args.model_paths)

    print('\nModel paths:')
    print(args.model_paths)

    return returns, hash_lists, args


if __name__ == '__main__':
    print(datetime.datetime.now())
    # Define the parser
    parser = argparse.ArgumentParser(description='Short sample app')

    parser.add_argument('--run', action="store", dest='run', default=None) #sacred directory of the run from cross-play evaluation
    parser.add_argument('--hash_run', action="store", dest='hash_run', default=None) #sacred directory of the run from hash calculation
    parser.add_argument('--threshold', action="store", dest='threshold', default='0.6') #threshold for payoffs for calculating different equivalence classes
    parser.add_argument('--index', action="store", dest='index', default='0') #index of seed of hash function to be used
    parser.add_argument('--n_bins', action="store", dest='n_bins', default='100') #number of bins for histogram
    parser.add_argument('--save', action="store", dest='save', default='True') #save plot?
    parser.add_argument('--title', action="store", dest='title', default='') #title of plot

    # Now, parse the command line arguments and store the
    # values in the `args` variable
    args = parser.parse_args()

    print("\nargs:")
    print(args)

    args.threshold = json.loads(args.threshold)
    args.index = json.loads(args.index)
    args.n_bins = json.loads(args.n_bins)

    if args.hash_run is None:
        args.hash_run = args.run

    hash_lists, returns, args = load_info(args)

    op_tie_breaking_evaluation(hash_lists, returns, args)

    print('\n\n')


