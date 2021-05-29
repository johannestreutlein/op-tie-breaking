import argparse
import matplotlib.pyplot as plt
import numpy as np
import datetime
import pandas as pd

import json

import os

from utils.uniquify import uniquify

def op_tie_breaking_evaluation(hash_lists, args):
    '''This function evaluates our method, other-play with tie-breaking. It applies the tie-breaking function to different training runs to
    choose policies and
    then starts an experiment that calculates cross-play values for the chosen policies.

    It would probably be better to run this with sacred, as part of an experiment, instead
    of an extra python file, etc., but this suffices for now.
    '''

    chosen_indices_dict = {}
    n_seeds_total = args.n_seeds_per_run
    number_of_runs = len(hash_lists[args.hash_function_seeds[0]]) // args.n_seeds_per_run

    for hash_seed, hashs in hash_lists.items():
        print('\n\n=============================================')
        print('-----------seed of hash function: {}---------'.format(hash_seed))

        n_seeds = 1 #the case of one seed represents simply doing other-play

        chosen_indices_dict[hash_seed] = {}

        while n_seeds <= n_seeds_total:
            print('\n\n----------- seeds per run: {}--------'.format(n_seeds))
            print('-------------------------------------')

            chosen_indices = []

            for index in range(number_of_runs):
                print('\n-------- run {} -------'.format(index))

                hash_list = np.array(hashs[index*n_seeds_total:index*n_seeds_total+n_seeds_total])
                hash_list = hash_list[:n_seeds]

                print('\nhash_list:')
                print(hash_list)

                chosen_indices.append(op_with_tie_breaking(hash_list))

            print('\nchosen indices:')
            print(chosen_indices)

            chosen_indices_dict[hash_seed][n_seeds] = chosen_indices
            n_seeds *= 2

    print('\nDoing a new cross play evaluation with the chosen models, for each hash-function seed\n\n')

    #now, constructing new csv with model paths
    #very inefficient with a loop and creating copies, but it doesn't matter at the moment

    new_model_paths = pd.DataFrame()

    for n_seeds in range(int(np.floor(np.log2(n_seeds_total))) + 1):
        for hash_seed in args.hash_function_seeds:
            for runs, chosen_policy in enumerate(chosen_indices_dict[hash_seed][2 ** n_seeds]):
                new_model_paths = new_model_paths.append(args.model_paths.iloc[runs * n_seeds_total + chosen_policy], ignore_index=True)

    print('Prepared model paths:')
    print(new_model_paths)

    filename = os.path.join('results', 'op_with_tie_breaking', 'chosen_model_list_hash_run_{}.csv'.format(args.hash_run))

    filename = uniquify(filename)

    new_model_paths.to_csv(filename)

    os.system('python3 src/main.py --env-config={} --config=policy_gradient \
        with seed={} \
        evaluate=True \
        cross_play=True \
        calculate_hash=False \
        test_nepisode={} \
        n_seeds_per_run={} \
        hash_function_seeds={} \
        model_paths={}'.format(args.env, args.seed, args.test_nepisode, number_of_runs,
                               str(args.hash_function_seeds).replace(' ', ''), filename))
    #here, each run corresponds to a chosen amount of seeds and each seed is a chosen policy from a different run

def op_with_tie_breaking(hashes):
    '''this implements other-play with tie-breaking.
    input is a list of tie-breaking values, output is index of policy with highest value.
    The actual values are calculated in the hash-run, by a method of the environment
    '''

    best_policy = hashes.argmax()

    print('\nIndex chosen policy:')
    print(best_policy)

    return best_policy

def load_info(args):
    returns = None

    filename = os.path.join('results', 'sacred', args.hash_run, 'config.json')
    print('\n\nloading config from {}'.format(filename))
    with open(filename) as json_file:
        config = json.load(json_file)

    args.env = config["env"]
    args.hash_function_seeds = config["hash_function_seeds"]

    print(args.hash_function_seeds)

    filename = os.path.join('results', 'sacred', args.hash_run, 'info.json')
    print('\n\nloading hashs from ' + filename)

    hash_lists = {}

    with open(filename) as json_file:
        data = json.load(json_file)

    for seed in args.hash_function_seeds:
        hashs = data['trajectories_hash_{}'.format(seed)]
        print('\n')
        print('hash values seed {}:'.format(seed))
        print(hashs)
        hash_lists[seed] = hashs

    args.model_paths = data['model_paths']

    args.model_paths = pd.DataFrame(args.model_paths)

    print('\nModel paths:')
    print(args.model_paths)

    return hash_lists, args


if __name__ == '__main__':
    print(datetime.datetime.now())
    # Define the parser
    parser = argparse.ArgumentParser(description='Short sample app')

    parser.add_argument('--hash_run', action="store", dest='hash_run', default=None) #sacred directory of the run from hash calculation
    parser.add_argument('--test_nepisode', action="store", dest='test_nepisode', default='2048') #number of episodes to use to calculate cross-play values
    parser.add_argument('--seed', action="store", dest='seed', default='100') # seed for environment and policy randomness
    parser.add_argument('--n_seeds_per_run', action="store", dest='n_seeds_per_run', default='32') # seed for environment and policy randomness


    # Now, parse the command line arguments and store the
    # values in the `args` variable
    args = parser.parse_args()
    args.test_nepisode = json.loads(args.test_nepisode)
    args.seed = json.loads(args.seed)
    args.n_seeds_per_run= json.loads(args.n_seeds_per_run)

    hash_lists, args = load_info(args)

    op_tie_breaking_evaluation(hash_lists, args)

    print('\n\n')


