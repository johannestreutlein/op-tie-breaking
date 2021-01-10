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
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from collections import defaultdict

from utils.uniquify import uniquify

def plot_learning_curve(returns, t, args):
    mean = returns.mean(axis=1)
    dev = returns.std(axis=1)

    print('\nMeans:')
    print(mean)
    print('\nStandard deviations:')
    print(dev)

    x_vals = t

    fig1, ax1 = plt.subplots()
    ax1.plot(x_vals, mean)
    ax1.fill_between(x_vals, mean - dev, mean + dev, alpha=0.1)

    plt.xlabel('Environment steps')
    plt.ylabel('Average return')

    if args.save:
        filename = 'results/learning_curves/learning_curve_paths_{}.pdf'.format(os.path.basename(args.model_paths))
        filename = uniquify(filename)
        plt.savefig(filename)

    plt.show()

#the following function is slightly modified from https://stackoverflow.com/questions/42355122/can-i-export-a-tensorflow-summary-to-csv
def tabulate_events(summary_iterators):

    #tags = summary_iterators[0].Tags()['scalars']

    #for it in summary_iterators:
    #    assert it.Tags()['scalars'] == tags

    out = defaultdict(list)
    steps = []

    #for tag in tags:
    for tag in ['return_mean']:
        steps = [e.step for e in summary_iterators[0].Scalars(tag)]

        for index, acc in enumerate(summary_iterators):
            if len(acc.Scalars(tag))<16:
                print(index)

        i=0
        for events in zip(*[acc.Scalars(tag) for acc in summary_iterators]):
            if len(set(e.step for e in events)) != 1:
                print(i)
            #assert len(set(e.step for e in events)) == 1

            out[tag].append([e.value for e in events])
            i+=1

    return out, steps

def load_returns(args):

    print('\n\nLoading list of model paths from ' + args.model_paths)
    models = pd.read_csv(args.model_paths, index_col=0)

    summary_iterators = []
    for index in models.index:
        directory = os.path.basename(models.loc[index]['model_path'])
        tb_path = os.path.join('results', 'tb_logs', directory)
        for filename in os.listdir(tb_path):
            summary_iterators.append(EventAccumulator(os.path.join(tb_path, filename)).Reload())

    out, steps = tabulate_events(summary_iterators)

    returns = np.array(out['return_mean']) #first axis time, second axis different runs



    return returns, steps, args


if __name__ == '__main__':
    print('\n\n===============')
    print(datetime.datetime.now())
    # Define the parser
    parser = argparse.ArgumentParser(description='Short sample app')

    #needs a list of model paths, over which the returns during training are averaged
    parser.add_argument('--model_paths', action="store", dest='model_paths', default=None)
    parser.add_argument('--save', action="store", dest='save', default=True)

    # Now, parse the command line arguments and store the
    # values in the `args` variable
    args = parser.parse_args()

    returns, t, args = load_returns(args)

    plot_learning_curve(returns, t, args)

    print('\n\n')


