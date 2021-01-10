import torch as th
import numpy as np
from zlib import crc32
import torch.nn as nn
import torch.nn.functional as F

class HashFunction(nn.Module):
    #this implements the neural network for hashing

    def __init__(self, input_shape, seed, args):
        with th.random.fork_rng():
            #we set the random seed here to coordinate between the different principals
            th.manual_seed(seed)
            np.random.seed(seed)

            super(HashFunction, self).__init__()
            self.args = args

            self.fc1 = nn.Linear(input_shape, args.hash_hidden_dim).double()
            self.fc2 = nn.Linear(args.hash_hidden_dim, args.hash_hidden_dim).double()
            self.fc3 = nn.Linear(args.hash_hidden_dim, args.hash_hidden_dim).double()
            self.fc4 = nn.Linear(args.hash_hidden_dim, args.hash_hidden_dim).double()
            self.fc5 = nn.Linear(args.hash_hidden_dim, 1).double()

    def forward(self, inputs):
        h1 = F.relu(self.fc1(inputs*np.sqrt(inputs.shape[-1])))
        h2 = F.relu(self.fc2(h1*np.sqrt(h1.shape[-1])))
        h3 = F.relu(self.fc3(h2*np.sqrt(h2.shape[-1])))
        h4 = F.relu(self.fc4(h3*np.sqrt(h3.shape[-1])))
        hashs = self.fc5(h4*np.sqrt(h4.shape[-1]))
        return hashs

def hash_function(histories, seed, args, logger):
    hash_type = args.hash_type

    if hash_type == 'cov':
        histories = np.array(histories.T,
                             dtype=np.float64)  # prepare for calculating moments
        histories -= histories.mean(axis=1, keepdims=True)  # center each random variable

        logger.console_logger.info(
            'Calculating central moments with {} samples of {}-dim RV'.format(histories.shape[0], histories.shape[1]))
        cov_matrix = np.cov(histories) #could also calculate higher moments potentially
        logger.console_logger.info(
            'Cov matrix:\n' + str(cov_matrix))
        history_hash = cov_matrix.mean()
        logger.console_logger.info(
            'Cov matrix mean: ' + str(history_hash))

    if hash_type == 'nn':
        input_shape = histories.shape[-1]
        hash_network = HashFunction(input_shape, seed, args)
        input = th.tensor(histories, dtype=th.float64)
        hashs = hash_network.forward(input)
        history_hash = hashs.mean().item()

    return history_hash