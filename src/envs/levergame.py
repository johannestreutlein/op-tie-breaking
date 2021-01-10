from envs.multiagentenv import MultiAgentEnv
from utils.dict2namedtuple import convert
import numpy as np
import pandas as pd
import os
from numpy.random import default_rng
from utils.uniquify import uniquify


class LeverGame(MultiAgentEnv):
    def __init__(self, args):

        if isinstance(args, dict):
            args = convert(args)

        if isinstance(args.env_args, dict):
            args.env_args = convert(args.env_args)

        self.args = args

        # Unpack arguments from sacred

        self.device = self.args.device

        self.bs = self.args.batch_size_run

        # I treat randomness here different from other parts of the code. not sure whether that's bad/how bad that is
        self.rng = default_rng(seed=self.args.env_args.seed)
        self.extra_action = self.args.env_args.extra_action


        # Define the agents
        self.n_players = 2
        self.n_agents = 2
        if self.extra_action:
            self.size_A = 3
        else:
            self.size_A = 2

        self.n_obs = self.size_A

        self.n_actions = self.size_A


        self.r_success = self.args.env_args.r_success
        self.r_failure = self.args.env_args.r_failure

        self.n_comm_steps = self.args.env_args.n_comm_steps

        self.obs_size = 2

        self.state_size = 5

        self.episode_limit = self.args.env_args.episode_limit



    def reset(self):
        """ Returns initial observations and states"""

        self.steps = 0

        if self.extra_action:
            permutations_A = np.array([[0, 1, 2], [1, 0, 2]])
        else:
            permutations_A = np.array([[0, 1], [1, 0]])

        #Z stand for messages, which are not relevant in matchingpennies/matchingpennies2, but still implemented
        permutations_Z = np.array([[0, 1], [1, 0]])

        perm_choice_A = self.rng.choice(2, size=(self.bs, self.n_players))
        perm_choice_Z = self.rng.choice(2, size=(self.bs, self.n_players))

        self.perm_A=permutations_A[perm_choice_A]
        self.perm_Z=permutations_Z[perm_choice_Z]
        self.perm_A_inv=permutations_A[perm_choice_A]
        self.perm_Z_inv=permutations_Z[perm_choice_Z]

        self.state = np.concatenate([perm_choice_A, perm_choice_Z, np.zeros((self.bs,1))], axis=1)

        info = {}

        self.info = info

        observations = self.get_obs()

        return observations, self.state

    def step(self, action_codes):
        """ Returns reward, terminated, info """

        actions = action_codes.cpu()

        # now permute everything according to the sampled automorphisms

        if self.steps < self.n_comm_steps:
            self.Z = np.stack([
                self.perm_Z[:, i].T[actions[:, i]].diagonal() for i in range(self.n_players)
            ], axis=1)
        else:
            self.A = np.stack([
                self.perm_A[:, i].T[actions[:, i]].diagonal() for i in range(self.n_players)
            ], axis=1)

        #implemented only for two players....
        if self.n_players > 2:
            raise NotImplementedError("Environment only implemented for 2 players")

        reward = np.zeros(self.bs)

        if self.steps >= self.n_comm_steps and self.steps < self.episode_limit - 1 - self.args.env_args.extra_action:
            reward[(self.A[:, 0] == self.A[:, 1])] = self.r_success
            reward[(self.A[:, 0] != self.A[:, 1])] = self.r_failure

            reward[(self.A[:, 0] == 2)] = self.r_failure
            reward[(self.A[:, 1] == 2)] = self.r_failure

        elif self.steps>=self.episode_limit - 1 - self.extra_action and self.steps< self.episode_limit-1:
            reward[self.A[:, 1] == 2] = self.r_success
            reward[self.A[:, 1] != 2] = 0

        if self.steps < self.episode_limit-1:
            terminated = np.zeros(self.bs)

        else:
            terminated = np.ones(self.bs)

        self.steps += 1


        return reward, terminated, self.info

    def get_obs(self):
        """ Returns all agent observations in an array of shape (self.bs, self.n_players)"""

        if self.n_players > 2:
            raise NotImplementedError

        if self.steps == 0:
            observations = np.zeros((self.bs, self.n_players, self.obs_size))

        elif self.steps <= self.n_comm_steps:

            #irrelevant since we never use messages, but still implemented
            msg_list = [self.perm_Z_inv[:, i].T[self.Z[:,j]].diagonal()[:, np.newaxis] for i in range(self.n_players) for j in range(self.n_players)]

            msg0 = np.concatenate(msg_list[:2],axis=1)
            msg1 = np.concatenate(msg_list[2:],axis=1)[:, np.array([1,0])]

            observations = np.stack([msg0, msg1], axis=1)

        else:
            #this also applies the automorphisms to the actions, to get observations for both players
            act_list = [self.perm_A_inv[:, i].T[self.A[:,j]].diagonal()[:, np.newaxis] for i in range(self.n_players) for j in range(self.n_players)]
            act0 = np.concatenate(act_list[:2], axis=1)
            act1 = np.concatenate(act_list[2:], axis=1)[:, np.array([1, 0])]

            observations = np.stack([act0, act1], axis=1)

        return observations

    def get_obs_agent(self, agent_id):
        """ Returns observation for agent_id """
        return self.get_obs()[:, agent_id]

    def get_obs_size(self):
        """ Returns the shape of the observation """
        return self.obs_size

    def get_state(self):
        self.state[:, 4] = self.steps
        return self.state

    def get_state_size(self):
        """ Returns the shape of the state"""
        return self.state_size

    def get_avail_actions(self):
        if self.extra_action:
            #only for the asymmetriclevergame
            avail_actions = np.ones((self.bs, self.n_players, 3))
        else:
            avail_actions = np.ones((self.bs, self.n_players, 2))
        return avail_actions

    def get_avail_agent_actions(self, agent_id):
        """ Returns the available actions for agent_id """
        return self.get_avail_actions()[:, agent_id, :]

    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
        return self.n_actions

    def save_replay(self):
        raise NotImplementedError

    def get_stats(self):
        return None

    def render(self):
        raise NotImplementedError

    def close(self):
        pass

    def seed(self):
        raise NotImplementedError

    def save_episodes(self, batch_list):
        #this is a method for saving the histories for a batch to a file so it is possible
        #to examine what the agents are doing

        title = '_'.join([self.args.name, self.args.env_args.title, 'seed_' + str(self.args.seed), '.txt'])

        #this has to use the normal episode runner, not the cross-play
        if self.args.cross_play:
            raise NotImplementedError
        else:
            save_path = os.path.join("results/episodes", os.path.basename(self.args.checkpoint_path))

        if not os.path.isdir(save_path):
            os.makedirs(save_path)
            print('creating new directory ' + save_path)

        filename = uniquify(os.path.join(save_path, title))

        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', -1)

        with open(filename, 'w') as f:
            f.write('Path:' + str(self.args.checkpoint_path))

            f.write('Seed: ' + str(self.args.seed) + '\n\n')

            index = 0
            return_list = []

            for runner in batch_list:
                for i in range(runner.batch_size):
                    #for each episode, this reads out the relevant data and stores it in a data frame
                    #the dataframe is then printed to a file

                    test_return = runner.data.transition_data['reward'][i].flatten()[:self.episode_limit-1].sum().cpu().item()
                    return_list.append(test_return)

                    action_codes = np.array(runner.data.transition_data['actions'][i].T[0].cpu())

                    observations0 = np.array(runner.data.transition_data['obs'][i][:,0].T.cpu())
                    observations1 = np.array(runner.data.transition_data['obs'][i][:,1].T.cpu())

                    rewards = np.array(runner.data.transition_data['reward'][i].T.cpu())
                    terminated = np.array(runner.data.transition_data['terminated'][i].T.cpu())

                    data = np.concatenate([action_codes, rewards, terminated], axis=0)

                    data = np.concatenate([data, observations0, observations1], axis=0)

                    dataframe_index = ['actions_codes_1', 'action_codes_2', 'rewards', 'terminated', 'obs00',
                                       'obs01', 'obs10', 'obs11']

                    dataframe = pd.DataFrame(data, dataframe_index)

                    f.write('Episode ' + str(index) + '\n')

                    f.write('Test return: ' + str(test_return) + '\n\n')

                    f.write('Episode data:\n')
                    print(dataframe, file=f)

                    f.write('\n\n')

                    index += 1

            f.write("Mean return: " + str(np.array(return_list).mean()))
        print("Test episodes saved at: %s" % filename)

    def prepare_histories(self, batch_list):
        '''prepares histories into a matrix of shape batch_size * history_length, for hashing.
        This picks out one representative of all the relabelings for each history'''

        episode_list = []

        for batch in batch_list:

            # implementing this only for one specific game for now
            actions = batch.data.transition_data['actions']

            rewards = batch.data.transition_data['reward'].squeeze()

            player_relabelings = np.array([[0,1],[1,0]])

            prepared_histories = []
            actions = actions.squeeze()

            #sum over all agent permutations
            for perm_player in player_relabelings:

                #this picks one representative for both players actions.
                # That is the one representative that starts with 0, next unique action is 1, etc.
                new_a = np.zeros((2,actions.shape[0],3))
                for i in range(2):
                    new_actions = np.zeros((actions.shape[0], 3))
                    repeat_a01 = actions[:,0,perm_player[i]] == actions[:,1,perm_player[i]]
                    new_actions[~repeat_a01, 1] = 1.

                    if self.extra_action:
                        repeat_a21 = actions[:,2,perm_player[i]] == actions[:,1,perm_player[i]]
                        repeat_a20 = actions[:,2,perm_player[i]] == actions[:,0,perm_player[i]]

                        new_actions[repeat_a21, 2] = new_actions[repeat_a21, 1]
                        new_actions[repeat_a20, 2] = 0.
                        new_actions[~(repeat_a21 | repeat_a20), 2] = new_actions[~(repeat_a21 | repeat_a20), 1]+1

                    new_a[i] = new_actions

                if self.args.env_args.extra_action:
                    prepared_histories.append(np.concatenate([#new_payoff_matrix,
                         new_a[0].squeeze(),
                         new_a[1].squeeze(),
                         rewards[:, :3]
                         ], axis=1))
                else:
                    prepared_histories.append(np.concatenate([#new_payoff_matrix,
                         new_a[0][:, :2].squeeze(),
                         new_a[1][:, :2].squeeze(),
                         rewards[:, :2]
                         ], axis=1))

            episode_list.append(np.concatenate(prepared_histories, axis=0))

        action_list = np.concatenate(episode_list,
                                     axis=0)  # first axis episodes, second axis features

        return action_list