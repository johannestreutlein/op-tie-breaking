from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
import numpy as np
import torch as th


class EpisodeCrossRunner:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run

        self.env = env_REGISTRY[self.args.env](args=args)

        self.episode_limit = self.env.episode_limit
        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        # Log the first run
        self.log_train_stats_t = -1000000

    def setup(self, scheme, groups, preprocess, mac1, mac2):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac1 = mac1
        self.mac2 = mac2

    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self):
        pass

    def close_env(self):
        self.env.close()

    def reset(self):
        self.batch = self.new_batch()
        self.env.reset()
        self.t = 0

    def run(self, test_mode=True):
        #this is a function for cross-play evaluation.
        #two different macs are loaded and then the two players are
        #controlled by the different macs


        #cross play only for testing!
        assert test_mode

        self.reset()

        terminated = np.zeros(self.args.batch_size_run)
        episode_return = 0
        self.mac1.init_hidden(batch_size=self.batch_size)
        self.mac2.init_hidden(batch_size=self.batch_size)

        while not terminated.all():

            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()]
            }

            self.batch.update(pre_transition_data, ts=self.t)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1

            actions1 = self.mac1.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
            actions2 = self.mac2.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)

            #concatenate actions from both macs and throw away the ones I don't need.
            actions = th.stack([actions1[:, 0], actions2[:, 1]], dim=1)


            reward, terminated, env_info = self.env.step(actions)
            episode_return += reward

            post_transition_data = {
                "actions": actions,
                "reward": [(reward,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }

            self.batch.update(post_transition_data, ts=self.t)

            self.t += 1

        last_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs()]
        }
        self.batch.update(last_data, ts=self.t)

        # Select actions in the last stored state

        actions1 = self.mac1.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
        actions2 = self.mac2.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)

        # concatenate actions from both macs and throw away the ones I don't need.
        actions = th.stack([actions1[:, 0], actions2[:, 1]], dim=1)

        self.batch.update({"actions": actions}, ts=self.t)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        self.t_env += self.t * self.batch_size

        cur_returns.append(episode_return)

        if len(self.test_returns) == max(1, self.args.test_nepisode//self.batch_size):
            self._log(cur_returns, cur_stats, log_prefix)

        return self.batch

    def _log(self, returns, stats, prefix):

        self.logger.log_stat(prefix + "return_mean", np.mean(returns, dtype=np.float64), self.t_env)

        #jackknife std leaving out one batch at a time
        jackknife_means = []
        idx = np.arange(len(returns))
        for i in idx:
            jackknife_means.append(np.array(returns, dtype=np.float64)[idx != i].mean())

        self.logger.log_stat(prefix + "return_jackknife_std", np.std(jackknife_means, dtype=np.float64), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], self.t_env)
        stats.clear()
