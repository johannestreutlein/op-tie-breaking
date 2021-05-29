import datetime
import os
import pprint
import time
import threading
import torch as th
import numpy as np
import pandas as pd
from types import SimpleNamespace as SN
from utils.logging import Logger
from utils.timehelper import time_left, time_str
from os.path import dirname, abspath

from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from controllers import REGISTRY as mac_REGISTRY
from components.episode_buffer import ReplayBuffer
from components.transforms import OneHot
from utils.hash_function import hash_function


def run(_run, _config, _log, pymongo_client):

    # check args sanity
    _config = args_sanity_check(_config, _log)

    args = SN(**_config)
    args.device = "cuda" if args.use_cuda else "cpu"

    # setup loggers
    logger = Logger(_log)

    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config,
                                       indent=4,
                                       width=1)
    _log.info("\n\n" + experiment_params + "\n")

    # configure tensorboard logger
    unique_token = "{}__{}".format(args.name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    args.unique_token = unique_token
    if args.use_tensorboard:
        tb_logs_direc = os.path.join(dirname(dirname(abspath(__file__))), "results", "tb_logs")
        tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(unique_token)
        logger.setup_tb(tb_exp_direc)

    # sacred is on by default
    logger.setup_sacred(_run)

    # Run and train
    if args.cross_play and args.evaluate:
        run_sequential_cross(args=args, logger=logger)
    else:
        run_sequential(args=args, logger=logger)

    # Clean up after finishing
    print("Exiting Main")

    if pymongo_client is not None:
        print("Attempting to close mongodb client")
        pymongo_client.close()
    print("Mongodb client closed")


    #had to comment out the following, otherwise this terminates my filestorage observer before it
    #saves the results.

    #print("Stopping all threads")
    #for t in threading.enumerate():
    #    if t.name != "MainThread":
    #        print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
    #        t.join(timeout=1)
    #        print("Thread joined")

    #print("Exiting script")

    # Making sure framework really exits
    #os._exit(os.EX_OK)


def evaluate_sequential(args, runner, logger, self_play=True):
    batch_list = []
    episode = 0

    n_test_runs = max(1, args.test_nepisode // runner.batch_size)
    for _ in range(n_test_runs):
        if args.save_episodes or args.calculate_hash:
            batch_list.append(runner.run(test_mode=True))
        else:
            runner.run(test_mode=True)
        episode += args.batch_size_run

    if args.save_replay:
        runner.save_replay()

    if args.save_episodes:
        runner.env.save_episodes(batch_list)

    if args.calculate_hash and self_play:
        histories = runner.env.prepare_histories(batch_list)
        logger.console_logger.info('Prepared {} ground-truth action-observation-histories for hashing'.format(histories.shape[0]))

        for seed in args.hash_function_seeds:
            trajectories_hash = hash_function(histories, seed, args, logger)
            logger.log_stat("trajectories_hash_{}".format(seed), trajectories_hash, runner.t_env)

    logger.log_stat("episode", episode, runner.t_env)
    log_str = "Test Stats | t_env: {:>10} | Episode: {:>8}\n".format(*logger.stats["episode"][-1])
    i = 0
    for (k, v) in sorted(logger.stats.items()):
        if k == "episode":
            continue
        i += 1
        window = 1
        item = "{:.4f}".format(np.mean([x[1] for x in logger.stats[k][-window:]]))
        log_str += "{:<25}{:>8}".format(k + ":", item)
        log_str += "\n" if i % 4 == 0 else "\t"
    logger.console_logger.info(log_str)

    runner.close_env()

def run_sequential(args, logger):

    # Init runner so we can get env info
    runner = r_REGISTRY[args.runner](args=args, logger=logger)

    # Set up schemes and groups here
    env_info = runner.get_env_info()
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]
    args.n_obs = env_info['n_obs']

    # Default/Base scheme
    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents", "dtype": th.float32},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }
    groups = {
        "agents": args.n_agents
    }
    preprocess = {
        "actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])
    }

    buffer = ReplayBuffer(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                          preprocess=preprocess,
                          device="cpu" if args.buffer_cpu_only else args.device)

    # Setup multiagent controller here
    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)

    # Give runner the scheme
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)

    # Learner
    learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)

    if args.use_cuda:
        learner.cuda()

    if args.checkpoint_path != "":
        timesteps = []
        timestep_to_load = 0

        if not os.path.isdir(args.checkpoint_path):
            logger.console_logger.info("Checkpoint directiory {} doesn't exist".format(args.checkpoint_path))
            return

        # Go through all files in args.checkpoint_path
        for name in os.listdir(args.checkpoint_path):
            full_name = os.path.join(args.checkpoint_path, name)
            # Check if they are dirs the names of which are numbers
            if os.path.isdir(full_name) and name.isdigit():
                timesteps.append(int(name))

        if args.load_step == 0:
            # choose the max timestep
            timestep_to_load = max(timesteps)
        else:
            # choose the timestep closest to load_step
            timestep_to_load = min(timesteps, key=lambda x: abs(x - args.load_step))

        model_path = os.path.join(args.checkpoint_path, str(timestep_to_load))

        logger.console_logger.info("Loading model from {}".format(model_path))
        learner.load_models(model_path)
        runner.t_env = timestep_to_load

        if args.evaluate or args.save_replay:
            evaluate_sequential(args, runner, logger)
            return

    assert not args.evaluate

    #this saves the path of the saved model to a csv file to reuse later
    #only if a csv file path is specifies
    if args.model_paths:
        assert args.checkpoint_path == ""
        save_path = os.path.join(args.local_results_path, "models", args.unique_token)
        if os.path.exists(args.model_paths):
            models = pd.read_csv(args.model_paths, index_col=0)
            models = models.append({'model_path': save_path, 'seed': int(args.seed)}, ignore_index=True)
            with open(args.model_paths, 'a') as f:
                models.iloc[-1:].to_csv(f, header=False)
        else:
            models = pd.DataFrame(columns=['model_path', 'seed'])
            models = models.append({'model_path': save_path, 'seed': int(args.seed)}, ignore_index=True)
            with open(args.model_paths, 'a') as f:
                models.to_csv(f, header=True)


    # start training
    episode = 0
    last_test_T = -args.test_interval - 1
    last_log_T = 0
    model_save_time = 0

    start_time = time.time()
    last_time = start_time

    logger.console_logger.info("Beginning training for {} timesteps".format(args.t_max))

    while runner.t_env <= args.t_max:

        # Run for a whole episode at a time
        episode_batch = runner.run(test_mode=False)
        buffer.insert_episode_batch(episode_batch)

        if buffer.can_sample(args.batch_size):
            episode_sample = buffer.sample(args.batch_size)

            # Truncate batch to only filled timesteps
            max_ep_t = episode_sample.max_t_filled()
            episode_sample = episode_sample[:, :max_ep_t]

            if episode_sample.device != args.device:
                episode_sample.to(args.device)

            learner.train(episode_sample, runner.t_env, episode)

        # Execute test runs once in a while
        n_test_runs = max(1, args.test_nepisode // runner.batch_size)
        if (runner.t_env - last_test_T) / args.test_interval >= 1.0:

            logger.console_logger.info("t_env: {} / {}".format(runner.t_env, args.t_max))
            logger.console_logger.info("Estimated time left: {}. Time passed: {}".format(
                time_left(last_time, last_test_T, runner.t_env, args.t_max), time_str(time.time() - start_time)))
            last_time = time.time()

            last_test_T = runner.t_env
            for _ in range(n_test_runs):
                runner.run(test_mode=True)

        if args.save_model and (runner.t_env - model_save_time >= args.save_model_interval or model_save_time == 0):
            model_save_time = runner.t_env
            save_path = os.path.join(args.local_results_path, "models", args.unique_token, str(runner.t_env))
            os.makedirs(save_path, exist_ok=True)
            logger.console_logger.info("Saving models to {}".format(save_path))

            # learner should handle saving/loading -- delegate actor save/load to mac,
            # use appropriate filenames to do critics, optimizer states
            learner.save_models(save_path)

        episode += args.batch_size_run

        if (runner.t_env - last_log_T) >= args.log_interval:
            logger.log_stat("episode", episode, runner.t_env)
            logger.print_recent_stats()
            last_log_T = runner.t_env

    runner.close_env()
    logger.console_logger.info("Finished Training")


def args_sanity_check(config, _log):

    # set CUDA flags
    # config["use_cuda"] = True # Use cuda whenever possible!
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning("CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!")

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (config["test_nepisode"]//config["batch_size_run"]) * config["batch_size_run"]

    return config


def run_sequential_cross(args, logger):
    #this is like run_sequential, but for doing cross-play among a list of models

    # Init runner so we can get env info
    runner = r_REGISTRY['cross'](args=args, logger=logger)

    # Set up schemes and groups here
    env_info = runner.get_env_info()
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]
    args.n_obs = env_info['n_obs']

    # Default/Base scheme
    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents", "dtype": th.float32},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }
    groups = {
        "agents": args.n_agents
    }
    preprocess = {
        "actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])
    }

    assert args.model_paths #need models to load

    args.models = pd.read_csv(args.model_paths, index_col=0)

    logger.sacred_info['model_paths'] = args.models.to_dict(orient='list')

    #this does cross-play evaluation to create a matrix of n times n values
    #it can create several such matrices, splitting the list of models into
    #args.models.shape[0]//n different "runs"

    n = args.n_seeds_per_run
    if n == 0:
        n = args.models.shape[0]
    for i in range(args.models.shape[0]//n):
        logger.console_logger.info("------ Evaluating run {} ------".format(i))
        for index_1 in range(n):
            logger.console_logger.info("------ run {}, seed nr {}------".format(i, index_1))
            for index_2 in range(n):
                self_play = index_1 == index_2

                #this can still support only doing self-play among the models, for instance, when calculating hash functions
                if args.only_self_play:
                    if not self_play:
                        continue

                checkpoint_path_1 = args.models.iloc[n*i+index_1]['model_path']
                checkpoint_path_2 = args.models.iloc[n*i+index_2]['model_path']

                #shouldn't change the args during run but this is the easiest
                #way to give this config to the environment later
                args.checkpoint_path_1 = checkpoint_path_1

                args.checkpoint_path_2 = checkpoint_path_2

                seed_1 = args.models.iloc[n*i+index_1]['seed']
                seed_2 = args.models.iloc[n*i+index_2]['seed']

                logger.console_logger.info("Model path 1: {}, Seed 1: {}".format(checkpoint_path_1,seed_1))
                logger.console_logger.info("Model path 2: {}, Seed 2: {}".format(checkpoint_path_2,seed_2))

                buffer = ReplayBuffer(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                                      preprocess=preprocess,
                                      device="cpu" if args.buffer_cpu_only else args.device)

                # Setup multiagent controller here
                mac1 = mac_REGISTRY[args.mac](buffer.scheme, groups, args)
                mac2 = mac_REGISTRY[args.mac](buffer.scheme, groups, args)

                # Give runner the scheme
                runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac1=mac1, mac2=mac2)

                # Learner
                learner1 = le_REGISTRY[args.learner](mac1, buffer.scheme, logger, args)
                learner2 = le_REGISTRY[args.learner](mac2, buffer.scheme, logger, args)

                if args.use_cuda:
                    learner1.cuda()
                    learner2.cuda()

                timesteps_1 = []
                timesteps_2 = []

                timestep_to_load = 0

                if not os.path.isdir(checkpoint_path_1):
                    logger.console_logger.info("Checkpoint directiory {} doesn't exist".format(checkpoint_path_1))
                    return

                if not os.path.isdir(checkpoint_path_2):
                    logger.console_logger.info("Checkpoint directiory {} doesn't exist".format(checkpoint_path_2))
                    return

                # Go through all files in args.checkpoint_path
                for name in os.listdir(checkpoint_path_1):
                    full_name = os.path.join(checkpoint_path_1, name)
                    # Check if they are dirs the names of which are numbers
                    if os.path.isdir(full_name) and name.isdigit():
                        timesteps_1.append(int(name))

                for name in os.listdir(checkpoint_path_2):
                    full_name = os.path.join(checkpoint_path_2, name)
                    # Check if they are dirs the names of which are numbers
                    if os.path.isdir(full_name) and name.isdigit():
                        timesteps_2.append(int(name))

                if args.load_step == 0:
                    # choose the max timesteps
                    timestep_to_load_1 = max(timesteps_1)
                    timestep_to_load_2 = max(timesteps_2)
                else:
                    #not implemented yet!
                    raise NotImplementedError
                    # choose the timestep closest to load_step
                    timestep_to_load = min(timesteps, key=lambda x: abs(x - args.load_step))

                model_path_1 = os.path.join(checkpoint_path_1, str(timestep_to_load_1))
                model_path_2 = os.path.join(checkpoint_path_2, str(timestep_to_load_2))

                logger.console_logger.info("Loading model 1 from {}".format(model_path_1))
                logger.console_logger.info("Loading model 2 from {}".format(model_path_2))

                learner1.load_models(model_path_1)
                learner2.load_models(model_path_2)

                runner.t_env = timestep_to_load

                #training doesn't make sense in cross-play, only evaluation
                assert args.evaluate

                evaluate_sequential(args, runner, logger, self_play=self_play)

    runner.close_env
    logger.console_logger.info("Finished cross-play evaluation")
