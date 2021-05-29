# Code for "A New Formalism, Method and Open Issues for Zero-Shot Coordination"

This github repository contains the code for the ICML 2021 paper "A New Formalism, Method and Open Issues for Zero-Shot Coordination".

It is based on a beta-release of the PyMARL framework. For more specific information on Pymarl, see https://github.com/oxwhirl/pymarl.


## Overview

Here, we give an overview over the parts of the PyMARL framework created and edited by us.

To implement vanilla policy gradient (modification of the `central-V` algorithm).
```
src/config/algs/policy_gradient.yaml
src/learners/policy_gradient_learner.py
```

Added functionality for evaluation of a list of models in cross-play.
```
src/run.py
src/runners/episode_cross_runner.py
```

For computing a hash function.
```
src/utils/hash_function.py
```

The following implements the two levergame environments. It also implements other-play for that environment, preparation
of histories for hashing, and 
a method for printing histories from an evaluation run.

```
src/envs/levergame.py
src/config/envs/asymmetriclevergame.yaml
src/config/envs/twostagelevergame.yaml
```

This runs other-play with tie-breaking, and starts a cross-play evaluation of the method.
```
src/op_with_tie_breaking.py
```

These are all methods for evaluation and plotting.
```
src/analysis_of_equiv_classes.py
src/plot_op_w_tie_breaking.py
src/plot_cross_play_heatmap.py
src/plot_training_curve.py
```

## Results folder

Results are in the folder `src/results/`. Due to the space restriction, we did not
include the trained RNN models themselves.

## Running experiments

Here, we describe how to run our experiments. The files under "src/config" contain default configuration values.
Models can be trained by running, for instance,

```
python3 src/main.py --env-config=twostagelevergame --config=policy_gradient
```

This trains a model using policy gradient on the two-stage lever game.
For asymmetric lever game, the environment is `asymmetriclevergame`.
The model is saved under the directory `src/results/models`, and logs and data of the experiment is created using the
`sacred` module, as a new directory with a specific ID in the directory `results/sacred`. Data from training is also
logged by tensorboard and added to `results/tb_logs`.

To train several models and keep track of them, we use the shell-script `./train_models.sh`:

```
for ((seed = 100; seed < 420; seed+=1))
do
  python3 src/main.py --env-config=twostagelevergame --config=policy_gradient \
  with seed=$seed checkpoint_path="" model_paths='results/model_lists/twostagelevergame.csv'
done
```

For instance, this trains models for  `twostagelevergame` for seeds 100 to 419 and adds paths to the models to the file
`results/model_lists/twostagelevergame.csv`.

A list with model paths and seeds of models that we did train is in the files `results/model_lists/twostagelevergame.csv`
and `results/model_lists/asymmetriclevergame.csv`.

Given this list of trained models and associated tensorboard logs, one can plot learning curves that average over all the models by executing

```
python3 src/plot_learning_curve.py --model_paths=results/model_lists/twostagelevergame.csv >> results/learning_curves/twostagelevergame.log
python3 src/plot_learning_curve.py --model_paths=results/model_lists/asymmetriclevergame.csv >> results/learning_curves/asymmetriclevergame.log
```

This saves a pdf file of a plot in the directory `results/learning_curves`.

For a given model, we can print episodes to inspect what the model is doing. For instance, this is done by running

```
python3 src/main.py --env-config=twostagelevergame --config=policy_gradient \
with seed=100 \
evaluate=True \
save_episodes=True \
checkpoint_path=results/models/policy_gradient__2020-11-24_12-58-20 
```

This adds a .txt file in the directory `results/episodes/policy_gradient__2020-11-24_12-58-20/` with printouts of episodes.

Next, we turn to calculating tie-breaking values for all policies. We calculate these values simultaneously using 20 different seeds
for the hash-function. For the two-stage lever game, this is done by running the command

```
python3 src/main.py --env-config=twostagelevergame --config=policy_gradient \
with seed=100 \
evaluate=True \
cross_play=True \
only_self_play=True \
calculate_hash=True \
test_nepisode=2048 \
hash_type=nn \
hash_function_seeds="[100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119]" \
model_paths=results/model_lists/twostagelevergame.csv
```

Setting `test_nepisode=2048` means that the hash-function will be calculated for 2048 episodes and the
tie-breaking function will be an average of those. `hash_function_seeds` specifies a list of seeds used for the
hash-function. `seed` specifies randomness of the environment and of agent policies.

The following python scripts are based on the data generated from this run, using the sacred experiment ID of the run.
For our experiments in this folder, the IDs for the hash-function run for both environments are `2` and `3`.

After having calculated hash-values, other-play with tie-breaking can be run and evaluated in cross-play, using the
following python script, which applies OP with tie-breaking and starts a cross-play evaluation.

```
python3 src/op_with_tie_breaking.py --hash_run="2" --n_seeds_per_run="32" --seed="100" --test_nepisode="2048" \
>> results/op_with_tie_breaking/op_with_tie_breaking_hash_run_2.log
```

This indicates that we use the hash values and list of models used from the sacred ID `2`, and split the runs into 10
groups of 32 each. The specified `seed` and `test_nepisode` are relevant to the evaluation of the
tie-breaking method: they specify the seed for environment/policy stochasticity and number of test episodes per model for the
evaluation run.

Now assume we have run the above script for both environments. In our case, the sacred-IDs
associated to the cross-play evaluations
are `4` respectively `5` for the two environments. We can now plot heatmaps and graphs to
illustrate the results from these. We run

```
python3 src/plot_cross_play_heatmap.py --run='4' >> results/op_with_tie_breaking/heatmap_run_4.log
python3 src/plot_op_w_tie_breaking.py --run='4' >> results/op_with_tie_breaking/plot_run_4.log

python3 src/plot_cross_play_heatmap.py --run='5' >> results/op_with_tie_breaking/heatmap_run_5.log
python3 src/plot_op_w_tie_breaking.py --run='5' >> results/op_with_tie_breaking/plot_run_5.log
```

This creates pdf files with plots in the directory `results/op_with_tie_breaking`. For the heatmaps, one seed for the 
hash-function is used, and we print cross-play heatmaps for the case where no ties are broken and for the case where
ties are broken between all 32 seeds per each of the 10 different sets of policies. In the other plot, we plot
performance of OP with tie-breaking for different numbers of seeds (1,2,4,8,16,32) and plot standard deviations over
the 20 seeds for the hash function.

Finally, we can analyze the policies in terms of classes of mutually compatible strategies, plot a histogram of hash
values for these classes, and compare their relative sizes. To that end, we need to first
perform cross-play across all the different policies (i.e., calculate 320^2=102400 cross-play values).
To do so efficiently, we only test each combination with 256 episodes.

```
python3 src/main.py --env-config=twostagelevergame --config=policy_gradient \
with seed=100 \
evaluate=True \
cross_play=True \
test_nepisode=256 \
model_paths=results/model_lists/twostagelevergame.csv
```

The data is saved under sacred IDs `6` and `7`. Using these, we implemented a method that can build classes of mutually
compatible policies, based on the cross-play data:

```
python3 src/analysis_of_equiv_classes.py --run='6' --hash_run='2' --index='0' >> results/analysis_of_equiv_classes/cross_play_run_6_hash_run_2.log
python3 src/analysis_of_equiv_classes.py --run='7' --hash_run='3' --index='0' >> results/analysis_of_equiv_classes/cross_play_run_7_hash_run_3.log
```

This prints a lot of data on the equivalence classes and saves a histogram to the directory
`results/analysis_of_equiv_classes/`. Here, `--index=0` specifies the index of the seed for the hash function that is used for the
histogram.


## License

Code licensed under the Apache License v2.0
