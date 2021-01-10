# A new Formalism, Method and Open Issues for Zero-Shot Coordination code


This directory contains code corresponding to the Bachelor thesis "A new Formalism, Method and Open Issues for Zero-Shot Coordination" by Johannes Treutlein

It is based on a beta-release of the PyMARL framework. For more specific information on Pymarl, see the file `PYMARL.md` and https://github.com/oxwhirl/pymarl


## Contributions

Here, we give an overview over the files created and edited by us.

The following files were edited by us:

To implement vanilla policy gradient (modification of the `central-V` algorithm)
```
src/config/algs/policy_gradient.yaml
```

Added functionality for evaluation of a list of models in cross play
```
src/run.py
src/runners/episode_cross_runner.py
```

The following files were created by us:

For computing a hash function
```
src/utils/hash_function.py
```

The following implements the two levergame environments. It also implements other-play for that environment, and preparation of histories for hashing, and 
a method for printing histories from an evaluation run.

```
src/envs/levergame.py
src/config/envs/asymmetriclevergame.yaml
src/config/envs/twostagelevergame.yaml
```

This runs other-play with tie-breaking for different hyperparameters, and runs an evaluation on all the runs.
```
src/op_with_tie_breaking.py
```

These are all methods for evaluating and plotting
```
src/analysis_of_equiv_classes.py
src/plot_op_w_tie_breaking.py
src/plot_cross_play_heatmap.py
src/plot_training_curve.py
```

## Running experiments

Here, we describe how to run the experiments we did for the thesis. 
The files under "src/config" contain default configuration values. Models can be trained by running, for instance,

```
python3 src/main.py --env-config=twostagelevergame --config=policy_gradient
```

This trains a model using policy gradient on the two-stage lever game.
For asymmetric lever game, the environment is `asymmetriclevergame`.
The model is saved under the directory `src/results/models`, and logs and data of the experiment is created using the
`sacred` module, as a new directory with a specific ID in the directory `results/sacred`. Data from training are also
logged by tensorboard and added to `results/tb_logs`.

To train several models and keep track of them, we use the shell-script `./train_models.sh`:

```
for ((seed = 100; seed < 420; seed+=1))
do
  python3 src/main.py --env-config=twostagelevergame --config=policy_gradient \
  with seed=$seed checkpoint_path="" model_paths='results/model_lists/twostagelevergame.csv'
done
```

For instance, this trains models for  `twostagelevergame` for seeds 100 to 419 and add the models to the file
`results/model_lists/twostagelevergame.csv`.

A list with model paths and seeds of models that we did train is in the files

```
results/model_lists/twostagelevergame.csv
results/model_lists/asymmetriclevergame.csv
```

Given this list of trained models, one can plot learning curves that average over all the models by executing

```
python3 src/plot_learning_curve.py --model_paths=results/model_lists/twostagelevergame.csv
python3 src/plot_learning_curve.py --model_paths=results/model_lists/asymmetriclevergame.csv
```

This saves a pdf file of a plot in the directory `results/learning_curves/`

For a given model, we can print episodes to inspect what the model is doing. For instance, this is done by running

```
python3 src/main.py --env-config=twostagelevergame --config=policy_gradient \
with seed=100 \
evaluate=True \
save_episodes=True \
checkpoint_path=results/models/policy_gradient__2020-11-24_12-58-20 
```

This adds a .txt file in the directory `results/episodes/policy_gradient__2020-11-24_12-58-20/` with printouts of episodes

Next, we turn to calculating hash-values for all models. We calculate hash values simultaneously using 20 different seeds
for the hash-function. That is done by running the command.

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
tie-breaking function will be an average of those. `hash_function_seeds` specifies the list of seeds for the hash-function.
`seed` specifies randomness of the environment and agent policies.

The following programs are then based on the data generated from this run, using the sacred experiment ID of the run.
For our experiments, the IDs how the hash-function run for both environments are `1672` and `1673`.

After having calculated hash-values, other-play with tie-breaking can be run and evaluated. To that end, we run

```
python3 src/op_with_tie_breaking.py --hash_run="1672" --n_seeds_per_run="32" --seed="100" --test_nepisode="2048" 
python3 src/op_with_tie_breaking.py --hash_run="1673" --n_seeds_per_run="32" --seed="100" --test_nepisode="2048" 
```

This indicates that we use the hash values and list of models used from the sacred ID `1672`, and split the runs into 10
groups of 32 each for the evaluation. The specified `seed` and `test_nepisode` are relevant to the evaluation of the
tie-breaking method, they specify the environment/policy stochasticity and number of test episodes per model for the
evaluation run.

Now assume we have run the above script for both environments, and the sacred IDs associated to the cross-play evaluations
are `1674` respectively `1675`. We can now plot heatmaps and graphs to illustrate the results from these. For instance,
for the first environment, we run

```
python3 src/plot_cross_play_heatmap.py --run='1674' >> results/op_with_tie_breaking/evalrun_1674_plot_cross_play_heatmap.log
python3 src/plot_op_w_tie_breaking.py --run='1674' >> results/op_with_tie_breaking/evalrun_1674_plot_graph.log
```

This creates pdf files with plots in the directory `results/op_with_tie_breaking/`

                   
Finally we can analyze the policies in terms of their equivalence classes, such as how frequent each is, and plot a
histogram of hash values in terms of the equivalence classes. To that end, we need to first perform cross-play across
all the different policies (i.e., calculate 102400 cross-play values). To do so efficiently, we only test each
combination with 256 episodes.

```
python3 src/main.py --env-config=twostagelevergame --config=policy_gradient \
with seed=100 \
evaluate=True \
cross_play=True \
test_nepisode=256 \
model_paths=results/model_lists/twostagelevergame.csv
```

The data is saved under sacred IDs `??` and `??`. Using these, we implemented a method that can build up equivalence classes
of policies, based on the cross-play data. These classes are then analyzed in terms of their size, and a histogram with
hash values for each class is created

```
python3 src/analysis_of_equiv_classes.py --run='1640' --hash_run='1672' --index='0' >> results/analysis_of_equiv_classes/run1640_hashrun1672_analysis.log
```

This prints a lot of data on the equivalence classes and saves a histogram to the directory
`results/analysis_of_equiv_classes/`. The `index` is the index of the seed for the hash function that is used for the
histogram.