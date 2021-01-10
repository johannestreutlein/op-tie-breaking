for ((seed = 100; seed < 420; seed+=1))
do
  python3 src/main.py --env-config=twostagelevergame --config=policy_gradient \
  with seed=$seed checkpoint_path="" model_paths='results/model_lists/twostagelevergame.csv'
done