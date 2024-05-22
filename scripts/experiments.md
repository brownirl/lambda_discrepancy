# Running experiments w/ batch_run_ppo.py

To run experiments defined in the `hyperparams` directory,
you need to write jobs according to the `write_jobs.py` script 
in this directory. For example, in the `scripts` directory:
```shell
python write_jobs.py hyperparams/pocman_ppo.py
```
This will create a `runs` file with `runs_pocman_ppo.txt`, where
each line is an experiment run.

The alternative is to use the `onager` library, and instead:
```shell
python onager_write_jobs.py hyperparams/pocman_ppo.py
```
which would allow you to run with any slurm-based cluster.

To run jobs, you can either launch them with `onager`
(where you need to `launch` after prelaunches are done
with `onager_write_jobs.py`), or use
the script in `launching/run_locally.sh` script, which
you'll have to modify with the runs file you just made.

After finishing a job, the results should be in the `results`
folder in the root of the project directory. So for the above
example, we'd see a new folder in the root directory that's
named `results/pocman_ppo`.

## PPO jobs
The PPO sweeps listed in all hyperparameter files in
`scripts/hyperparams` that end with `*_ppo.py`.

In the same file, the best hyperparameter runs for
30 seeds for each environment all are listed in files
that end with `*_ppo_best.py`.

For the PPO jobs, you can get the best hyperparameters 
of a sweep by first going back into `scripts` and running
```shell
python parse_batch_experiments.py ../results/pocman_ppo
python best_hyperparams_per_env.py ../results/pocman_ppo/parsed_hparam_scores.pkl
```
which will create a `pkl` file `best_hyperparam_per_env_res.pkl`. 
You can visualize this file with the file `plot_best_hyperparams.py`.

## Analytical Jobs
The analytical experiments were done with the two files
in the directory `scripts/hyperparams/`
with names `analytical_30seeds.py` and `analytical_8_30seeds.py`.

You can visualize these results with `scripts/plot_analytical.py`.

