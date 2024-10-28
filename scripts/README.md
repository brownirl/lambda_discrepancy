# Scripts

This `scripts` directory includes scripts for plotting all experimental
results in our work, as well as scripts for a few additional experiments
in the paper.

## P.O. PacMan Memory Probe
To train our memory probe, we need to first collect checkpoints from 
a P.O. PacMan run. We can do so with the `pocman_*ppo_best_ckpt.py` scripts
in `scripts/hyperparams`. This script will train P.O. PacMan agents with the
best swept hyperparams.

After training, we need to run the `scripts/collect_rnn_trajectories.py` script
to collect 1M samples from each behavior policy (LD and vanilla PPO). This script
will collect RNN hidden states from two RNNs (`--rnn_path_0` and `--rnn_path_1`),
while following the `--behavior_path` RNN as the behavior policy. We collect 
1M time steps with each variant as the behavior policy, for a combined dataset of
2M samples. We use the `scripts/combine_probe_datasets.py` script to combine these
datasets, resulting in a `results/combined_probe_datasets` data buffer.

Now we train our probe with the `scripts/train_probe.py` script. Pass in the 
PATH to the combined dataset above as the argument to `--dataset_path`. Use the 
`--features_idx` argument to select which RNN hidden states to use for training (0 or 1).
The index and ordering of these hidden states will depend on which RNN paths were
used in `--rnn_path_0` and `--rnn_path_1`. 

Once our probe has been trained, we can collect trajectories with each trained probe
with `scripts/collect_probe_trajectories.py`. We can visualize these collected 
probe trajectories with `scripts/visualization/viz_pocman_probe.py`.
We provide the collected probe trajectories in `results/pocman_pellet_probe_trajectory.zip`.
To generate this visualization, simply uncompress this file and pass each file in as
the argument to `scripts/visualization/viz_pocman_probe.py`.




