#!/usr/bin/env bash

N_JOBS=1

cd ../../
source venv/bin/activate

#TO_RUN=$(sed -n "101,800p" scripts/runs/runs_rnn_reruns_sweep_mc.txt)

parallel --eta -u --jobs $N_JOBS < 'scripts/runs/runs_hopper_v_batch_ppo.txt'
#parallel --eta -u < "$TO_RUN"
