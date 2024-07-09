# λ-Discrepancy
Code for the paper **Mitigating Partial Observability in Sequential Decision Processes via the Lambda Discrepancy**.

For set up, make a virtual environment with
Python version >=3.11, and
```shell
pip install -r requirements.txt
```

To run the analytical experiments, the entry point
is `batch_run_analytical.py`.

The RNN experiments have the entry point
`batch_run_ppo.py`.

Instructions to run hyperparameter sweeps are
in `scripts/experiments.md`.

We also include the data used to generate our results in
`results/analytical_results.csv` and 
`results/rnn_results.csv.zip`, which you can unzip for the CSV.
