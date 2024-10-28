from pathlib import Path

exp_name = Path(__file__).stem

lrs = [2.5e-5]
lambda0s = [0.5]
lambda1s = [0.95]
alphas = [1]
ld_weights = [0]

hparams = {
    'file_name':
        f'runs_{exp_name}.txt',
    'entry': '-m scripts.batch_run_ppo_epoch',
    'args': [
        {
            'env': 'pocman',
            'double_critic': False,
            'action_concat': True,
            'lr': lrs,
            'lambda0': ' '.join(map(str, lambda0s)),
            'lambda1': ' '.join(map(str, lambda1s)),
            'alpha': ' '.join(map(str, alphas)),
            'ld_weight': ' '.join(map(str, ld_weights)),
            'hidden_size': 512,
            'entropy_coeff': 0.05,
            'num_epochs': 25,
            'steps_log_freq': 4,
            'update_log_freq': 200,
            'total_steps': int(1e7),
            'save_checkpoints': True,
            'save_runner_state': True,
            'seed': 2036,
            'n_seeds': 5,
            'platform': 'gpu',
            'study_name': exp_name
        }
    ]
}
