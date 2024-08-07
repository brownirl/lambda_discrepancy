from pathlib import Path

exp_name = Path(__file__).stem

lrs = [2.5e-4]
lambda0s = [0.1]
lambda1s = [0.95]
alphas = [1]
ld_weights = [0.5]

hparams = {
    'file_name':
        f'runs_{exp_name}.txt',
    'entry': 'batch_run_ppo.py',
    'args': [
        {
            'env': 'rocksample_11_11',
            'double_critic': True,
            'lr': ' '.join(map(str, lrs)),
            'lambda0': ' '.join(map(str, lambda0s)),
            'lambda1': lambda1s,
            'alpha': ' '.join(map(str, alphas)),
            'ld_weight': ld_weights,
            'entropy_coeff': 0.35,
            'steps_log_freq': 4,
            'update_log_freq': 5,
            'total_steps': int(3e6),
            'seed': 2029,
            'n_seeds': 30,
            'platform': 'gpu',
            'study_name': exp_name
        }
    ]
}
