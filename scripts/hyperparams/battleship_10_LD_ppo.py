from pathlib import Path

exp_name = Path(__file__).stem

lrs = [2.5e-3, 2.5e-4, 2.5e-5, 2.5e-6]
lambda0s = [0.1, 0.5, 0.7, 0.9, 0.95]
lambda1s = [0.1, 0.5, 0.7, 0.9, 0.95]
alphas = [1]
ld_weights = [0, 0.125, 0.25, 0.5]

hparams = {
    'file_name':
        f'runs_{exp_name}.txt',
    'entry': 'batch_run_ppo.py',
    'args': [
        {
            'env': 'battleship_10',
            'double_critic': True,
            'action_concat': True,
            'lr': lrs,
            'lambda0': ' '.join(map(str, lambda0s)),
            'lambda1': lambda1s,
            'alpha': ' '.join(map(str, alphas)),
            'ld_weight': ld_weights,
            'hidden_size': 512,
            'entropy_coeff': 0.05,
            'steps_log_freq': 8,
            'update_log_freq': 10,
            'total_steps': int(1e7),
            'seed': 2024,
            'n_seeds': 5,
            'platform': 'gpu',
            'study_name': exp_name
        }
    ]
}
