from pathlib import Path

exp_name = Path(__file__).stem

lrs = [2.5e-5]
lambda0s = [0.1]
lambda1s = [0.95]
alphas = [1]
ld_weights = [0]

hparams = {
    'file_name':
        f'runs_{exp_name}.txt',
    'entry': 'batch_run_ppo.py',
    'args': [
        {
            'env': 'battleship_10',
            'double_critic': False,
            'action_concat': True,
            'memoryless': True,
            'lr': lrs,
            'lambda0': ' '.join(map(str, lambda0s)),
            'lambda1': ' '.join(map(str, lambda1s)),
            'alpha': ' '.join(map(str, alphas)),
            'ld_weight': ' '.join(map(str, ld_weights)),
            'hidden_size': 512,
            'entropy_coeff': 0.05,
            'steps_log_freq': 8,
            'update_log_freq': 10,
            'total_steps': int(1e7),
            'seed': 2029,
            'n_seeds': 30,
            'platform': 'gpu',
            'study_name': exp_name
        }
    ]
}
