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
    'args': [{
        'env': [
            '4x3', 'cheese.95', 'hallway', 'network', 'heavenhell', 'paint',
            'shuttle', 'tiger-alt-start', 'tmaze_5'
        ],
        'double_critic': True,
        'lr': ' '.join(map(str, lrs)),
        'lambda0': lambda0s,
        'lambda1': ' '.join(map(str, lambda1s)),
        'alpha': ' '.join(map(str, alphas)),
        'ld_weight': ' '.join(map(str, ld_weights)),
        'entropy_coeff': 0.05,
        'hidden_size': 32,
        'total_steps': int(3e6),
        'steps_log_freq': 4,
        'update_log_freq': 10,
        'seed': 2024,
        'n_seeds': 10,
        'platform': 'gpu',
        'study_name': exp_name
    }]
}
