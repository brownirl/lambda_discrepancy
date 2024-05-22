from pathlib import Path

exp_name = Path(__file__).stem

hparams = {
    'file_name':
        f'runs_{exp_name}.txt',
    'entry': '-m batch_run_kitchen_sinks_ld_only',
    'args': [{
        'spec': [
            'tiger-alt-start', 'tmaze_5_two_thirds_up', '4x3.95',
            'cheese.95', 'network', 'shuttle.95', 'paint.95'
            # 'hallway'
            # 'bridge-repair',
        ],
        'policy_optim_alg': 'policy_grad',
        'leave_out_optimal': True,
        'value_type': 'q',
        'error_type': 'l2',
        'alpha': 1.,
        'mi_steps': 10000,
        'pi_steps': 10000,
        'optimizer': 'adam',
        'lr': 0.01,
        'n_mem_states': [2, 4],
        'mi_iterations': 1,
        'random_policies': 100,
        'n_seeds': 30,
        'platform': 'gpu'
    },

    ]
}
