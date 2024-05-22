import importlib
import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem

from definitions import ROOT_DIR

emerald = "#2ecc71"
turquoise = "#1abc9c"
peter_river = "#3498db"
sunflower = "#f1c40f"
alizarin = "#e74c3c"
pumpkin = "#d35400"
green_sea = "#16a085"
wisteria = "#8e44ad"
midnight_blue = "#2c3e50"
all_colors = [peter_river, pumpkin, emerald, alizarin, wisteria, sunflower]

env_name_to_title = {
    'rocksample_15_15': 'RockSample (15, 15)',
    'rocksample_11_11': 'RockSample (11, 11)',
    'pocman': 'Pocman',
    'battleship_10': 'Battleship 10x10',
    'battleship_5': 'Battleship 5x5',
}

env_name_to_x_upper_lim = {
    '4x3': 1e6,
    'cheese.95': 1e6,
    'hallway': 2e6,
    'network': 1e6,
    'paint': 1e6,
    'tiger-alt-start': 1e6,
    'tmaze_5': 2e6
}


def plot_reses(all_reses: list[tuple], n_rows: int = 2,
               individual_runs: bool = False):
    plt.rcParams.update({'font.size': 32})

    # check to see that all our envs are the same across all reses.
    all_envs = [set(x['envs']) for _, x in all_reses]
    for envs in all_envs:
        assert envs == all_envs[0]

    envs = list(sorted(all_envs[0]))

    n_rows = min(n_rows, len(envs))
    n_cols = max((len(envs) + 1) // n_rows, 1) if len(envs) > 1 else 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(30, 20))

    for k, (study_name, res) in enumerate(all_reses):
        scores = res['scores']
        if isinstance(scores, list):
            mean_over_steps = [score.mean(axis=1)[..., 0] for score in scores]
            mean = [m.mean(axis=-1) for m in mean_over_steps]
            std_err = [sem(m, axis=-1) for m in mean_over_steps]
        else:
            # take mean over both
            mean_over_steps = scores.mean(axis=1)
            mean = mean_over_steps.mean(axis=-2)
            std_err = sem(mean_over_steps, axis=-2)

        for i, env in enumerate(envs):
            row = i // n_cols
            col = i % n_cols

            env_idx = res['envs'].index(env)
            if isinstance(mean, list):
                env_mean, env_std_err = mean[env_idx], std_err[env_idx]
                n_seeds = mean_over_steps[0].shape[-1]
            else:
                env_mean, env_std_err = mean[..., env_idx], std_err[..., env_idx]
                n_seeds = mean_over_steps.shape[-2]

            # env_mean, env_std_err = env_mean[10:], env_std_err[10:]
            x_axis_multiplier = res['step_multiplier'][env_idx]

            if len(envs) == 1:
                ax = axes
            else:
                ax = axes[row, col] if n_cols > 1 else axes[i]
            x = np.arange(env_mean.shape[0]) * x_axis_multiplier
            x_upper_lim = env_name_to_x_upper_lim.get(env, None)

            ax.plot(x, env_mean, label=study_name, color=all_colors[k])
            if individual_runs:
                # -2 index is seeds.
                for j in range(mean_over_steps.shape[-2]):
                    alpha = 1 / mean_over_steps.shape[-2]
                    m = mean_over_steps[..., j, env_idx] if isinstance(mean_over_steps, np.ndarray) else mean_over_steps[env_idx][..., j]
                    ax.plot(x, m, color=all_colors[k], alpha=alpha)
            else:
                ax.fill_between(x, env_mean - env_std_err, env_mean + env_std_err,
                                color=all_colors[k], alpha=0.35)
            ax.set_title(env_name_to_title.get(env, env))
            if x_upper_lim is not None:
                ax.set_xlim(right=x_upper_lim)
            # ax.margins(x=0.015)
            ax.locator_params(axis='x', nbins=3, min_n_ticks=3)
            ax.spines[['right', 'top']].set_visible(False)

    # Customize legend to use square markers
    legend = plt.legend(loc='lower right')

    # Change line in legend to square
    for line in legend.get_lines():
        line.set_marker('s')
        line.set_markerfacecolor(line.get_color())
        line.set_linestyle('')
        line.set_markersize(20)  # Increase the marker size

    fig.supxlabel('Environment steps')
    fig.supylabel(f'Online discounted returns ({n_seeds} runs)')
    #
    fig.tight_layout()

    plt.show()
    return fig, axes


def get_total_steps_multiplier(saved_steps: int, hparam_path: Path):
    spec = importlib.util.spec_from_file_location('temp', hparam_path)
    var_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(var_module)
    all_list_hparams = getattr(var_module, 'hparams')['args']

    steps_multipliers = []
    for hparams in all_list_hparams:
        assert 'total_steps' in hparams
        total_steps = hparams['total_steps']
        steps_multiplier = total_steps // saved_steps
        steps_multipliers.append(steps_multiplier)

    assert all(m == steps_multipliers[0] for m in steps_multipliers)
    return steps_multipliers[0]


if __name__ == "__main__":
    env_name = 'pomdps'

    # normal
    # study_paths = [
    #     ('λ-discrepancy + PPO', Path(ROOT_DIR, 'results', f'{env_name}_LD_ppo')),
    #     ('PPO', Path(ROOT_DIR, 'results', f'{env_name}_ppo')),
    # ]

    # fixedlambda
    # study_paths = [
    #     ('λ-discrepancy + PPO', Path(ROOT_DIR, 'results', f'{env_name}_fixedlambda_LD_ppo')),
    #     ('PPO', Path(ROOT_DIR, 'results', f'{env_name}_fixedlambda_ppo')),
    # ]

    # best
    study_paths = [
        ('λ-discrepancy + PPO', Path(ROOT_DIR, 'results', f'{env_name}_LD_ppo_best')),
        ('PPO', Path(ROOT_DIR, 'results', f'{env_name}_ppo_best')),
    ]

    hyperparam_type = 'per_env'  # (all_env | per_env)
    plot_name = f'{env_name}_{hyperparam_type}'

    if study_paths[0][1].stem.endswith('best'):
        plot_name += '_best'

    envs = None

    all_reses = []

    for name, study_path in study_paths:
        if hyperparam_type == 'all_env':
            fname = "best_hyperparam_res.pkl"
            if name == 'PPO Markov':
                fname = 'best_hyperparam_res_F_split.pkl'
        elif hyperparam_type == 'per_env':
            fname = "best_hyperparam_per_env_res.pkl"

        with open(study_path / fname, "rb") as f:
            best_res = pickle.load(f)

        all_reses.append((name, best_res))
        hyperparam_path = study_path.parent.parent / 'scripts' / 'hyperparams' / f'{study_path.stem}.py'
        step_multiplier = get_total_steps_multiplier(best_res['scores'].shape[0], hyperparam_path)
        best_res['step_multiplier'] = [step_multiplier] * len(best_res['envs'])

    fig, axes = plot_reses(all_reses, individual_runs=False, n_rows=3)

    save_plot_to = Path(ROOT_DIR, 'results', f'{plot_name}.pdf')

    fig.savefig(save_plot_to, bbox_inches='tight')
    print(f"Saved figure to {save_plot_to}")

