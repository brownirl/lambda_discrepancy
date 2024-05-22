import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from jax import config
from pathlib import Path
from tqdm import tqdm

config.update('jax_platform_name', 'cpu')
np.set_printoptions(precision=4)
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams.update({'font.size': 18})

from lamb.envs.classic import load_pomdp
from lamb.utils.file_system import load_info
from definitions import ROOT_DIR

kitchen_obj_map = {
    'tde': 'mstde',
    'tde_residual': 'mstde_res',
    'discrep': 'ld'
}


def parse_batch_dirs(exp_dirs: list[Path],
                     baseline_dict: dict,
                     args_to_keep: list[str]):
    all_results = []

    def parse_exp_dir(exp_dir: Path):
        print(f"Parsing {exp_dir}")
        for results_path in tqdm(list(exp_dir.iterdir())):
            if results_path.is_dir() or results_path.suffix != '.npy':
                continue

            info = load_info(results_path)
            args = info['args']
            logs = info['logs']

            if args['spec'] not in baseline_dict:
                continue

            pomdp, _ = load_pomdp(args['spec'])
            # n_random_policies = args['random_policies']

            beginning = logs['beginning']
            aim_measures = beginning['measures']
            if 'kitchen' in exp_dir.stem:
                # if we're doing kitchen sinks policies, we need to take the mean over
                # initial policies
                init_policy_perf_seeds = (aim_measures['values']['state_vals']['v'] * aim_measures['values']['p0'])
                init_policy_perf_seeds = init_policy_perf_seeds.sum(axis=-1).mean(axis=-1)
            else:
                init_policy_perf_seeds = np.einsum('ij,ij->i',
                                                   aim_measures['values']['state_vals']['v'],
                                                   aim_measures['values']['p0'])

            after_pi_op = logs['after_pi_op']
            if 'measures' not in after_pi_op:
                assert 'initial_improvement_measures' in after_pi_op
                apo_measures = after_pi_op['initial_improvement_measures']
            else:
                apo_measures = after_pi_op['measures']
            init_improvement_perf_seeds = np.einsum('ij,ij->i',
                                                    apo_measures['values']['state_vals']['v'],
                                                    apo_measures['values']['p0'])
            compare_to_perf = baseline_dict[args['spec']]

            if isinstance(args['objectives'], str):
                keys = [args['objectives']]
            else:
                keys = args['objectives']

            for i, key in enumerate(keys):
                key = kitchen_obj_map.get(key, key)
                objective, residual = key, False
                if key == 'mstde_res':
                    objective, residual = 'mstde', True

                args['residual'] = residual
                args['objective'] = objective

                single_res = {k: args[k] for k in args_to_keep}
                single_res['experiment'] = exp_dir.name + f'_{objective}'
                single_res['objective'] = objective

                final = logs['final']
                final_measures = final[key]['measures']
                if 'kitchen' in exp_dir.stem:
                    # For now, we assume kitchen selection objective == mem learning objective
                    final_mem_perf = np.einsum('ij,ij->i',
                                               final_measures['values']['state_vals']['v'][:, i],
                                               final_measures['values']['p0'][:, i])

                else:
                    final_mem_perf = np.einsum('ij,ij->i',
                                               final_measures['values']['state_vals']['v'],
                                               final_measures['values']['p0'])

                for i in range(args['n_seeds']):
                    all_results.append({
                        **single_res,
                        'seed': i,
                        'init_policy_perf': init_policy_perf_seeds[i],
                        'init_improvement_perf': init_improvement_perf_seeds[i],
                        'final_mem_perf': final_mem_perf[i],
                        'compare_to_perf': compare_to_perf,
                    })

    for exp_dir in exp_dirs:
        parse_exp_dir(exp_dir)

    all_res_df = pd.DataFrame(all_results)

    return all_res_df


def parse_baselines(
        plot_order: list[str],
        vi_dir: Path = Path(ROOT_DIR, 'results', 'vi'),
        pomdp_files_dir: Path = Path(ROOT_DIR, 'results', 'belief'),
        compare_to: str = 'belief'):
    compare_to_dict = {}
    spec_to_belief_state = {'tmaze_5_two_thirds_up': 'tmaze5'}

    def load_state_val(spec: str):
        for vi_path in vi_dir.iterdir():
            if spec_to_belief_state.get(spec, spec) in vi_path.name:
                vi_info = load_info(vi_path)
                max_start_vals = vi_info['optimal_vs']
                return np.dot(max_start_vals, vi_info['p0'])

    for spec in plot_order:
        if compare_to == 'belief':

            for fname in pomdp_files_dir.iterdir():
                if 'pomdp-solver-results' in fname.stem:
                    if (fname.stem ==
                            f"{spec_to_belief_state.get(spec, spec)}-pomdp-solver-results"
                    ):
                        belief_info = load_info(fname)
                        compare_to_dict[spec] = belief_info['start_val']
                        break
            else:
                compare_to_dict[spec] = load_state_val(spec)

        elif compare_to == 'state':
            compare_to_dict[spec] = load_state_val(spec)

    return compare_to_dict


if __name__ == "__main__":
    experiment_dirs = [
        Path(ROOT_DIR, 'results', 'analytical_30seeds'),
    ]

    args_to_keep = ['spec', 'n_mem_states', 'seed', 'alpha', 'residual', 'objective']
    split_by = [arg for arg in args_to_keep if arg != 'seed'] + ['experiment']

    # this option allows us to compare to either the optimal belief state soln
    # or optimal state soln. ('belief' | 'state')
    compare_to = 'belief'

    policy_optim_alg = 'policy_grad'

    spec_plot_order = [
        'network', 'paint.95', '4x3.95', 'tiger-alt-start', 'shuttle.95', 'cheese.95', 'tmaze_5_two_thirds_up'
    ]

    plot_key = 'final_mem_perf'  # for single runs

    # %% codecell

    compare_to_dict = parse_baselines(spec_plot_order,
                                      compare_to=compare_to)

    all_res_df = parse_batch_dirs(experiment_dirs,
                                  compare_to_dict,
                                  args_to_keep)

    # %% codecell

    # FILTER OUT for what we want to plot
    # alpha = 1.
    #
    # all_res_df
    residual = False
    alpha = 1.
    filtered_df = all_res_df[
        (all_res_df['experiment'] == f'{experiment_dirs[0].stem}_ld')
    ].reset_index()

    # %% codecell
    all_res_groups = filtered_df.groupby(split_by, as_index=False)
    all_res_means = all_res_groups.mean()
    del all_res_means['seed']

    # %% codecell
    cols_to_normalize = ['init_improvement_perf', plot_key]
    merged_df = filtered_df

    # for col_name in cols_to_normalize:

    normalized_df = merged_df.copy()
    normalized_df['init_improvement_perf'] = (normalized_df['init_improvement_perf'] - merged_df['init_policy_perf']) / (merged_df['compare_to_perf'] - merged_df['init_policy_perf'])
    normalized_df[plot_key] = (normalized_df[plot_key] - merged_df['init_policy_perf']) / (merged_df['compare_to_perf'] - merged_df['init_policy_perf'])
    del normalized_df['init_policy_perf']
    del normalized_df['compare_to_perf']

    # %% codecell
    normalized_df.loc[(normalized_df['spec'] == 'hallway') & (normalized_df['n_mem_states'] == 8), plot_key] = 0

    # %% codecell
    # normalized_df[normalized_df['spec'] == 'prisoners_dilemma_all_c']
    seeds = normalized_df[normalized_df['spec'] == normalized_df['spec'][0]]['seed'].unique()
    # %% codecell
    def maybe_spec_map(id: str):
        spec_map = {
            '4x3.95': '4x3',
            'cheese.95': 'cheese',
            'paint.95': 'paint',
            'shuttle.95': 'shuttle',
            'example_7': 'ex. 7',
            'tmaze_5_two_thirds_up': 'tmaze',
            'tiger-alt-start': 'tiger'
        }

    #     spec_map |= prisoners_spec_map

        if id not in spec_map:
            return id
        return spec_map[id]

    groups = normalized_df.groupby(split_by, as_index=False)
    all_means = groups.mean()
    all_means['init_improvement_perf'].clip(lower=0, upper=1, inplace=True)
    all_means[plot_key].clip(lower=0, upper=1, inplace=True)

    all_std_errs = groups.std()
    all_std_errs['init_improvement_perf'] /= np.sqrt(len(seeds))
    all_std_errs[plot_key] /= np.sqrt(len(seeds))

    # %%

    # SORTING
    sorted_mean_df = pd.DataFrame()
    sorted_std_err_df = pd.DataFrame()

    for spec in spec_plot_order:
        mean_spec_df = all_means[all_means['spec'] == spec]
        std_err_spec_df = all_std_errs[all_std_errs['spec'] == spec]
        sorted_mean_df = pd.concat([sorted_mean_df, mean_spec_df])
        sorted_std_err_df = pd.concat([sorted_std_err_df, std_err_spec_df])

    # %%
    experiments = normalized_df['experiment'].unique()
    objectives = normalized_df['objective'].unique()


    group_width = 1
    num_n_mem = list(sorted(normalized_df['n_mem_states'].unique()))
    specs = sorted_mean_df['spec'].unique()

    spec_order_mapping = np.arange(len(specs), dtype=int)

    exp_group_width = group_width / (len(experiments))
    bar_width = exp_group_width / (len(num_n_mem) + 2)

    fig, ax = plt.subplots(figsize=(12, 6))

    xlabels = [maybe_spec_map(l) for l in specs]
    x = np.arange(len(specs))

    init_improvement_perf_mean = np.array(all_means[
                                            (all_means['n_mem_states'] == num_n_mem[0]) &
                                            (all_means['experiment'] == experiments[0]) &
                                            (all_means['objective'] == objectives[0])
                                            ]['init_improvement_perf'])
    init_improvement_perf_std = np.array(all_std_errs[
                                            (all_std_errs['n_mem_states'] == num_n_mem[0]) &
                                            (all_std_errs['experiment'] == experiments[0]) &
                                            (all_std_errs['objective'] == objectives[0])
                                            ]['init_improvement_perf'])

    ax.bar(x + 0 * exp_group_width + (0 + 1) * bar_width,
           init_improvement_perf_mean,
           bar_width,
           yerr=init_improvement_perf_std,
           label='Memoryless',
           color='#5B97E0')

    mem_colors = ['#E0B625', '#DD8453', '#C44E52']
    exp_hatches = ['/', 'o', '+', '.']

    for i, exp_name in enumerate(experiments):
        means = sorted_mean_df[sorted_mean_df['experiment'] == exp_name]
        std_errs = sorted_std_err_df[sorted_std_err_df['experiment'] == exp_name]

        # means = sorted_mean_df
        # std_errs = sorted_std_err_df

        for j, n_mem_states in enumerate(num_n_mem):
            curr_mem_mean = np.array(means[means['n_mem_states'] == n_mem_states][plot_key])
            curr_mem_std = np.array(std_errs[std_errs['n_mem_states'] == n_mem_states][plot_key])
            to_add = i * exp_group_width + (j + 2) * bar_width
            if i != 0:
                to_add -= 2 * bar_width
            ax.bar(x + to_add,
                   curr_mem_mean,
                   bar_width,
                   yerr=curr_mem_std,
                   label=f"{int(np.log2(n_mem_states))} Memory Bits",
                   # hatch=exp_hatches[i],
                   color=mem_colors[j])

    ax.set_ylim([0, 1.05])
    ax.set_ylabel(f'Relative Performance\n (w.r.t. optimal {compare_to} & initial policy)')
    ax.set_xticks(x + group_width / 2)
    ax.set_xticklabels(xlabels)
    # ax.legend(bbox_to_anchor=(0.317, 0.62), framexalpha=0.95)
    # ax.set_title(f"Memory Iteration ({policy_optim_alg})")
    # alpha_str = 'uniform' if alpha == 1. else 'occupancy'
    # residual_str = 'semi_grad' if not residual else 'residual'
    title_str = " vs. ".join([f"{exp} ({hatch})" for exp, hatch in zip(experiments, exp_hatches)]) + f"\n init_policy: {plot_key}"
    # ax.set_title(f"Memory: (MSTDE (dashes, {residual_str}) vs LD (dots))")
    ax.set_title(title_str)

    plt.show()
    downloads = Path().home() / 'Downloads'
    # fig_path = downloads / f"{results_dir.stem}_{residual_str}_{alpha_str}.pdf"
    # fig.savefig(fig_path, bbox_inches='tight')
    # %% codecell
