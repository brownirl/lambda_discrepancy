from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint

from lamb.utils.file_system import numpyify_and_save
from definitions import ROOT_DIR


if __name__ == "__main__":

    jax.config.update('jax_platform_name', 'cpu')
    d0_path = Path("../results/pocman_ppo_best_ckpt/buffer_1000000_timestep_2_seed_2024_390f282614e5b7398cf10e565ea811e7")
    d1_path = Path("../results/pocman_LD_ppo_best_ckpt/buffer_1000000_timestep_2_seed_2024_e232a0c2fa52cb6e68f01be9dac6b7b9")

    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    restored = orbax_checkpointer.restore(d0_path)

    args_0, ckpt_0, dataset_0 = restored['args'], restored['ckpt'], restored['dataset']
    dataset_0['ghost_occupancy'] = dataset_0['ghost_occupancy'].astype(np.int8)

    restored = orbax_checkpointer.restore(d1_path)

    args_1, ckpt_1, dataset_1 = restored['args'], restored['ckpt'], restored['dataset']
    dataset_1['ghost_occupancy'] = dataset_1['ghost_occupancy'].astype(np.int8)

    combined_dataset = jax.tree.map(lambda x, y: jnp.concatenate((x, y), axis=0), dataset_0, dataset_1)

    save_dir = Path(ROOT_DIR, 'results', 'combined_probe_datasets')
    save_dir.mkdir(exist_ok=True)
    save_path = save_dir / f'combined_{d0_path.stem.split("_")[-1]}.npy'
    to_save = {
        'args': [args_0, args_1],
        'ckpt': [ckpt_0, ckpt_1],
        'dataset': combined_dataset
    }

    print(f"Saving results to {save_path}")
    numpyify_and_save(save_path, to_save)

    print("Done.")
