from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple, Tuple

import numpy as np
import matplotlib.animation
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

class Position(NamedTuple):
    x: np.int32
    y: np.int32

    def __eq__(self, other: object) -> np.ndarray:
        if not isinstance(other, Position):
            return NotImplemented
        return (self.x == other.x) & (self.y == other.y)


@dataclass
class State:
    """The state of the environment.

    key: random key used for auto-reset.
    grid: jax array (int) of the ingame maze with walls.
    pellets: int tracking the number of pellets.
    frightened_state_time: jax array (int) of shape ()
        tracks number of steps for the scatter state.
    pellet_locations: jax array (int) of pellet locations.
    power_up_locations: jax array (int) of power-up locations
    player_locations: current 2D position of agent.
    ghost_locations: jax array (int) of current ghost positions.
    initial_player_locations: starting 2D position of agent.
    initial_ghost_positions: jax array (int) of initial ghost positions.
    ghost_init_targets: jax array (int) of initial ghost targets.
        used to direct ghosts on respawn.
    old_ghost_locations: jax array (int) of shape ghost positions from last step.
        used to prevent ghost backtracking.
    ghost_init_steps: jax array (int) of number of initial ghost steps.
        used to determine per ghost initialisation.
    ghost_actions: jax array (int) of ghost action at current step.
    last_direction: (int) tracking the last direction of the player.
    dead: (bool) used to track player death.
    visited_index: jax array (int) of visited locations.
        used to prevent repeated pellet points.
    ghost_starts: jax array (int) of reset positions for ghosts
        used to reset ghost positions if eaten
    scatter_targets: jax array (int) of scatter targets.
            target locations for ghosts when scatter behavior is active.
    step_count: (int32) of total steps taken from reset till current timestep.
    ghost_eaten: jax array (bool) tracking if ghost has been eaten before.
    score: (int32) of total points aquired.
    """

    key: np.ndarray  # (2,)
    grid: np.ndarray  # (31,28)
    pellets: np.int32  # ()
    frightened_state_time: np.int32  # ()
    pellet_locations: np.ndarray  # (316,2)
    power_up_locations: np.ndarray  # (4,2)
    player_locations: Position  # Position(row, col) each of shape ()
    ghost_locations: np.ndarray  # (4,2)
    initial_player_locations: Position  # Position(row, col) each of shape ()
    initial_ghost_positions: np.ndarray  # (4,2)
    ghost_init_targets: np.ndarray  # (4,2)
    old_ghost_locations: np.ndarray  # (4,2)
    ghost_init_steps: np.ndarray  # (4,)
    ghost_actions: np.ndarray  # (4,)
    last_direction: np.int32  # ()
    dead: bool  # ()
    visited_index: np.ndarray  # (320,2)
    ghost_starts: np.ndarray  # (4,2)
    scatter_targets: np.ndarray  # (4,2)
    step_count: np.int32  # ()
    ghost_eaten: np.ndarray  # (4,)
    score: np.int32  # ()

def create_grid_image(observation: State) -> np.ndarray:
    """
    Generate the observation of the current state.

    Args:
        state: 'State` object corresponding to the new state of the environment.

    Returns:
        rgb: A 3-dimensional array representing the RGB observation of the current state.
    """

    # Make walls blue and passages black
    layer_1 = (1 - observation.grid) * 0.0
    layer_2 = (1 - observation.grid) * 0.0
    layer_3 = (1 - observation.grid) * 0.6

    player_loc = observation.player_locations
    ghost_pos = observation.ghost_locations
    pellets_loc = observation.power_up_locations
    is_scared = observation.frightened_state_time
    idx = observation.pellet_locations
    n = 3

    # Power pellet are pink
    for i in range(len(pellets_loc)):
        p = pellets_loc[i]
        layer_1[p[1], p[0]] = 1.0
        layer_2[p[1], p[0]] = 0.8
        layer_3[p[1], p[0]] = 0.6

    # Set player is yellow
    layer_1[player_loc.x, player_loc.y] = 1
    layer_2[player_loc.x, player_loc.y] = 1
    layer_3[player_loc.x, player_loc.y] = 0

    cr = np.array([1, 1, 0, 1])
    cg = np.array([0, 0.7, 1, 0.5])
    cb = np.array([0, 1, 1, 0.0])
    # Set ghost locations

    layers = (layer_1, layer_2, layer_3)

    def set_ghost_colours(
            layers: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        layer_1, layer_2, layer_3 = layers
        for i in range(4):
            y = ghost_pos[i][0]
            x = ghost_pos[i][1]

            layer_1[x, y] = cr[i]
            layer_2[x, y] = cg[i]
            layer_3[x, y] = cb[i]
        return layer_1, layer_2, layer_3

    def set_ghost_colours_scared(
            layers: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        layer_1, layer_2, layer_3 = layers
        for i in range(4):
            y = ghost_pos[i][0]
            x = ghost_pos[i][1]
            layer_1[x, y] = 0
            layer_2[x, y] = 0
            layer_3[x, y] = 1
        return layer_1, layer_2, layer_3

    if is_scared > 0:
        layers = set_ghost_colours_scared(layers)
    else:
        layers = set_ghost_colours(layers)

    layer_1, layer_2, layer_3 = layers

    layer_1[0, 0] = 0
    layer_2[0, 0] = 0
    layer_3[0, 0] = 0.6

    obs = [layer_1, layer_2, layer_3]
    rgb = np.stack(obs, axis=-1)

    expand_rgb = np.kron(rgb, np.ones((n, n, 1)))
    layer_1 = expand_rgb[:, :, 0]
    layer_2 = expand_rgb[:, :, 1]
    layer_3 = expand_rgb[:, :, 2]

    # place normal pellets
    for i in range(len(idx)):
        if np.array(idx[i]).sum != 0:
            loc = idx[i]
            c = loc[1] * n + 1
            r = loc[0] * n + 1
            layer_1[c, r] = 1.0
            layer_2[c, r] = 0.8
            layer_3[c, r] = 0.6

    layers = (layer_1, layer_2, layer_3)

    # Draw details
    def set_ghost_colours_details(
            layers: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        layer_1, layer_2, layer_3 = layers
        for i in range(4):
            y = ghost_pos[i][0]
            x = ghost_pos[i][1]
            c = x * n + 1
            r = y * n + 1

            layer_1[c, r] = cr[i]
            layer_2[c, r] = cg[i]
            layer_3[c, r] = cb[i]

            # Make notch in top
            layer_1[c - 1, r - 1] = 0.0
            layer_2[c - 1, r - 1] = 0.0
            layer_3[c - 1, r - 1] = 0.0

            # Make notch in top
            layer_1[c - 1, r + 1] = 0.0
            layer_2[c - 1, r + 1] = 0.0
            layer_3[c - 1, r + 1] = 0.0

            # Eyes
            layer_1[c, r + 1] = 1
            layer_2[c, r + 1] = 1
            layer_3[c, r + 1] = 1

            layer_1[c, r - 1] = 1
            layer_2[c, r - 1] = 1
            layer_3[c, r - 1] = 1

        return layer_1, layer_2, layer_3

    def set_ghost_colours_scared_details(
            layers: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        layer_1, layer_2, layer_3 = layers
        for i in range(4):
            y = ghost_pos[i][0]
            x = ghost_pos[i][1]

            c = x * n + 1
            r = y * n + 1

            layer_1[x * n + 1, y * n + 1] = 0
            layer_2[x * n + 1, y * n + 1] = 0
            layer_3[x * n + 1, y * n + 1] = 1

            # Make notch in top
            layer_1[c - 1, r - 1] = 0.0
            layer_2[c - 1, r - 1] = 0.0
            layer_3[c - 1, r - 1] = 0.0

            # Make notch in top
            layer_1[c - 1, r + 1] = 0.0
            layer_2[c - 1, r + 1] = 0.0
            layer_3[c - 1, r + 1] = 0.0

            # Eyes
            layer_1[c, r + 1] = 1
            layer_2[c, r + 1] = 0.6
            layer_3[c, r + 1] = 0.2

            layer_1[c, r - 1] = 1
            layer_2[c, r - 1] = 0.6
            layer_3[c, r - 1] = 0.2

        return layer_1, layer_2, layer_3

    if is_scared > 0:
        layers = set_ghost_colours_scared_details(layers)
    else:
        layers = set_ghost_colours_details(layers)

    layer_1, layer_2, layer_3 = layers

    # Power pellet is pink
    for i in range(len(pellets_loc)):
        p = pellets_loc[i]
        layer_1[p[1] * n + 2, p[0] * n + 1] = 1
        layer_2[p[1] * n + 1, p[0] * n + 1] = 0.8
        layer_3[p[1] * n + 1, p[0] * n + 1] = 0.6

    # Set player is yellow
    layer_1[player_loc.x * n + 1, player_loc.y * n + 1] = 1
    layer_2[player_loc.x * n + 1, player_loc.y * n + 1] = 1
    layer_3[player_loc.x * n + 1, player_loc.y * n + 1] = 0

    obs = [layer_1, layer_2, layer_3]
    rgb = np.stack(obs, axis=-1)
    expand_rgb

    return rgb


def visualize_grid(grid, fig, ax, add_colorbar=False):
    # Check if the grid has the correct dimensions
    if grid.shape != (21, 19):
        raise ValueError("Grid must be 21x19 in size")

    # Create a custom colormap for values between 0 and 1
    cmap = mcolors.LinearSegmentedColormap.from_list('custom_cmap', ['white', '#f39c12'])  # Using a brighter blue

    # Create a masked array for the grid
    masked_grid = np.ma.masked_where(grid == -1, grid)

    # Plot the grid
    if fig is None:
        fig, ax = plt.subplots()
    cax = ax.imshow(masked_grid, cmap=cmap, vmin=0, vmax=1)

    # Add color bar
    if add_colorbar:
        cbar = fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Value')

    # Plot the -1 values as black
    for (i, j), value in np.ndenumerate(grid):
        if value == -1:
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color='black'))

    # Set grid lines
    ax.set_xticks(np.arange(-.5, 19, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 21, 1), minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
    ax.tick_params(which='minor', size=0)

    if fig is None:
        # Display the grid
        plt.show()


def load_info(results_path: Path) -> dict:
    return np.load(results_path, allow_pickle=True).item()


def pack_states(states: list[dict]) -> list[State]:
    packed_states = []
    for s in states:
        dict_s = {}
        for k, v in list(s.items()):
            if isinstance(v, dict):
                assert 'x' in v and 'y' in v
                dict_s[k] = Position(x=v['x'], y=v['y'])
            else:
                assert isinstance(v, np.ndarray) or isinstance(v, list)
                dict_s[k] = v
        packed_states.append(State(**dict_s))
    return packed_states


if __name__ == "__main__":
    traj_path = Path('../../results/pocman_pellet_probe_trajectory_bidx_1.npy')
    save_vod_path = Path('../../results/pocman_pellet_probe_trajectory_bidx_1.mp4')

    dataset = load_info(traj_path)
    states, predictions = pack_states(dataset['states']), dataset['predictions']

    # now we make our animation
    fig, axes = plt.subplots(3, num=f"PocmanPredictionAnimation", figsize=(4, 12))

    def make_frame(idx: int) -> None:
        state_ax, p0_ax, p1_ax = axes
        for a in axes:
            a.clear()

        # First we make our state image
        state = states[idx]
        state_img = create_grid_image(state)
        state_ax.set_axis_off()
        state_ax.imshow(state_img)

        # Now we make our prediction images
        prediction0, prediction1 = predictions[idx]
        visualize_grid(prediction0, fig, p0_ax, add_colorbar=(idx==0))
        visualize_grid(prediction1, fig, p1_ax, add_colorbar=(idx==0))

        fig.suptitle(f"PacMan    Score: {int(state.score)}", size=10)


    animation = matplotlib.animation.FuncAnimation(
        fig,
        make_frame,
        frames=len(states),
        interval=400,
    )


    # plt.show()
    # Save the animation as a gif.
    animation.save(save_vod_path)

    print(f"Saved animation to {save_vod_path}")
