"""Visualize learned value function V(s, g) as a 2D heatmap over the maze.

For a fixed starting observation s, evaluates V(s, g) for goals g at a grid of
(x, y) positions and plots the result as a heatmap overlaid on the maze.

Usage (single experiment):
    uv run python viz_value_map.py \
        --exp_dir=exp/OGBench/ltrl-hmaze-giant-stitch-100m-baseline/sd000_... \
        --epoch=1000000 \
        --env_name=humanoidmaze-giant-navigate-v0

Usage (comparison of multiple experiments):
    uv run python viz_value_map.py \
        --exp_dir=exp1/sd000_...,exp2/sd000_...,exp3/sd000_... \
        --epoch=1000000 \
        --env_name=humanoidmaze-giant-navigate-v0
"""

import pathlib
import json
import pickle

import gymnasium
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from absl import app, flags
from ml_collections import ConfigDict

import ogbench.locomaze  # noqa

FLAGS = flags.FLAGS

flags.DEFINE_string('exp_dir', None, 'Experiment directory (or comma-separated list for comparison).')
flags.DEFINE_integer('epoch', 1000000, 'Checkpoint epoch to load.')
flags.DEFINE_string('env_name', 'humanoidmaze-giant-v0', 'Environment name.')
flags.DEFINE_integer('grid_res', 50, 'Grid resolution per maze cell.')
flags.DEFINE_string('output', None, 'Output path (default: <exp_dir>/value_map.png).')
flags.DEFINE_string('start_cells', None, 'Semicolon-separated start cells as i,j pairs e.g. "1,1;5,5".')
flags.DEFINE_string('exp_labels', None, 'Comma-separated labels for experiments (default: inferred from path).')
flags.DEFINE_bool('show_steps', False, 'Show implied step distance log(V)/log(discount) instead of raw V.')
flags.DEFINE_float('discount', 0.995, 'Discount factor (used for step conversion when --show_steps).')
flags.DEFINE_integer('max_steps', 1000, 'Clamp step distance display at this value.')
flags.DEFINE_integer('n_goal_samples', 1, 'Number of robot poses to sample and average per goal cell.')


def extract_exp_name(exp_dir):
    """Extract a readable experiment name from the path."""
    p = pathlib.Path(exp_dir.rstrip('/'))
    # Path like .../OGBench/<exp_name>/sd000_s_...
    return p.parent.name


def load_agent(exp_dir, epoch, env):
    """Load agent from checkpoint."""
    with open(f'{exp_dir}/flags.json', 'r') as f:
        flag_dict = json.load(f)

    config = ConfigDict(flag_dict['agent'])

    # Import the right agent class.
    agent_name = config['agent_name']
    if agent_name == 'latent_trl':
        from agents.latent_trl import LatentTRLAgent as AgentClass
    elif agent_name == 'trl':
        from agents.trl import TRLAgent as AgentClass
    else:
        raise ValueError(f'Unknown agent: {agent_name}')

    # Create agent with dummy example batch.
    ob_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    batch_size = 2
    example_batch = {
        'observations': np.zeros((batch_size, ob_dim)),
        'actions': np.zeros((batch_size, action_dim)),
        'actor_goals': np.zeros((batch_size, ob_dim)),
        'value_midpoint_observations': np.zeros((batch_size, ob_dim)),
    }
    agent = AgentClass.create(
        seed=0,
        example_batch=example_batch,
        config=config,
    )

    # Restore params.
    params_path = f'{exp_dir}/params_{epoch}.pkl'
    with open(params_path, 'rb') as f:
        load_dict = pickle.load(f)
    import flax.serialization
    agent = flax.serialization.from_state_dict(agent, load_dict['agent'])
    print(f'Loaded {params_path}')
    return agent


def get_maze_info(env):
    """Extract maze map and coordinate system."""
    maze_env = env.unwrapped
    maze_map = maze_env.maze_map
    maze_unit = maze_env._maze_unit
    offset_x = maze_env._offset_x
    offset_y = maze_env._offset_y
    return maze_map, maze_unit, offset_x, offset_y


def make_goal_grid(maze_map, maze_unit, offset_x, offset_y, resolution):
    """Create a grid of (x, y) positions covering the maze, with mask for open cells."""
    rows, cols = maze_map.shape

    # Compute xy bounds of the maze.
    x_min = 0 * maze_unit - offset_x - maze_unit / 2
    x_max = (cols - 1) * maze_unit - offset_x + maze_unit / 2
    y_min = 0 * maze_unit - offset_y - maze_unit / 2
    y_max = (rows - 1) * maze_unit - offset_y + maze_unit / 2

    xs = np.linspace(x_min, x_max, resolution)
    ys = np.linspace(y_min, y_max, resolution)
    xx, yy = np.meshgrid(xs, ys)

    # Mask: which grid points are in open cells.
    mask = np.zeros_like(xx, dtype=bool)
    for gi in range(resolution):
        for gj in range(resolution):
            x, y = xx[gi, gj], yy[gi, gj]
            ci = int((y + offset_y + 0.5 * maze_unit) / maze_unit)
            cj = int((x + offset_x + 0.5 * maze_unit) / maze_unit)
            if 0 <= ci < rows and 0 <= cj < cols and maze_map[ci, cj] == 0:
                mask[gi, gj] = True

    return xx, yy, mask, (x_min, x_max, y_min, y_max)


def make_observation_at_xy(env, xy):
    """Reset the env and place the agent at the given xy, return the observation."""
    ob, _ = env.reset()
    env.unwrapped.set_xy(np.array(xy))
    import mujoco
    mujoco.mj_forward(env.unwrapped.model, env.unwrapped.data)
    ob = env.unwrapped.get_ob()
    return ob.astype(np.float32)


def compute_goal_observations(env, flat_xys, n_samples=1):
    """Compute full observations at each goal xy position.

    Args:
        n_samples: Number of random robot poses to sample per cell.
            If 1, returns shape (n_goals, ob_dim).
            If >1, returns shape (n_samples, n_goals, ob_dim).
    """
    ob_dim = env.observation_space.shape[0]
    if n_samples == 1:
        goal_obs = np.empty((len(flat_xys), ob_dim), dtype=np.float32)
        for i, xy in enumerate(flat_xys):
            goal_obs[i] = make_observation_at_xy(env, xy)
        return goal_obs
    else:
        goal_obs = np.empty((n_samples, len(flat_xys), ob_dim), dtype=np.float32)
        for s in range(n_samples):
            for i, xy in enumerate(flat_xys):
                goal_obs[s, i] = make_observation_at_xy(env, xy)
        return goal_obs


def get_values(agent, s_batch, goal_obs):
    """Get V(s, g) values, dispatching based on agent type."""
    agent_name = agent.config['agent_name']

    if agent_name == 'latent_trl':
        v_logits = agent.network.select('value')(
            jnp.array(s_batch),
            goals=jnp.array(goal_obs),
        )
        vs = jax.nn.sigmoid(v_logits)
        v_min = jnp.minimum(vs[0], vs[1])
        return np.array(v_min)

    elif agent_name == 'trl':
        # TRL has no standalone V(s,g) — use Q(s, pi(s,g), g).
        # Process one obs at a time since FRS sampling creates extra dims.
        n = len(s_batch)
        all_values = np.empty(n, dtype=np.float32)
        critic_module = 'oracle_critic' if agent.config['oracle_distill'] else 'critic'
        for i in range(n):
            s_i = jnp.array(s_batch[i])
            g_i = jnp.array(goal_obs[i])
            action = agent.sample_actions(
                s_i, goals=g_i,
                seed=jax.random.PRNGKey(0), temperature=0.0,
            )
            q_logits = agent.network.select(critic_module)(
                s_i, goals=g_i, actions=action,
            )
            qs = jax.nn.sigmoid(q_logits)
            all_values[i] = float(jnp.minimum(qs[0], qs[1]))
        return all_values

    else:
        raise ValueError(f'Unknown agent type: {agent_name}')


def evaluate_values(agent, start_obs, goal_obs):
    """Evaluate V(s, g) for a fixed s and precomputed goal observations.

    If goal_obs has shape (n_samples, n_goals, ob_dim), averages values across samples.
    If goal_obs has shape (n_goals, ob_dim), evaluates directly.
    """
    if goal_obs.ndim == 3:
        # Multiple samples: evaluate each and average.
        n_samples, n_goals, _ = goal_obs.shape
        all_values = np.zeros(n_goals, dtype=np.float32)
        for s in range(n_samples):
            s_batch = np.tile(start_obs, (n_goals, 1))
            all_values += get_values(agent, s_batch, goal_obs[s])
        return all_values / n_samples
    else:
        n_goals = len(goal_obs)
        s_batch = np.tile(start_obs, (n_goals, 1))
        return get_values(agent, s_batch, goal_obs)


def draw_maze_walls(ax, maze_map, maze_unit, offset_x, offset_y):
    """Draw maze walls on an axis."""
    rows, cols = maze_map.shape
    for i in range(rows):
        for j in range(cols):
            if maze_map[i, j] == 1:
                wx = j * maze_unit - offset_x
                wy = i * maze_unit - offset_y
                rect = plt.Rectangle(
                    (wx - maze_unit / 2, wy - maze_unit / 2),
                    maze_unit, maze_unit,
                    facecolor='gray', edgecolor='black', linewidth=0.5,
                )
                ax.add_patch(rect)


def plot_value_maps_comparison(agents, exp_names, env, maze_map, maze_unit,
                               offset_x, offset_y, start_cells, resolution,
                               output_path):
    """Plot value map comparison: rows=start cells, columns=experiments."""
    xx, yy, mask, bounds = make_goal_grid(maze_map, maze_unit, offset_x, offset_y, resolution)
    flat_xys = np.stack([xx[mask], yy[mask]], axis=-1)

    # Compute goal observations once.
    n_goal_samples = FLAGS.n_goal_samples
    print(f'Computing goal observations for {mask.sum()} open cells (n_samples={n_goal_samples})...')
    goal_obs = compute_goal_observations(env, flat_xys, n_samples=n_goal_samples)

    n_starts = len(start_cells)
    n_exps = len(agents)

    # Use gridspec to reserve space for colorbar.
    fig = plt.figure(figsize=(5 * n_exps + 0.5, 5 * n_starts))
    gs = fig.add_gridspec(n_starts, n_exps + 1, width_ratios=[1] * n_exps + [0.05],
                          wspace=0.05, hspace=0.15)
    axes = np.empty((n_starts, n_exps), dtype=object)
    for r in range(n_starts):
        for c in range(n_exps):
            axes[r, c] = fig.add_subplot(gs[r, c])

    show_steps = FLAGS.show_steps
    discount = FLAGS.discount
    max_steps = FLAGS.max_steps

    im = None
    for row, (si, sj) in enumerate(start_cells):
        sx, sy = sj * maze_unit - offset_x, si * maze_unit - offset_y
        start_obs = make_observation_at_xy(env, [sx, sy])

        for col, (agent, name) in enumerate(zip(agents, exp_names)):
            ax = axes[row, col]
            print(f'Evaluating {name} from cell ({si},{sj})...')
            values = evaluate_values(agent, start_obs, goal_obs)

            if show_steps:
                # Convert V to implied step distance, clamped.
                steps = np.log(np.clip(values, 1e-8, 1.0)) / np.log(discount)
                steps = np.clip(steps, 0, max_steps)
                display_map = np.full(xx.shape, np.nan)
                display_map[mask] = steps
                cmap = 'turbo_r'  # Reversed: close=hot, far=cold.
                vmin, vmax = 0, max_steps
                cbar_label = 'Implied steps'
            else:
                display_map = np.full(xx.shape, np.nan)
                display_map[mask] = values
                cmap = 'turbo'
                vmin, vmax = 0, 1
                cbar_label = 'V(s, g)'

            draw_maze_walls(ax, maze_map, maze_unit, offset_x, offset_y)

            im = ax.pcolormesh(
                xx, yy, display_map,
                cmap=cmap, vmin=vmin, vmax=vmax,
                shading='auto',
            )
            ax.plot(sx, sy, 'c*', markersize=15, markeredgecolor='white', markeredgewidth=1)

            if row == 0:
                ax.set_title(name, fontsize=10)
            if col == 0:
                ax.set_ylabel(f'Start ({si},{sj})')

            ax.set_aspect('equal')
            ax.set_xlim(bounds[0], bounds[1])
            ax.set_ylim(bounds[2], bounds[3])
            ax.set_xticks([])
            ax.set_yticks([])

    if im is not None:
        cbar_ax = fig.add_subplot(gs[:, -1])
        fig.colorbar(im, cax=cbar_ax, label=cbar_label)

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f'Saved to {output_path}')
    plt.close()


def main(_):
    env = gymnasium.make(FLAGS.env_name, terminate_at_goal=False, max_episode_steps=1000)
    maze_map, maze_unit, offset_x, offset_y = get_maze_info(env)

    # Parse experiment directories.
    exp_dirs = [d.strip() for d in FLAGS.exp_dir.split(',')]

    # Parse labels.
    if FLAGS.exp_labels:
        exp_names = [l.strip() for l in FLAGS.exp_labels.split(',')]
    else:
        exp_names = [extract_exp_name(d) for d in exp_dirs]

    # Load agents.
    agents = [load_agent(d, FLAGS.epoch, env) for d in exp_dirs]

    # Parse start cells.
    if FLAGS.start_cells:
        start_cells = []
        for pair in FLAGS.start_cells.split(';'):
            i, j = pair.strip().split(',')
            start_cells.append((int(i), int(j)))
    else:
        open_cells = []
        for i in range(maze_map.shape[0]):
            for j in range(maze_map.shape[1]):
                if maze_map[i, j] == 0:
                    open_cells.append((i, j))
        # Pick 4 evenly spaced.
        idxs = np.linspace(0, len(open_cells) - 1, 4, dtype=int)
        start_cells = [open_cells[i] for i in idxs]

    if len(exp_dirs) == 1:
        output = FLAGS.output or f'{exp_dirs[0]}/value_map.png'
    else:
        output = FLAGS.output or 'value_map_comparison.png'

    plot_value_maps_comparison(agents, exp_names, env, maze_map, maze_unit,
                               offset_x, offset_y, start_cells, FLAGS.grid_res,
                               output)


if __name__ == '__main__':
    app.run(main)
