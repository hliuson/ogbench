"""Quick sanity check: V(s, s) and V(s, s_next) for trained agents."""

import json
import pickle

import gymnasium
import jax
import jax.numpy as jnp
import numpy as np
from ml_collections import ConfigDict

import ogbench.locomaze  # noqa


def load_agent(exp_dir, epoch, env):
    with open(f'{exp_dir}/flags.json', 'r') as f:
        flag_dict = json.load(f)
    config = ConfigDict(flag_dict['agent'])
    agent_name = config['agent_name']
    if agent_name == 'cf_trl':
        from agents.cf_trl import CFTRLAgent as AgentClass
    elif agent_name == 'trl':
        from agents.trl import TRLAgent as AgentClass
    else:
        raise ValueError(f'Unknown agent: {agent_name}')
    ob_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    batch_size = 2
    example_batch = {
        'observations': np.zeros((batch_size, ob_dim)),
        'actions': np.zeros((batch_size, action_dim)),
        'actor_goals': np.zeros((batch_size, ob_dim)),
        'value_midpoint_observations': np.zeros((batch_size, ob_dim)),
    }
    agent = AgentClass.create(seed=0, example_batch=example_batch, config=config)
    params_path = f'{exp_dir}/params_{epoch}.pkl'
    with open(params_path, 'rb') as f:
        load_dict = pickle.load(f)
    import flax.serialization
    agent = flax.serialization.from_state_dict(agent, load_dict['agent'])
    print(f'Loaded {params_path}')
    return agent


def main():
    env = gymnasium.make('humanoidmaze-giant-v0', terminate_at_goal=False, max_episode_steps=1000)

    experiments = [
        ('ltrl-cf', 'exp/OGBench/ltrl-hmaze-giant-stitch-100m-cf/sd000_s_44791438.0.20260310_204950'),
        ('ltrl-base', 'exp/OGBench/ltrl-hmaze-giant-stitch-100m-baseline/sd000_s_44791437.0.20260310_204950'),
        ('trl', 'exp/OGBench/trl-hmaze-giant-stitch-100m/sd000_s_44791439.0.20260310_205543'),
    ]

    # Get a few test observations at different positions.
    import mujoco
    test_positions = [(1, 1), (5, 7), (10, 14)]
    maze_env = env.unwrapped
    maze_unit = maze_env._maze_unit
    offset_x = maze_env._offset_x
    offset_y = maze_env._offset_y

    def get_value(agent, s, g):
        """Get V(s, g) or Q(s, pi(s,g), g) depending on agent type."""
        agent_name = agent.config['agent_name']
        if agent_name == 'cf_trl':
            v_logits = agent.network.select('value')(s, goals=g)
            vs = jax.nn.sigmoid(v_logits)
            return float(jnp.minimum(vs[0], vs[1]).squeeze())
        elif agent_name == 'trl':
            action = agent.sample_actions(
                s.squeeze(0), goals=g.squeeze(0),
                seed=jax.random.PRNGKey(0), temperature=0.0,
            )
            critic_module = 'oracle_critic' if agent.config['oracle_distill'] else 'critic'
            q_logits = agent.network.select(critic_module)(
                s.squeeze(0), goals=g.squeeze(0), actions=action,
            )
            qs = jax.nn.sigmoid(q_logits)
            return float(jnp.minimum(qs[0], qs[1]).squeeze())

    for exp_name, exp_dir in experiments:
        agent = load_agent(exp_dir, 1000000, env)
        print(f'\n=== {exp_name} (discount=0.995) ===')

        for (ci, cj) in test_positions:
            # Get observation at this cell.
            sx = cj * maze_unit - offset_x
            sy = ci * maze_unit - offset_y
            ob, _ = env.reset()
            env.unwrapped.set_xy(np.array([sx, sy]))
            mujoco.mj_forward(env.unwrapped.model, env.unwrapped.data)
            obs = env.unwrapped.get_ob().astype(np.float32)

            # Get observation at a nearby cell (offset by 1 in j).
            nearby_cj = cj + 1
            if maze_env.maze_map[ci, nearby_cj] == 1:
                nearby_cj = cj - 1
            sx2 = nearby_cj * maze_unit - offset_x
            env.unwrapped.set_xy(np.array([sx2, sy]))
            mujoco.mj_forward(env.unwrapped.model, env.unwrapped.data)
            obs_nearby = env.unwrapped.get_ob().astype(np.float32)

            s = jnp.array(obs)[None]

            # V(s, s) — should be ~1.0
            v_self = get_value(agent, s, jnp.array(obs)[None])

            # V(s, s_nearby)
            v_nearby = get_value(agent, s, jnp.array(obs_nearby)[None])

            # V(s, same_xy_diff_limbs) — reset to get different limb state.
            ob2, _ = env.reset()
            env.unwrapped.set_xy(np.array([sx, sy]))
            mujoco.mj_forward(env.unwrapped.model, env.unwrapped.data)
            obs_diff_limbs = env.unwrapped.get_ob().astype(np.float32)
            v_diff_limbs = get_value(agent, s, jnp.array(obs_diff_limbs)[None])

            # Multiple resets to get variance of V(s, same_xy_diff_limbs).
            diff_limb_vals = [v_diff_limbs]
            for _ in range(9):
                env.reset()
                env.unwrapped.set_xy(np.array([sx, sy]))
                mujoco.mj_forward(env.unwrapped.model, env.unwrapped.data)
                obs_dl = env.unwrapped.get_ob().astype(np.float32)
                diff_limb_vals.append(get_value(agent, s, jnp.array(obs_dl)[None]))

            print(f'  Cell ({ci},{cj}):')
            print(f'    V(s, s)                    = {v_self:.4f}  (expected ~1.0)')
            print(f'    V(s, nearby_cell)           = {v_nearby:.4f}')
            print(f'    V(s, same_xy_diff_limbs)   = {np.mean(diff_limb_vals):.4f} +/- {np.std(diff_limb_vals):.4f}  (10 resets)')


if __name__ == '__main__':
    main()
