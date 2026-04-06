from typing import Any

import jax
import jax.numpy as jnp


class QActionPolicyExtractionMixin:
    """Shared q_action + actor policy extraction used by lightweight baselines."""

    @staticmethod
    def _get_pe_info_from_config(config):
        if config['pe_type'] == 'discrete':
            return config['pe_discrete']
        return config[config['pe_type']]

    def _get_pe_info(self):
        return self._get_pe_info_from_config(self.config)

    @staticmethod
    def bce_loss(pred_logit, target):
        log_pred = jax.nn.log_sigmoid(pred_logit)
        log_not_pred = jax.nn.log_sigmoid(-pred_logit)
        return -(log_pred * target + log_not_pred * (1 - target))

    def q_action_loss(self, batch, grad_params):
        """Distill q_action(s, a, g) from an n-step bootstrap into target value."""
        q_action_n_step = int(self.config.get('q_action_n_step', 1))
        next_obs_key = (
            'actor_nstep_observations'
            if q_action_n_step > 1 and 'actor_nstep_observations' in batch
            else 'next_observations'
        )
        next_observations = batch[next_obs_key]
        actor_goals = batch['actor_goals']

        v_next = self.network.select('target_value')(next_observations, goals=actor_goals)
        v_next = jax.nn.sigmoid(v_next)
        if v_next.ndim > 1:
            v_next = jnp.minimum(v_next[0], v_next[1])

        if q_action_n_step > 1 and 'actor_nstep_steps' in batch:
            n_steps = batch['actor_nstep_steps']
        else:
            n_steps = jnp.ones_like(batch['actions'][..., 0], dtype=jnp.int32)
        bootstrap_target = (self.config['discount'] ** n_steps) * v_next

        if q_action_n_step > 1 and 'actor_goals_is_intraj' in batch and 'actor_goal_offsets' in batch:
            actor_is_intraj = batch['actor_goals_is_intraj']
            actor_goal_offsets = batch['actor_goal_offsets']
            exact_reached = actor_is_intraj * (actor_goal_offsets <= n_steps)
            exact_target = self.config['discount'] ** actor_goal_offsets
            target = jnp.where(exact_reached > 0, exact_target, bootstrap_target)
        else:
            exact_reached = jnp.zeros_like(bootstrap_target)
            target = bootstrap_target

        q_action_logits = self.network.select('q_action')(
            batch['observations'],
            goals=actor_goals,
            actions=batch['actions'],
            params=grad_params,
        )
        q_action = jax.nn.sigmoid(q_action_logits)
        q_action_loss = self.bce_loss(q_action_logits, jax.lax.stop_gradient(target)).mean()

        return q_action_loss, {
            'q_action_loss': q_action_loss,
            'q_action_mean': q_action.mean(),
            'q_action_max': q_action.max(),
            'q_action_min': q_action.min(),
            'v_next_target_mean': target.mean(),
            'n_step': jnp.asarray(q_action_n_step, dtype=jnp.float32),
            'exact_reached_frac': exact_reached.mean(),
        }

    def actor_loss(self, batch, grad_params, rng=None):
        """Use the same action extraction interface as the main latent TRL agent."""
        pe_info = self._get_pe_info()

        if self.config['pe_type'] == 'rpg':
            dist = self.network.select('actor')(
                batch['observations'],
                batch['actor_goals'],
                params=grad_params,
            )
            if pe_info['const_std']:
                q_actions = jnp.clip(dist.mode(), -1, 1)
            else:
                q_actions = jnp.clip(dist.sample(seed=rng), -1, 1)
            u_next = batch['next_observations']
            v_next = self.network.select('value')(
                u_next,
                goals=batch['actor_goals'],
            )
            if v_next.ndim > 1:
                v_next = jnp.minimum(v_next[0], v_next[1])
            v_loss = -v_next.mean() / jax.lax.stop_gradient(jnp.abs(v_next).mean() + 1e-6)
            log_prob = dist.log_prob(batch['actions'])
            bc_loss = -(pe_info['alpha'] * log_prob).mean()
            actor_loss = v_loss + bc_loss
            return actor_loss, {
                'actor_loss': actor_loss,
                'v_loss': v_loss.mean(),
                'bc_loss': bc_loss,
                'v_mean': v_next.mean(),
                'v_abs_mean': jnp.abs(v_next).mean(),
                'bc_log_prob': log_prob.mean(),
                'mse': jnp.mean((dist.mode() - batch['actions']) ** 2),
                'std': jnp.mean(dist.scale_diag),
            }

        if self.config['pe_type'] == 'frs':
            batch_size, action_dim = batch['actions'].shape
            x_rng, t_rng = jax.random.split(rng, 2)
            x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
            x_1 = batch['actions']
            t = jax.random.uniform(t_rng, (batch_size, 1))
            x_t = (1 - t) * x_0 + t * x_1
            y = x_1 - x_0
            pred = self.network.select('actor')(
                batch['observations'],
                batch['actor_goals'],
                x_t,
                t,
                params=grad_params,
            )
            actor_loss = jnp.mean((pred - y) ** 2)
            return actor_loss, {
                'actor_loss': actor_loss,
            }

        raise ValueError(f"Unsupported pe_type for q_action baselines: {self.config['pe_type']}")

    def target_update(self, network, module_name):
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @jax.jit
    def sample_actions(
        self,
        observations,
        goals=None,
        seed=None,
        temperature=1.0,
    ):
        pe_info = self._get_pe_info()

        if self.config['pe_type'] == 'frs':
            n_observations = jnp.repeat(jnp.expand_dims(observations, 0), pe_info['num_samples'], axis=0)
            n_goals = jnp.repeat(jnp.expand_dims(goals, 0), pe_info['num_samples'], axis=0)
            n_actions = jax.random.normal(
                seed,
                (
                    pe_info['num_samples'],
                    *observations.shape[:-1],
                    self.config['action_dim'],
                ),
            )
            for i in range(pe_info['flow_steps']):
                t = jnp.full(
                    (pe_info['num_samples'], *observations.shape[:-1], 1),
                    i / pe_info['flow_steps'],
                )
                vels = self.network.select('actor')(n_observations, n_goals, n_actions, t)
                n_actions = n_actions + vels / pe_info['flow_steps']
            n_actions = jnp.clip(n_actions, -1, 1)

            q_action = self.network.select('q_action')(
                n_observations,
                goals=n_goals,
                actions=n_actions,
            )
            if len(observations.shape) == 2:
                return n_actions[jnp.argmax(q_action, axis=0), jnp.arange(observations.shape[0])]
            return n_actions[jnp.argmax(q_action)]

        dist = self.network.select('actor')(observations, goals, temperature=temperature)
        actions = dist.sample(seed=seed)
        return jnp.clip(actions, -1, 1)
