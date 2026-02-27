import copy
from typing import Any, Sequence

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import optax
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import ActorVectorField, GCActor, GCDiscreteActor, GCValue, MLP


class SubgoalEncoder(nn.Module):
    """Midpoint encoder: s_mid -> z_mid."""

    hidden_dims: Sequence[int]
    layer_norm: bool
    z_dim: int

    @nn.compact
    def __call__(self, x):
        return MLP((*self.hidden_dims, self.z_dim), activate_final=False, layer_norm=self.layer_norm)(x)


class LatentTRLAgent(flax.struct.PyTreeNode):
    """Minimal latent TRL with in-trajectory midpoint supervision and a flat actor."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    @staticmethod
    def _get_pe_info_from_config(config):
        if config['pe_type'] == 'discrete':
            return config['pe_discrete']
        return config[config['pe_type']]

    def _get_pe_info(self):
        return self._get_pe_info_from_config(self.config)

    def _encode_subgoal(self, midpoint_observations, grad_params=None):
        if grad_params is None:
            return self.network.select('subgoal_encoder')(midpoint_observations)
        return self.network.select('subgoal_encoder')(midpoint_observations, params=grad_params)

    def _use_hybrid_state_midpoint(self):
        cutoff = self.config['hybrid_state_midpoint_max_offset']
        return cutoff is not None and cutoff >= 0

    def _hybrid_short_mask(self, batch):
        return (batch['value_midpoint_offsets'] <= self.config['hybrid_state_midpoint_max_offset'])[None, ...]

    @staticmethod
    def bce_loss(pred_logit, target):
        log_pred = jax.nn.log_sigmoid(pred_logit)
        log_not_pred = jax.nn.log_sigmoid(-pred_logit)
        return -(log_pred * target + log_not_pred * (1 - target))

    def _aggregate_q_ensembles(self, values):
        if self.config['q_agg'] == 'min':
            return jnp.minimum(values[0], values[1])
        if self.config['q_agg'] == 'mean':
            return values.mean(axis=0)
        raise ValueError(f"Unsupported q_agg: {self.config['q_agg']}")

    def q_loss(self, batch, grad_params):
        """Train Q with optional short-horizon state-midpoint fallback.

        Core latent TRL target: Q(s, z_mid, g) <- V(s, z_mid) * V(z_mid, g),
        where z_mid = enc(s_mid).
        """
        goal_key = 'value_goal_observations' if self.config['oracle_distill'] else 'value_goals'
        midpoint_goal_key = 'value_midpoint_observations' if self.config['oracle_distill'] else 'value_midpoint_goals'

        z_mid = self._encode_subgoal(batch['value_midpoint_observations'], grad_params=grad_params)
        q_latent_logits = self.network.select('q')(
            batch['observations'],
            goals=batch[goal_key],
            actions=z_mid,
            params=grad_params,
        )
        q_logits = q_latent_logits
        q_state = None
        if self._use_hybrid_state_midpoint():
            q_state_logits = self.network.select('q_state')(
                batch['observations'],
                goals=batch[goal_key],
                actions=batch['value_midpoint_observations'],
                params=grad_params,
            )
            short_mask = self._hybrid_short_mask(batch)
            q_logits = jnp.where(short_mask, q_state_logits, q_latent_logits)
            q_state = jax.nn.sigmoid(q_state_logits)
        qs = jax.nn.sigmoid(q_logits)

        second_offset = batch['value_offsets'][None, ...] - batch['value_midpoint_offsets']
        z_mid_target = self._encode_subgoal(batch['value_midpoint_observations'])
        z_mid_target = jax.lax.stop_gradient(z_mid_target)

        # State-space factors (teacher values over concrete midpoint state s').
        state_first_v_logits = self.network.select('target_value')(
            batch['observations'],
            goals=batch[midpoint_goal_key],
        )
        state_second_v_logits = self.network.select('target_value')(
            batch['value_midpoint_observations'],
            goals=batch[goal_key],
        )

        # Latent factors V(s, z) and V(z, g). Use the same latent-target choice as alt Bellman.
        if self.config['alt_bellman_use_target_latent_values']:
            latent_first_v_logits = self.network.select('target_value_sz')(
                batch['observations'],
                goals=z_mid_target,
            )
            latent_second_v_logits = self.network.select('target_value_zg')(
                z_mid_target,
                goals=batch[goal_key],
            )
        else:
            latent_first_v_logits = self.network.select('value_sz')(
                batch['observations'],
                goals=z_mid_target,
                params=grad_params,
            )
            latent_second_v_logits = self.network.select('value_zg')(
                z_mid_target,
                goals=batch[goal_key],
                params=grad_params,
            )

        if self.config['alt_bellman_backup']:
            first_v_logits = latent_first_v_logits
            second_v_logits = latent_second_v_logits
        else:
            first_v_logits = state_first_v_logits
            second_v_logits = state_second_v_logits

        first_v = jnp.where(
            (batch['value_midpoint_offsets'] <= 1)[None, ...],
            self.config['discount'] ** batch['value_midpoint_offsets'][None, ...],
            jax.nn.sigmoid(first_v_logits),
        )
        second_v = jnp.where(
            (second_offset <= 1)[None, ...],
            self.config['discount'] ** second_offset[None, ...],
            jax.nn.sigmoid(second_v_logits),
        )
        target = first_v * second_v

        # Single-sample covariance residual estimator:
        # (V(s,s') - V(s,z)) * (V(s',g) - V(z,g))
        state_first_v = jnp.where(
            (batch['value_midpoint_offsets'] <= 1)[None, ...],
            self.config['discount'] ** batch['value_midpoint_offsets'][None, ...],
            jax.nn.sigmoid(state_first_v_logits),
        )
        state_second_v = jnp.where(
            (second_offset <= 1),
            self.config['discount'] ** second_offset,
            jax.nn.sigmoid(state_second_v_logits),
        )
        latent_first_v = jax.nn.sigmoid(latent_first_v_logits)
        latent_second_v = jax.nn.sigmoid(latent_second_v_logits)

        state_first_scalar = self._aggregate_q_ensembles(state_first_v)
        state_second_scalar = self._aggregate_q_ensembles(state_second_v)
        latent_first_scalar = self._aggregate_q_ensembles(latent_first_v)
        latent_second_scalar = self._aggregate_q_ensembles(latent_second_v)

        cov_point_estimate = (state_first_scalar - latent_first_scalar) * (
            state_second_scalar - latent_second_scalar
        )

        dist = jax.lax.stop_gradient(jnp.log(target) / jnp.log(self.config['discount']))
        dist_weight = (1 / (1 + dist)) ** self.config['lam']
        q_loss = dist_weight * self.bce_loss(q_logits, jax.lax.stop_gradient(target))
        total_loss = q_loss.mean()

        info = {
            'total_loss': total_loss,
            'q_loss': q_loss.mean(),
            'q_mean': qs.mean(),
            'q_max': qs.max(),
            'q_min': qs.min(),
            'first_factor_mean': first_v.mean(),
            'second_factor_mean': second_v.mean(),
            'cov_point_estimate_mean': cov_point_estimate.mean(),
            'cov_point_estimate_min': cov_point_estimate.min(),
            'cov_point_estimate_max': cov_point_estimate.max(),
            'alt_bellman_backup': jnp.asarray(float(self.config['alt_bellman_backup']), dtype=jnp.float32),
            'alt_bellman_use_target_latent_values': jnp.asarray(
                float(self.config['alt_bellman_use_target_latent_values']), dtype=jnp.float32
            ),
        }
        if self._use_hybrid_state_midpoint():
            info['short_frac'] = self._hybrid_short_mask(batch).mean()
            info['q_latent_mean'] = jax.nn.sigmoid(q_latent_logits).mean()
            info['q_state_mean'] = q_state.mean()

        if self.config['oracle_distill']:
            oracle_q_logits = self.network.select('oracle_q')(
                batch['observations'],
                goals=batch['value_goals'],
                actions=z_mid,
                params=grad_params,
            )
            oracle_q = jax.nn.sigmoid(oracle_q_logits)
            oracle_distill_loss = self.bce_loss(oracle_q_logits, jax.lax.stop_gradient(qs)).mean()
            total_loss = total_loss + self.config['oracle_q_distill_coef'] * oracle_distill_loss
            info['total_loss'] = total_loss
            info['oracle_q_distill_loss'] = oracle_distill_loss
            info['oracle_q_mean'] = oracle_q.mean()
            info['oracle_q_max'] = oracle_q.max()
            info['oracle_q_min'] = oracle_q.min()

        return total_loss, info

    def value_loss(self, batch, grad_params):
        """In-trajectory value target from midpoint latent subgoals only."""
        goal_key = 'value_goal_observations' if self.config['oracle_distill'] else 'value_goals'
        v_logits = self.network.select('value')(
            batch['observations'],
            goals=batch[goal_key],
            params=grad_params,
        )
        vs = jax.nn.sigmoid(v_logits)

        z_mid = self._encode_subgoal(batch['value_midpoint_observations'])
        z_mid = jax.lax.stop_gradient(z_mid)
        q_latent_logits = self.network.select('target_q')(
            batch['observations'],
            goals=batch[goal_key],
            actions=z_mid,
        )
        q_logits = q_latent_logits
        q_state = None
        if self._use_hybrid_state_midpoint():
            q_state_logits = self.network.select('target_q_state')(
                batch['observations'],
                goals=batch[goal_key],
                actions=batch['value_midpoint_observations'],
            )
            short_mask = self._hybrid_short_mask(batch)
            q_logits = jnp.where(short_mask, q_state_logits, q_latent_logits)
            q_state = jax.nn.sigmoid(q_state_logits)
        q = jax.nn.sigmoid(q_logits)
        target = self._aggregate_q_ensembles(q)

        expectile_weight = jnp.where(
            target >= vs,
            self.config['expectile'],
            (1 - self.config['expectile']),
        )
        dist = jax.lax.stop_gradient(jnp.log(target) / jnp.log(self.config['discount']))
        dist_weight = (1 / (1 + dist)) ** self.config['lam']
        v_loss = expectile_weight * dist_weight * self.bce_loss(v_logits, jax.lax.stop_gradient(target))

        total_loss = v_loss.mean()

        predicted_steps = jnp.log(vs + 1e-8) / jnp.log(self.config['discount'])
        actual_steps = batch['value_offsets']
        relative_gap = (predicted_steps - actual_steps) / (actual_steps + 1e-8)

        info = {
            'total_loss': total_loss,
            'v_loss': v_loss.mean(),
            'v_mean': vs.mean(),
            'v_max': vs.max(),
            'v_min': vs.min(),
            'calibration_rel_gap_mean': relative_gap.mean(),
            'calibration_rel_gap_max': relative_gap.max(),
        }
        if self._use_hybrid_state_midpoint():
            info['short_frac'] = self._hybrid_short_mask(batch).mean()
            info['q_target_latent_mean'] = jax.nn.sigmoid(q_latent_logits).mean()
            info['q_target_state_mean'] = q_state.mean()
        return total_loss, info

    def value_sz_loss(self, batch, grad_params):
        """Distill V(s, z_mid) from V(s, s_mid) to test V(s, z) viability."""
        midpoint_goal_key = 'value_midpoint_observations' if self.config['oracle_distill'] else 'value_midpoint_goals'
        subgoal_grad_params = None if self.config['value_sz_stopgrad_encoder'] else grad_params
        z_mid = self._encode_subgoal(batch['value_midpoint_observations'], grad_params=subgoal_grad_params)

        v_teacher_logits = self.network.select('value')(
            batch['observations'],
            goals=batch[midpoint_goal_key],
            params=grad_params,
        )
        v_teacher = jax.nn.sigmoid(v_teacher_logits)
        v_teacher_target = self._aggregate_q_ensembles(v_teacher)
        v_teacher_metric = v_teacher_target

        v_sz_logits = self.network.select('value_sz')(
            batch['observations'],
            goals=z_mid,
            params=grad_params,
        )
        v_sz = jax.nn.sigmoid(v_sz_logits)
        v_sz_metric = self._aggregate_q_ensembles(v_sz)

        target = jax.lax.stop_gradient(v_teacher_target)
        distill_loss = self.bce_loss(v_sz_logits, target[None, ...]).mean()

        z_norms = jnp.linalg.norm(z_mid, axis=-1)
        return distill_loss, {
            'distill_loss': distill_loss,
            'v_teacher_mean': v_teacher_metric.mean(),
            'v_teacher_max': v_teacher_metric.max(),
            'v_teacher_min': v_teacher_metric.min(),
            'v_sz_mean': v_sz_metric.mean(),
            'v_sz_max': v_sz_metric.max(),
            'v_sz_min': v_sz_metric.min(),
            'vsz_abs_err': jnp.abs(v_sz_metric - v_teacher_metric).mean(),
            'z_norm_mean': z_norms.mean(),
            'z_norm_std': z_norms.std(),
        }

    def value_zg_loss(self, batch, grad_params):
        """Distill V(z_mid, g) from teacher V(s_mid, g)."""
        goal_key = 'value_goal_observations' if self.config['oracle_distill'] else 'value_goals'
        subgoal_grad_params = None if self.config['value_zg_stopgrad_encoder'] else grad_params
        z_mid = self._encode_subgoal(batch['value_midpoint_observations'], grad_params=subgoal_grad_params)

        v_teacher_logits = self.network.select('value')(
            batch['value_midpoint_observations'],
            goals=batch[goal_key],
            params=grad_params,
        )
        v_teacher = jax.nn.sigmoid(v_teacher_logits)
        v_teacher_target = self._aggregate_q_ensembles(v_teacher)
        v_teacher_metric = v_teacher_target

        v_zg_logits = self.network.select('value_zg')(
            z_mid,
            goals=batch[goal_key],
            params=grad_params,
        )
        v_zg = jax.nn.sigmoid(v_zg_logits)
        v_zg_metric = self._aggregate_q_ensembles(v_zg)

        target = jax.lax.stop_gradient(v_teacher_target)
        distill_loss = self.bce_loss(v_zg_logits, target[None, ...]).mean()

        z_norms = jnp.linalg.norm(z_mid, axis=-1)
        return distill_loss, {
            'distill_loss': distill_loss,
            'v_teacher_mean': v_teacher_metric.mean(),
            'v_teacher_max': v_teacher_metric.max(),
            'v_teacher_min': v_teacher_metric.min(),
            'v_zg_mean': v_zg_metric.mean(),
            'v_zg_max': v_zg_metric.max(),
            'v_zg_min': v_zg_metric.min(),
            'vzg_abs_err': jnp.abs(v_zg_metric - v_teacher_metric).mean(),
            'z_norm_mean': z_norms.mean(),
            'z_norm_std': z_norms.std(),
        }

    def q_short_loss(self, batch, grad_params):
        """Distill Q_short(s, g, a) from gamma * V(s', g) for FRS action selection."""
        q_short_goal_key = 'value_goals'
        target_goal_key = 'value_goal_observations' if self.config['oracle_distill'] else 'value_goals'

        v_next_logits = self.network.select('target_value')(
            batch['next_observations'],
            goals=batch[target_goal_key],
        )
        v_next = jax.nn.sigmoid(v_next_logits)
        v_next_min = jnp.minimum(v_next[0], v_next[1])
        target = self.config['discount'] * v_next_min

        q_short_logits = self.network.select('q_short')(
            batch['observations'],
            goals=batch[q_short_goal_key],
            actions=batch['actions'],
            params=grad_params,
        )
        q_short = jax.nn.sigmoid(q_short_logits)

        q_short_loss = self.bce_loss(q_short_logits, jax.lax.stop_gradient(target)).mean()

        return q_short_loss, {
            'q_short_loss': q_short_loss,
            'q_short_mean': q_short.mean(),
            'q_short_max': q_short.max(),
            'q_short_min': q_short.min(),
            'v_next_target_mean': target.mean(),
        }

    def actor_loss(self, batch, grad_params, rng=None):
        """Flat actor loss (same policy extraction interface as TRL)."""
        pe_info = self._get_pe_info()

        if self.config['pe_type'] == 'rpg':
            dist = self.network.select('actor')(
                batch['observations'],
                batch['actor_goals'],
                params=grad_params,
            )

            actor_goal_key = 'actor_goal_observations' if self.config['oracle_distill'] else 'actor_goals'
            v_next = self.network.select('value')(
                batch['next_observations'],
                goals=batch[actor_goal_key],
            )
            v = jnp.minimum(v_next[0], v_next[1])

            v_loss = -v.mean() / jax.lax.stop_gradient(jnp.abs(v).mean() + 1e-6)
            log_prob = dist.log_prob(batch['actions'])
            bc_loss = -(pe_info['alpha'] * log_prob).mean()

            actor_loss = v_loss + bc_loss
            return actor_loss, {
                'actor_loss': actor_loss,
                'v_loss': v_loss.mean(),
                'bc_loss': bc_loss,
                'v_mean': v.mean(),
                'v_abs_mean': jnp.abs(v).mean(),
                'bc_log_prob': log_prob.mean(),
                'mse': jnp.mean((dist.mode() - batch['actions']) ** 2),
                'std': jnp.mean(dist.scale_diag),
            }

        if self.config['pe_type'] == 'discrete':
            dist = self.network.select('actor')(
                batch['observations'],
                batch['actor_goals'],
                params=grad_params,
            )

            actor_goal_key = 'actor_goal_observations' if self.config['oracle_distill'] else 'actor_goals'
            v = self.network.select('value')(
                batch['next_observations'],
                goals=batch[actor_goal_key],
            ).mean(axis=0)
            v_loss = -v.mean()

            log_prob = dist.log_prob(batch['actions'])
            bc_loss = -(pe_info['alpha'] * log_prob).mean()
            actor_loss = v_loss + bc_loss

            return actor_loss, {
                'actor_loss': actor_loss,
                'v_loss': v_loss.mean(),
                'bc_loss': bc_loss,
                'v_mean': v.mean(),
                'v_abs_mean': jnp.abs(v).mean(),
                'bc_log_prob': log_prob.mean(),
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

        raise ValueError(f"Unsupported pe_type: {self.config['pe_type']}")

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        info = {}
        rng = rng if rng is not None else self.rng
        rng, value_rng, q_rng, value_sz_rng, value_zg_rng, q_short_rng, actor_rng = jax.random.split(rng, 7)

        del value_rng, q_rng, value_sz_rng, value_zg_rng, q_short_rng  # deterministic losses in this minimal impl.

        value_loss, value_info = self.value_loss(batch, grad_params)
        for k, v in value_info.items():
            info[f'value/{k}'] = v

        q_loss, q_info = self.q_loss(batch, grad_params)
        for k, v in q_info.items():
            info[f'q/{k}'] = v

        value_sz_loss, value_sz_info = self.value_sz_loss(batch, grad_params)
        for k, v in value_sz_info.items():
            info[f'value_sz/{k}'] = v

        value_zg_loss = 0.0
        if self.config['value_zg_coef'] > 0:
            value_zg_loss, value_zg_info = self.value_zg_loss(batch, grad_params)
            for k, v in value_zg_info.items():
                info[f'value_zg/{k}'] = v

        q_short_loss, q_short_info = self.q_short_loss(batch, grad_params)
        for k, v in q_short_info.items():
            info[f'q_short/{k}'] = v

        actor_loss, actor_info = self.actor_loss(batch, grad_params, rng=actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = (
            value_loss
            + q_loss
            + self.config['value_sz_coef'] * value_sz_loss
            + self.config['value_zg_coef'] * value_zg_loss
            + q_short_loss
            + actor_loss
        )
        return loss, info

    def target_update(self, network, module_name, tau=None):
        tau = self.config['tau'] if tau is None else tau
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * tau + tp * (1 - tau),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @jax.jit
    def update(self, batch):
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, 'value')
        self.target_update(new_network, 'value_sz')
        self.target_update(new_network, 'value_zg')
        self.target_update(new_network, 'q')
        self.target_update(new_network, 'q_state')
        if self.config['oracle_distill']:
            self.target_update(new_network, 'oracle_q')

        return self.replace(network=new_network, rng=new_rng), info

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

            q_short = self.network.select('q_short')(
                n_observations,
                goals=n_goals,
                actions=n_actions,
            )

            if len(observations.shape) == 2:
                actions = n_actions[jnp.argmax(q_short, axis=0), jnp.arange(observations.shape[0])]
            else:
                actions = n_actions[jnp.argmax(q_short)]

            return actions

        dist = self.network.select('actor')(observations, goals, temperature=temperature)
        actions = dist.sample(seed=seed)

        if self.config['pe_type'] != 'discrete':
            actions = jnp.clip(actions, -1, 1)

        return actions

    @classmethod
    def create(
        cls,
        seed,
        example_batch,
        config,
    ):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_observations = example_batch['observations']
        ex_actions = example_batch['actions']
        ex_goals = example_batch['actor_goals']
        ex_mid_obs = example_batch.get('value_midpoint_observations', ex_observations)
        ex_value_goals = ex_observations if config['oracle_distill'] else ex_goals
        ex_times = ex_actions[..., :1]
        action_dim = ex_actions.shape[-1]
        pe_info = cls._get_pe_info_from_config(config)

        ex_z = jnp.zeros((ex_observations.shape[0], config['z_dim']), dtype=ex_observations.dtype)

        value_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=2,
        )
        value_sz_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=2,
        )
        value_zg_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=2,
        )
        q_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=2,
        )
        q_state_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=2,
        )
        q_short_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=1,
        )
        subgoal_encoder_def = SubgoalEncoder(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            z_dim=config['z_dim'],
        )

        if config['pe_type'] == 'frs':
            actor_def = ActorVectorField(
                hidden_dims=config['actor_hidden_dims'],
                action_dim=action_dim,
                layer_norm=config['layer_norm'],
            )
            ex_actor_in = (ex_observations, ex_goals, ex_actions, ex_times)
        elif config['pe_type'] == 'discrete':
            actor_def = GCDiscreteActor(
                hidden_dims=config['actor_hidden_dims'],
                action_dim=config['pe_discrete']['action_ct'],
                layer_norm=config['layer_norm'],
            )
            ex_actor_in = (ex_observations, ex_goals, ex_actions)
        elif config['pe_type'] == 'rpg':
            actor_def = GCActor(
                hidden_dims=config['actor_hidden_dims'],
                action_dim=action_dim,
                layer_norm=config['layer_norm'],
                state_dependent_std=False,
                const_std=pe_info['const_std'],
            )
            ex_actor_in = (ex_observations, ex_goals, ex_actions)
        else:
            raise ValueError(f"Unsupported pe_type: {config['pe_type']}")

        network_info = dict(
            subgoal_encoder=(subgoal_encoder_def, (ex_mid_obs,)),
            value=(value_def, (ex_observations, ex_value_goals)),
            target_value=(copy.deepcopy(value_def), (ex_observations, ex_value_goals)),
            value_sz=(value_sz_def, (ex_observations, ex_z)),
            target_value_sz=(copy.deepcopy(value_sz_def), (ex_observations, ex_z)),
            value_zg=(value_zg_def, (ex_z, ex_value_goals)),
            target_value_zg=(copy.deepcopy(value_zg_def), (ex_z, ex_value_goals)),
            q=(q_def, (ex_observations, ex_value_goals, ex_z)),
            target_q=(copy.deepcopy(q_def), (ex_observations, ex_value_goals, ex_z)),
            q_state=(q_state_def, (ex_observations, ex_value_goals, ex_mid_obs)),
            target_q_state=(copy.deepcopy(q_state_def), (ex_observations, ex_value_goals, ex_mid_obs)),
            q_short=(q_short_def, (ex_observations, ex_goals, ex_actions)),
            actor=(actor_def, ex_actor_in),
        )
        if config['oracle_distill']:
            network_info.update(
                dict(
                    oracle_q=(copy.deepcopy(q_def), (ex_observations, ex_goals, ex_z)),
                    target_oracle_q=(copy.deepcopy(q_def), (ex_observations, ex_goals, ex_z)),
                )
            )

        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']

        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network_params
        params['modules_target_value'] = params['modules_value']
        params['modules_target_value_sz'] = params['modules_value_sz']
        params['modules_target_value_zg'] = params['modules_value_zg']
        params['modules_target_q'] = params['modules_q']
        params['modules_target_q_state'] = params['modules_q_state']
        if config['oracle_distill']:
            params['modules_target_oracle_q'] = params['modules_oracle_q']

        config['action_dim'] = action_dim
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='latent_trl',
            lr=3e-4,
            batch_size=1024,
            actor_hidden_dims=(1024,) * 4,
            value_hidden_dims=(1024,) * 4,
            layer_norm=True,
            discount=0.999,
            tau=0.005,
            lam=0.0,
            expectile=0.7,
            oracle_distill=False,
            oracle_q_distill_coef=1.0,
            q_agg='min',
            z_dim=8,
            value_sz_coef=1.0,
            value_sz_stopgrad_encoder=True,
            value_zg_coef=0.0,
            value_zg_stopgrad_encoder=True,
            alt_bellman_backup=False,
            alt_bellman_use_target_latent_values=True,
            hybrid_state_midpoint_max_offset=-1,
            pe_type='frs',  # frs, rpg, discrete
            frs=ml_collections.ConfigDict(dict(flow_steps=10, num_samples=32)),
            rpg=ml_collections.ConfigDict(dict(alpha=0.03, const_std=True)),
            pe_discrete=ml_collections.ConfigDict(dict(alpha=0.03, action_ct=0)),
            discrete=False,
            dataset_class='GCDataset',
            value_p_curgoal=0.0,
            value_p_trajgoal=1.0,
            value_p_randomgoal=0.0,
            value_geom_sample=True,
            actor_p_curgoal=0.0,
            actor_p_trajgoal=0.5,
            actor_p_randomgoal=0.5,
            actor_geom_sample=True,
            gc_negative=True,
            p_aug=0.0,
            frame_stack=ml_collections.config_dict.placeholder(int),
        )
    )
    return config
