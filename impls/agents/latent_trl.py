import copy
from typing import Any, Sequence

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import optax
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.flows import imf_cfg_loss, imf_loss, imf_one_shot_sample
from utils.networks import (
    ActorVectorField,
    ActorVectorFieldDualHead,
    GCActor,
    GCDiscreteActor,
    GCValue,
    MLP,
    ResMLPActorVectorFieldDualHead,
    TimeConditioner,
)

class SubgoalEncoder(nn.Module):
    """Subgoal encoder: s_j -> z_j."""

    hidden_dims: Sequence[int]
    layer_norm: bool
    z_dim: int

    @nn.compact
    def __call__(self, x):
        return MLP((*self.hidden_dims, self.z_dim), activate_final=False, layer_norm=self.layer_norm)(x)


class LatentTRLAgent(flax.struct.PyTreeNode):
    """Latent TRL agent with action-free V(s, g) and latent subgoal Q(s, z, g)."""

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

    def _subgoal_flow_type(self):
        flow_type = self.config.get('subgoal_flow_type', 'auto')
        if flow_type in (None, 'auto'):
            return 'frs' if self.config['pe_type'] == 'frs' else 'dist'
        return flow_type

    def _encode_subgoal(self, midpoint_observations, grad_params=None):
        if grad_params is None:
            return self.network.select('subgoal_encoder')(midpoint_observations)
        return self.network.select('subgoal_encoder')(midpoint_observations, params=grad_params)

    def _sample_subgoals(self, observations, goals, rng, num_samples=None):
        """Sample latent subgoals from the high-level policy."""
        flow_type = self._subgoal_flow_type()
        if num_samples is None:
            num_samples = self.config['subgoal_num_samples']

        if flow_type == 'frs':
            n_observations = jnp.repeat(jnp.expand_dims(observations, 0), num_samples, axis=0)
            n_goals = jnp.repeat(jnp.expand_dims(goals, 0), num_samples, axis=0)

            z = jax.random.normal(
                rng,
                (
                    num_samples,
                    *observations.shape[:-1],
                    self.config['z_dim'],
                ),
            )
            for i in range(self.config['subgoal_flow_steps']):
                t = jnp.full(
                    (num_samples, *observations.shape[:-1], 1),
                    i / self.config['subgoal_flow_steps'],
                )
                vels = self.network.select('subgoal_actor')(n_observations, n_goals, z, t)
                z = z + vels / self.config['subgoal_flow_steps']
            return z

        if flow_type == 'imf':
            n_observations = jnp.repeat(jnp.expand_dims(observations, 0), num_samples, axis=0)
            n_goals = jnp.repeat(jnp.expand_dims(goals, 0), num_samples, axis=0)
            sample_shape = (
                num_samples,
                *observations.shape[:-1],
                self.config['z_dim'],
            )
            imf_cfg = self.config.get('subgoal_imf', None)
            cfg_scale = 1.0 if imf_cfg is None else imf_cfg.get('cfg_scale', 1.0)

            def vf(z, times):
                return self.network.select('subgoal_actor')(n_observations, n_goals, z, times)

            return imf_one_shot_sample(rng, sample_shape, vf, w=cfg_scale)

        if flow_type == 'dist':
            dist = self.network.select('subgoal_actor')(observations, goals)
            return dist.sample(seed=rng, sample_shape=(num_samples,))
        raise ValueError(f'Unsupported subgoal_flow_type: {flow_type}')

    def _expectile_agg(self, samples, tau, num_iters):
        """Compute expectile over samples with a small fixed-point iteration."""
        mu = jnp.mean(samples, axis=0)

        def body(_, cur):
            weights = jnp.where(samples > cur, tau, 1 - tau)
            return jnp.sum(weights * samples, axis=0) / jnp.sum(weights, axis=0)

        return jax.lax.fori_loop(0, num_iters, body, mu)

    @staticmethod
    def bce_loss(pred_logit, target):
        log_pred = jax.nn.log_sigmoid(pred_logit)
        log_not_pred = jax.nn.log_sigmoid(-pred_logit)
        loss = -(log_pred * target + log_not_pred * (1 - target))
        return loss

    def q_short_loss(self, batch, grad_params):
        """Distill Q_short(s, g, a) from V(s', g).

        Q_short approximates the 1-step Q value: Q(s, g, a) ≈ γ * V(s', g)
        This allows rejection sampling at inference without a dynamics model.
        """
        # For oracle distill, use oracle goals for q_short, but no oracle teacher for value/Q.
        q_short_goal_key = 'value_goals'
        target_goal_key = 'value_goal_observations' if self.config['oracle_distill'] else 'value_goals'
        target_value_module = 'target_value'

        # Target: γ * V(s', g) from the target value network
        v_next_logits = self.network.select(target_value_module)(
            batch['next_observations'],
            goals=batch[target_goal_key],
        )
        v_next = jax.nn.sigmoid(v_next_logits)
        # Take min over ensembles for conservative target
        v_next_min = jnp.minimum(v_next[0], v_next[1])
        target = self.config['discount'] * v_next_min

        # Predict Q_short(s, g, a)
        q_short_logits = self.network.select('q_short')(
            batch['observations'],
            goals=batch[q_short_goal_key],
            actions=batch['actions'],
            params=grad_params,
        )
        q_short = jax.nn.sigmoid(q_short_logits)

        # MSE loss (or BCE if we want to stay in logit space)
        q_short_loss = self.bce_loss(q_short_logits, jax.lax.stop_gradient(target)).mean()

        return q_short_loss, {
            'q_short_loss': q_short_loss,
            'q_short_mean': q_short.mean(),
            'q_short_max': q_short.max(),
            'q_short_min': q_short.min(),
            'v_next_target_mean': target.mean(),
        }

    def q_loss(self, batch, grad_params):
        """Compute Q(s, z_j, g) loss as BCE to the triangle inequality target."""
        goal_key = 'value_goal_observations' if self.config['oracle_distill'] else 'value_goals'
        midpoint_goal_key = 'value_midpoint_observations' if self.config['oracle_distill'] else 'value_midpoint_goals'

        z_mid = self._encode_subgoal(batch['value_midpoint_observations'], grad_params=grad_params)
        q_logits = self.network.select('q')(
            batch['observations'],
            goals=batch[goal_key],
            actions=z_mid,
            params=grad_params,
        )
        qs = jax.nn.sigmoid(q_logits)

        first_v_logits = self.network.select('target_value')(
            batch['observations'],
            goals=batch[midpoint_goal_key],
        )
        first_v = jnp.where(
            (batch['value_midpoint_offsets'] <= 1)[None, ...],
            self.config['discount'] ** batch['value_midpoint_offsets'][None, ...],
            jax.nn.sigmoid(first_v_logits),
        )

        second_v_logits = self.network.select('target_value')(
            batch['value_midpoint_observations'],
            goals=batch[goal_key],
        )
        second_offset = batch['value_offsets'][None, ...] - batch['value_midpoint_offsets']
        second_v = jnp.where(
            (second_offset <= 1)[None, ...],
            self.config['discount'] ** second_offset[None, ...],
            jax.nn.sigmoid(second_v_logits),
        )
        target = first_v * second_v

        dist = jax.lax.stop_gradient(jnp.log(target) / jnp.log(self.config['discount']))
        dist_weight = (1 / (1 + dist)) ** self.config['lam']
        q_loss = dist_weight * self.bce_loss(q_logits, jax.lax.stop_gradient(target))
        total_loss = q_loss.mean()

        return total_loss, {
            'total_loss': total_loss,
            'q_loss': q_loss.mean(),
            'q_mean': qs.mean(),
            'q_max': qs.max(),
            'q_min': qs.min(),
        }

    def value_loss(self, batch, grad_params, rng=None):
        """Compute the value loss using max/expectile over Q(s, z, g)."""
        goal_key = 'value_goal_observations' if self.config['oracle_distill'] else 'value_goals'
        v_logits = self.network.select('value')(
            batch['observations'],
            goals=batch[goal_key],
            params=grad_params,
        )
        vs = jax.nn.sigmoid(v_logits)

        def compute_in_traj_target():
            z_mid = self._encode_subgoal(batch['value_midpoint_observations'])
            z_mid = jax.lax.stop_gradient(z_mid)
            q_logits = self.network.select('target_q')(
                batch['observations'],
                goals=batch[goal_key],
                actions=z_mid,
            )
            q = jax.nn.sigmoid(q_logits)
            if self.config['q_agg'] == 'min':
                return jnp.minimum(q[0], q[1])
            if self.config['q_agg'] == 'mean':
                return q.mean(axis=0)
            return q

        def compute_generative_q_samples():
            """Return per-sample Q values (shape [N, batch]) and the aggregated target."""
            if rng is None:
                raise ValueError('rng is required for generative value maximization')
            goal_key = 'value_goal_observations' if self.config['oracle_distill'] else 'value_goals'
            z_samples = self._sample_subgoals(
                batch['observations'],
                batch[goal_key],
                rng,
                num_samples=self.config['subgoal_num_samples'],
            )
            n_observations = jnp.repeat(
                jnp.expand_dims(batch['observations'], 0),
                self.config['subgoal_num_samples'],
                axis=0,
            )
            n_goals = jnp.repeat(
                jnp.expand_dims(batch[goal_key], 0), self.config['subgoal_num_samples'], axis=0
            )
            q_logits = self.network.select('target_q')(
                n_observations,
                goals=n_goals,
                actions=z_samples,
            )
            q = jax.nn.sigmoid(q_logits)
            if self.config['q_agg'] == 'min':
                q = jnp.minimum(q[0], q[1])
            elif self.config['q_agg'] == 'mean':
                q = q.mean(axis=0)
            # q shape: [N, batch]
            if self.config['value_maximization_agg'] == 'max':
                target = jnp.max(q, axis=0)
            elif self.config['value_maximization_agg'] == 'expectile':
                target = self._expectile_agg(
                    q,
                    self.config['value_sample_expectile'],
                    self.config['value_sample_expectile_iters'],
                )
            else:
                target = q
            # Optionally take hard max with in-trajectory as a floor
            if self.config.get('value_maximization_intraj_floor', False):
                q_intraj = compute_in_traj_target()  # shape: [batch]
                target = jnp.maximum(target, q_intraj)
            return q, target

        def compute_generative_target():
            _, target = compute_generative_q_samples()
            return target

        def compute_loss_from_target(target):
            expectile_weight = jnp.where(
                target >= vs,
                self.config['expectile'],
                (1 - self.config['expectile']),
            )
            dist = jax.lax.stop_gradient(jnp.log(target) / jnp.log(self.config['discount']))
            dist_weight = (1 / (1 + dist)) ** self.config['lam']
            return expectile_weight * dist_weight * self.bce_loss(v_logits, target)

        info = {}

        if self.config['value_maximization'] == 'in-trajectory':
            v_loss = compute_loss_from_target(compute_in_traj_target())
        elif self.config['value_maximization'] == 'generative':
            interval = max(int(self.config['value_maximization_interval']), 1)
            do_update = (self.network.step % interval) == 0
            ramp_steps = int(self.config.get('value_maximization_ramp_steps', 0))

            # Compute generative target and log Q sample diagnostics.
            q_samples, gen_target = compute_generative_q_samples()
            # q_samples shape: [N, batch]. Reduce to scalars for logging.
            info['gen_q_mean'] = jnp.mean(q_samples)
            info['gen_q_std'] = jnp.mean(jnp.std(q_samples, axis=0))
            info['gen_q_min'] = jnp.mean(jnp.min(q_samples, axis=0))
            info['gen_q_max'] = jnp.mean(jnp.max(q_samples, axis=0))
            info['gen_target_mean'] = jnp.mean(gen_target)

            gen_loss = compute_loss_from_target(gen_target)

            if ramp_steps > 0:
                # Linear interpolation from in-trajectory to generative.
                alpha = jnp.clip(self.network.step / ramp_steps, 0.0, 1.0)
                traj_loss = compute_loss_from_target(compute_in_traj_target())
                v_loss = (1 - alpha) * traj_loss + alpha * gen_loss
                info['ramp_alpha'] = alpha
            elif self.config['value_maximization_fallback'] == 'in-trajectory':
                traj_loss = compute_loss_from_target(compute_in_traj_target())
                weight = self.config['value_maximization_weight']
                v_loss = jax.lax.cond(
                    do_update,
                    lambda _: weight * gen_loss + (1 - weight) * traj_loss,
                    lambda _: traj_loss,
                    operand=None,
                )
            else:
                v_loss = jax.lax.cond(
                    do_update,
                    lambda _: gen_loss,
                    lambda _: jnp.zeros_like(vs),
                    operand=None,
                )

        total_loss = v_loss.mean()

        # Calibration: relative error between predicted and actual steps
        # predicted_steps = log(V) / log(γ), actual_steps = k
        # relative_gap = (predicted - actual) / actual
        predicted_steps = jnp.log(vs + 1e-8) / jnp.log(self.config['discount'])
        actual_steps = batch['value_offsets']
        relative_gap = (predicted_steps - actual_steps) / (actual_steps + 1e-8)
        info['calibration_rel_gap_mean'] = relative_gap.mean()
        info['calibration_rel_gap_max'] = relative_gap.max()

        return total_loss, {
            'total_loss': total_loss,
            'v_loss': v_loss.mean(),
            'v_mean': vs.mean(),
            'v_max': vs.max(),
            'v_min': vs.min(),
            **info,
        }

    def subgoal_actor_loss(self, batch, grad_params, rng):
        """High-level subgoal proposer loss (flow-matching BC + optional value maximization)."""
        rng, flow_rng, sample_rng = jax.random.split(rng, 3)
        z_target = self._encode_subgoal(batch['value_midpoint_observations'])
        z_target = jax.lax.stop_gradient(z_target)

        goal_key = 'value_goal_observations' if self.config['oracle_distill'] else 'value_goals'
        flow_type = self._subgoal_flow_type()
        if flow_type == 'dist':
            dist = self.network.select('subgoal_actor')(
                batch['observations'], batch[goal_key], params=grad_params
            )
            bc_loss = -dist.log_prob(z_target).mean()
        elif flow_type == 'frs':
            batch_size, z_dim = z_target.shape
            x_rng, t_rng = jax.random.split(flow_rng, 2)
            x_0 = jax.random.normal(x_rng, (batch_size, z_dim))
            x_1 = z_target
            t = jax.random.uniform(t_rng, (batch_size, 1))
            x_t = (1 - t) * x_0 + t * x_1
            y = x_1 - x_0

            pred = self.network.select('subgoal_actor')(
                batch['observations'], batch[goal_key], x_t, t, params=grad_params
            )
            bc_loss = jnp.mean((pred - y) ** 2)
        elif flow_type == 'imf':
            imf_cfg = self.config.get('subgoal_imf', None)
            r_equals_t_prob = 0.0 if imf_cfg is None else imf_cfg.get('r_equals_t_prob', 0.0)
            cfg_w_min = 1.0 if imf_cfg is None else imf_cfg.get('cfg_w_min', 1.0)
            cfg_w_max = 1.0 if imf_cfg is None else imf_cfg.get('cfg_w_max', 1.0)
            cfg_w_power = 1.0 if imf_cfg is None else imf_cfg.get('cfg_w_power', 1.0)
            cfg_beta = None if imf_cfg is None else imf_cfg.get('cfg_beta', None)
            class_dropout_prob = 0.1 if imf_cfg is None else imf_cfg.get('class_dropout_prob', 0.1)
            logit_mean = -0.4 if imf_cfg is None else imf_cfg.get('logit_mean', -0.4)
            logit_std = 1.0 if imf_cfg is None else imf_cfg.get('logit_std', 1.0)
            adaptive_loss_eps = 0.01 if imf_cfg is None else imf_cfg.get('adaptive_loss_eps', 0.01)
            adaptive_loss_p = 1.0 if imf_cfg is None else imf_cfg.get('adaptive_loss_p', 1.0)
            use_cfg = cfg_w_max > cfg_w_min

            goals = batch[goal_key]
            zero_goals = jnp.zeros_like(goals)

            def cond_fn(z_in, times_in):
                return self.network.select('subgoal_actor')(
                    batch['observations'],
                    goals,
                    z_in,
                    times_in,
                    params=grad_params,
                )

            def uncond_fn(z_in, times_in):
                return self.network.select('subgoal_actor')(
                    batch['observations'],
                    zero_goals,
                    z_in,
                    times_in,
                    params=grad_params,
                )

            if use_cfg:
                bc_loss, _ = imf_cfg_loss(
                    flow_rng,
                    z_target,
                    cond_fn,
                    uncond_fn,
                    r_equals_t_prob=r_equals_t_prob,
                    w_min=cfg_w_min,
                    w_max=cfg_w_max,
                    w_power=cfg_w_power,
                    cfg_beta=cfg_beta,
                    class_dropout_prob=class_dropout_prob,
                    logit_mean=logit_mean,
                    logit_std=logit_std,
                    adaptive_loss_eps=adaptive_loss_eps,
                    adaptive_loss_p=adaptive_loss_p,
                )
            else:
                bc_loss, _ = imf_loss(
                    flow_rng,
                    z_target,
                    cond_fn,
                    r_equals_t_prob=r_equals_t_prob,
                    w=1.0,
                    logit_mean=logit_mean,
                    logit_std=logit_std,
                    adaptive_loss_eps=adaptive_loss_eps,
                    adaptive_loss_p=adaptive_loss_p,
                )
        else:
            raise ValueError(f'Unsupported subgoal_flow_type: {flow_type}')

        total_loss = self.config['subgoal_actor_bc_coef'] * bc_loss

        if self.config['subgoal_actor_value_coef'] > 0:
            goal_key = 'value_goal_observations' if self.config['oracle_distill'] else 'value_goals'
            z_samples = self._sample_subgoals(
                batch['observations'],
                batch[goal_key],
                sample_rng,
                num_samples=self.config['subgoal_num_samples'],
            )
            n_observations = jnp.repeat(
                jnp.expand_dims(batch['observations'], 0),
                self.config['subgoal_num_samples'],
                axis=0,
            )
            n_goals = jnp.repeat(
                jnp.expand_dims(batch[goal_key], 0),
                self.config['subgoal_num_samples'],
                axis=0,
            )
            q_logits = self.network.select('target_q')(
                n_observations,
                goals=n_goals,
                actions=z_samples,
            )
            q = jax.nn.sigmoid(q_logits)
            if self.config['q_agg'] == 'min':
                q = jnp.minimum(q[0], q[1])
            elif self.config['q_agg'] == 'mean':
                q = q.mean(axis=0)
            if self.config['value_maximization_agg'] == 'max':
                q_best = jnp.max(q, axis=0)
            elif self.config['value_maximization_agg'] == 'expectile':
                q_best = self._expectile_agg(
                    q,
                    self.config['value_sample_expectile'],
                    self.config['value_sample_expectile_iters'],
                )
            value_loss = -q_best.mean()
            total_loss = total_loss + self.config['subgoal_actor_value_coef'] * value_loss

        info = {
            'actor_loss': total_loss,
            'bc_loss': bc_loss,
        }
        return total_loss, info

    def _low_actor_goal_conditioning(self):
        if 'low_actor_goal_conditioning' in self.config:
            return self.config['low_actor_goal_conditioning']
        return None

    def _use_high_policy_inference(self):
        if 'low_actor_goal_conditioning' in self.config:
            return self.config['low_actor_goal_conditioning'] in ('latent', 'both')
        return self.config['use_high_policy_inference']

    def _low_actor_goals(self, z, goals, true_goals=None):
        """Build low actor goal input based on conditioning mode."""
        conditioning = self._low_actor_goal_conditioning()
        if conditioning is None:
            # Backward compatibility with boolean flags.
            parts = [z]
            if self.config.get('low_actor_goal_conditioned', False):
                parts.append(goals)
            if self.config.get('low_actor_true_goal_conditioned', False) and true_goals is not None:
                parts.append(true_goals)
            if len(parts) == 1:
                return z
            return jnp.concatenate(parts, axis=-1)
        if conditioning == 'latent':
            return z
        if conditioning == 'actual':
            if true_goals is not None:
                return true_goals
            if goals is not None:
                return goals
            return z
        # both: concatenate latent z with true goals when available, otherwise with goals.
        if true_goals is not None:
            return jnp.concatenate([z, true_goals], axis=-1)
        if goals is not None:
            return jnp.concatenate([z, goals], axis=-1)
        return z

    def low_actor_loss(self, batch, grad_params, rng):
        """Low-level actor loss conditioned on latent subgoals."""
        z_target = self._encode_subgoal(batch['value_midpoint_observations'])
        z_target = jax.lax.stop_gradient(z_target)

        goal_key = 'value_goal_observations' if self.config['oracle_distill'] else 'value_goals'
        z_goals = self._low_actor_goals(z_target, batch[goal_key], batch.get('actor_goals'))

        if self.config['pe_type'] == 'frs':
            batch_size, action_dim = batch['actions'].shape
            x_rng, t_rng = jax.random.split(rng, 2)
            x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
            x_1 = batch['actions']
            t = jax.random.uniform(t_rng, (batch_size, 1))
            x_t = (1 - t) * x_0 + t * x_1
            y = x_1 - x_0

            pred = self.network.select('low_actor')(
                batch['observations'], z_goals, x_t, t, params=grad_params
            )
            bc_loss = jnp.mean((pred - y) ** 2)
        else:
            dist = self.network.select('low_actor')(batch['observations'], z_goals, params=grad_params)
            bc_loss = -dist.log_prob(batch['actions']).mean()

        total_loss = self.config['low_actor_bc_coef'] * bc_loss
        return total_loss, {
            'actor_loss': total_loss,
            'bc_loss': bc_loss,
        }

    def actor_loss(self, batch, grad_params, rng=None):
        """Compute the actor loss."""
        pe_info = self._get_pe_info()
        conditioning = self._low_actor_goal_conditioning()

        if conditioning is not None:
            rng = rng if rng is not None else self.rng
            rng, flow_rng = jax.random.split(rng, 2)
            z_target = self._encode_subgoal(batch['value_midpoint_observations'])
            z_target = jax.lax.stop_gradient(z_target)
            goal_key = 'value_goal_observations' if self.config['oracle_distill'] else 'value_goals'
            z_goals = self._low_actor_goals(z_target, batch[goal_key], batch.get('actor_goals'))

            if self.config['pe_type'] == 'frs':
                batch_size, action_dim = batch['actions'].shape
                x_rng, t_rng = jax.random.split(flow_rng, 2)
                x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
                x_1 = batch['actions']
                t = jax.random.uniform(t_rng, (batch_size, 1))
                x_t = (1 - t) * x_0 + t * x_1
                y = x_1 - x_0

                pred = self.network.select('low_actor')(
                    batch['observations'], z_goals, x_t, t, params=grad_params
                )
                bc_loss = jnp.mean((pred - y) ** 2)
            else:
                dist = self.network.select('low_actor')(batch['observations'], z_goals, params=grad_params)
                bc_loss = -dist.log_prob(batch['actions']).mean()

            actor_loss = self.config['low_actor_bc_coef'] * bc_loss
            return actor_loss, {
                'actor_loss': actor_loss,
                'bc_loss': bc_loss,
            }

        actor_module = 'low_actor' if conditioning == 'actual' else 'actor'

        if self.config['pe_type'] == 'rpg':
            dist = self.network.select(actor_module)(
                batch['observations'], batch['actor_goals'], params=grad_params
            )
            if pe_info['const_std']:
                sampled_actions = jnp.clip(dist.mode(), -1, 1)
            else:
                sampled_actions = jnp.clip(dist.sample(seed=rng), -1, 1)

            # Use V(s', g) where s' is approximated by next_observations
            # For advantage: A = V(s', g) - V(s, g)
            actor_goal_key = 'actor_goal_observations' if self.config['oracle_distill'] else 'actor_goals'
            v_next = self.network.select('value')(
                batch['next_observations'], batch[actor_goal_key]
            )
            v_curr = self.network.select('value')(
                batch['observations'], batch[actor_goal_key]
            )
            # Take min over ensembles
            v = jnp.minimum(v_next[0], v_next[1])

            # Normalize V values by the absolute mean to make the loss scale invariant.
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

        elif self.config['pe_type'] == 'discrete':
            dist = self.network.select(actor_module)(
                batch['observations'], batch['actor_goals'], params=grad_params
            )

            # For discrete: compute V for each possible next state
            # Since we don't have dynamics, use V(s', g) from data
            actor_goal_key = 'actor_goal_observations' if self.config['oracle_distill'] else 'actor_goals'
            v = self.network.select('value')(
                batch['next_observations'], batch[actor_goal_key]
            ).mean(axis=0)  # Average over ensembles

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

        elif self.config['pe_type'] == 'frs':
            batch_size, action_dim = batch['actions'].shape
            x_rng, t_rng = jax.random.split(rng, 2)

            x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
            x_1 = batch['actions']
            t = jax.random.uniform(t_rng, (batch_size, 1))
            x_t = (1 - t) * x_0 + t * x_1
            y = x_1 - x_0

            pred = self.network.select(actor_module)(
                batch['observations'], batch['actor_goals'], x_t, t, params=grad_params
            )

            actor_loss = jnp.mean((pred - y) ** 2)

            actor_info = {
                'actor_loss': actor_loss,
            }

            return actor_loss, actor_info

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        info = {}
        rng = rng if rng is not None else self.rng

        rng, value_rng, actor_rng, subgoal_actor_rng, low_actor_rng = jax.random.split(rng, 5)

        value_loss, value_info = self.value_loss(batch, grad_params, rng=value_rng)
        for k, v in value_info.items():
            info[f'value/{k}'] = v

        q_loss, q_info = self.q_loss(batch, grad_params)
        for k, v in q_info.items():
            info[f'q/{k}'] = v

        q_short_loss, q_short_info = self.q_short_loss(batch, grad_params)
        for k, v in q_short_info.items():
            info[f'q_short/{k}'] = v

        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        subgoal_actor_loss, subgoal_actor_info = self.subgoal_actor_loss(batch, grad_params, subgoal_actor_rng)
        for k, v in subgoal_actor_info.items():
            info[f'subgoal_actor/{k}'] = v

        if self._low_actor_goal_conditioning() is not None:
            low_actor_loss = jnp.zeros(())
        else:
            low_actor_loss, low_actor_info = self.low_actor_loss(batch, grad_params, low_actor_rng)
            for k, v in low_actor_info.items():
                info[f'low_actor/{k}'] = v

        loss = value_loss + q_loss + q_short_loss + actor_loss + subgoal_actor_loss + low_actor_loss
        return loss, info

    def target_update(self, network, module_name):
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
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
        self.target_update(new_network, 'q')

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

        if self._use_high_policy_inference():
            high_seed, low_seed = jax.random.split(seed)
            z = self._sample_subgoals(
                observations,
                goals,
                high_seed,
                num_samples=self.config['subgoal_num_samples'],
            )
            if len(observations.shape) == 2:
                z = z[0, jnp.arange(observations.shape[0])]
            else:
                z = z[0]

            z_goals = self._low_actor_goals(z, goals, goals)

            if self.config['pe_type'] == 'frs':
                actions = jax.random.normal(low_seed, (*observations.shape[:-1], self.config['action_dim']))
                for i in range(pe_info['flow_steps']):
                    t = jnp.full((*observations.shape[:-1], 1), i / pe_info['flow_steps'])
                    vels = self.network.select('low_actor')(observations, z_goals, actions, t)
                    actions = actions + vels / pe_info['flow_steps']
                actions = jnp.clip(actions, -1, 1)
            else:
                dist = self.network.select('low_actor')(observations, z_goals, temperature=temperature)
                actions = dist.sample(seed=low_seed)
                if self.config['pe_type'] != 'discrete':
                    actions = jnp.clip(actions, -1, 1)
            return actions

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
                vels = self.network.select('low_actor')(n_observations, n_goals, n_actions, t)
                n_actions = n_actions + vels / pe_info['flow_steps']
            n_actions = jnp.clip(n_actions, -1, 1)

            q_short = self.network.select('q_short')(n_observations, goals=n_goals, actions=n_actions)

            if len(observations.shape) == 2:
                actions = n_actions[jnp.argmax(q_short, axis=0), jnp.arange(observations.shape[0])]
            else:
                actions = n_actions[jnp.argmax(q_short)]

            return actions

        else:
            dist = self.network.select('low_actor')(observations, goals, temperature=temperature)
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
        subgoal_flow_type = config.get('subgoal_flow_type', 'auto')
        if subgoal_flow_type in (None, 'auto'):
            subgoal_flow_type = 'frs' if config['pe_type'] == 'frs' else 'dist'
        ex_z = jnp.zeros((ex_observations.shape[0], config['z_dim']), dtype=ex_observations.dtype)

        # Build example low actor goal input based on conditioning.
        conditioning = config.get('low_actor_goal_conditioning')
        if conditioning is None:
            # Backward compatibility with boolean flags.
            ex_low_actor_goal_parts = [ex_z]
            if config.get('low_actor_goal_conditioned', False):
                ex_low_actor_goal_parts.append(ex_value_goals)
            if config.get('low_actor_true_goal_conditioned', False):
                ex_low_actor_goal_parts.append(ex_goals)
            ex_low_actor_z = (
                jnp.concatenate(ex_low_actor_goal_parts, axis=-1)
                if len(ex_low_actor_goal_parts) > 1
                else ex_z
            )
        elif conditioning == 'latent':
            ex_low_actor_z = ex_z
        elif conditioning == 'actual':
            ex_low_actor_z = ex_goals
        else:
            ex_low_actor_z = jnp.concatenate([ex_z, ex_goals], axis=-1)

        # Action-free value function V(s, g)
        value_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=2,
        )
        q_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=2,
        )
        subgoal_encoder_def = SubgoalEncoder(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            z_dim=config['z_dim'],
        )
        # Short-horizon Q distilled from V(s', g) for rejection sampling
        q_short_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=1,  # Single network, no ensemble needed for distillation target
        )

        if config['pe_type'] == 'frs':
            actor_def = ActorVectorField(
                hidden_dims=config['actor_hidden_dims'],
                action_dim=action_dim,
                layer_norm=config['layer_norm'],
            )
            ex_actor_in = (ex_observations, ex_goals, ex_actions, ex_times)
            low_actor_def = ActorVectorField(
                hidden_dims=config['high_actor_hidden_dims'],
                action_dim=action_dim,
                layer_norm=config['layer_norm'],
            )
            ex_low_actor_in = (ex_observations, ex_low_actor_z, ex_actions, ex_times)
        elif config['pe_type'] == 'discrete':
            actor_def = GCDiscreteActor(
                hidden_dims=config['actor_hidden_dims'],
                action_dim=config['pe_discrete']['action_ct'],
                layer_norm=config['layer_norm'],
            )
            ex_actor_in = (ex_observations, ex_goals, ex_actions)
            low_actor_def = GCDiscreteActor(
                hidden_dims=config['high_actor_hidden_dims'],
                action_dim=config['pe_discrete']['action_ct'],
                layer_norm=config['layer_norm'],
            )
            ex_low_actor_in = (ex_observations, ex_low_actor_z, ex_actions)
        else:
            actor_def = GCActor(
                hidden_dims=config['actor_hidden_dims'],
                action_dim=action_dim,
                layer_norm=config['layer_norm'],
                state_dependent_std=False,
                const_std=pe_info['const_std'],
            )
            ex_actor_in = (ex_observations, ex_goals, ex_actions)
            low_actor_def = GCActor(
                hidden_dims=config['high_actor_hidden_dims'],
                action_dim=action_dim,
                layer_norm=config['layer_norm'],
                state_dependent_std=False,
                const_std=pe_info['const_std'],
            )
            ex_low_actor_in = (ex_observations, ex_low_actor_z, ex_actions)

        if subgoal_flow_type in ('frs', 'imf'):
            time_encoder = None
            if subgoal_flow_type == 'imf':
                imf_cfg = config.get('subgoal_imf', {})
                time_encoder = TimeConditioner(
                    num_freqs=imf_cfg.get('time_embed_num_freqs', 32),
                    hidden_dim=imf_cfg.get('time_embed_hidden_dim', 128),
                    out_dim=imf_cfg.get('time_embed_dim', 128),
                    max_period=imf_cfg.get('time_embed_max_period', 10000.0),
                    scale=imf_cfg.get('time_embed_scale', 1.0),
                    layer_norm=config['layer_norm'],
                )
            if subgoal_flow_type == 'imf':
                use_resmlp = imf_cfg.get('use_resmlp', False)
                if use_resmlp:
                    resmlp_hidden_dim = imf_cfg.get('resmlp_hidden_dim', 256)
                    resmlp_num_blocks = imf_cfg.get('resmlp_num_blocks', 8)
                    subgoal_actor_def = ResMLPActorVectorFieldDualHead(
                        hidden_dim=resmlp_hidden_dim,
                        num_blocks=resmlp_num_blocks,
                        action_dim=config['z_dim'],
                        time_encoder=time_encoder,
                    )
                else:
                    head_hidden_dims = imf_cfg.get('head_hidden_dims', None)
                    subgoal_actor_def = ActorVectorFieldDualHead(
                        hidden_dims=config['high_actor_hidden_dims'],
                        action_dim=config['z_dim'],
                        layer_norm=config['layer_norm'],
                        time_encoder=time_encoder,
                        head_hidden_dims=head_hidden_dims,
                    )
            else:
                subgoal_actor_def = ActorVectorField(
                    hidden_dims=config['high_actor_hidden_dims'],
                    action_dim=config['z_dim'],
                    layer_norm=config['layer_norm'],
                    time_encoder=time_encoder,
                )
            if subgoal_flow_type == 'imf':
                ex_subgoal_times = jnp.zeros((*ex_actions.shape[:-1], 5), dtype=ex_actions.dtype)
                ex_subgoal_times = ex_subgoal_times.at[..., 2].set(1.0)
                ex_subgoal_times = ex_subgoal_times.at[..., 4].set(1.0)
            else:
                ex_subgoal_times = ex_times
            ex_subgoal_actor_in = (ex_observations, ex_value_goals, ex_z, ex_subgoal_times)
        elif subgoal_flow_type == 'dist':
            subgoal_actor_def = GCActor(
                hidden_dims=config['high_actor_hidden_dims'],
                action_dim=config['z_dim'],
                layer_norm=config['layer_norm'],
                state_dependent_std=False,
                const_std=pe_info['const_std'],
            )
            ex_subgoal_actor_in = (ex_observations, ex_value_goals, ex_z)
        else:
            raise ValueError(f'Unsupported subgoal_flow_type: {subgoal_flow_type}')

        ex_q_short_goals = ex_goals
        # No actions in value function inputs
        network_info = dict(
            subgoal_encoder=(subgoal_encoder_def, (ex_mid_obs,)),
            value=(value_def, (ex_observations, ex_value_goals)),
            target_value=(copy.deepcopy(value_def), (ex_observations, ex_value_goals)),
            q=(q_def, (ex_observations, ex_value_goals, ex_z)),
            target_q=(copy.deepcopy(q_def), (ex_observations, ex_value_goals, ex_z)),
            q_short=(q_short_def, (ex_observations, ex_q_short_goals, ex_actions)),  # Q_short takes actions
            actor=(actor_def, ex_actor_in),
            subgoal_actor=(subgoal_actor_def, ex_subgoal_actor_in),
            low_actor=(low_actor_def, ex_low_actor_in),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']

        # Print network architecture and parameter counts
        print("\n" + "=" * 60)
        print("Network Architecture - Parameter Counts")
        print("=" * 60)
        total_params = 0
        for name, params in network_params.items():
            module_params = sum(p.size for p in jax.tree_util.tree_leaves(params))
            total_params += module_params
            print(f"  {name:<30} {module_params:>12,} params")
        print("-" * 60)
        print(f"  {'TOTAL':<30} {total_params:>12,} params")
        print("=" * 60 + "\n")

        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network_params
        params['modules_target_value'] = params['modules_value']
        params['modules_target_q'] = params['modules_q']

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
            high_actor_hidden_dims=(256,) * 4,  # Subgoal actor + low actor (auxiliary, not used at inference).
            layer_norm=True,
            discount=0.999,
            tau=0.005,
            lam=0.0,
            expectile=0.7,
            oracle_distill=False,
            q_agg='min',
            z_dim=8,
            value_maximization='in-trajectory',  # in-trajectory or generative
            value_maximization_agg='max',  # max or expectile
            value_maximization_interval=1,
            value_maximization_fallback='none',  # none or in-trajectory
            value_maximization_weight=1.0,
            value_maximization_ramp_steps=0,  # Linear ramp from in-trajectory to generative over this many steps (0 = pure generative).
            value_maximization_intraj_floor=False,  # Take hard max with in-trajectory value as floor.
            value_sample_expectile=0.7,
            value_sample_expectile_iters=5,
            subgoal_num_samples=32,
            subgoal_flow_steps=10,
            subgoal_flow_type='auto',  # auto: use 'frs' if pe_type is frs else 'dist' (options: frs/imf/dist)
            subgoal_imf=ml_collections.ConfigDict(
                dict(
                    r_equals_t_prob=0.75,  # Official iMF default is 0.5; we use 0.75 here.
                    cfg_w_min=1.0,  # Train with w sampled from [w_min, w_max]
                    cfg_w_max=8.0,  # Enables CFG training (higher for weaker models)
                    cfg_w_power=1.0,  # Backward-compat fallback for cfg_beta when unset.
                    cfg_beta=1.0,  # Official iMF CFG scale distribution parameter.
                    cfg_scale=3.0,  # Inference CFG scale (higher for weaker models)
                    class_dropout_prob=0.1,  # Official iMF classifier-free dropout.
                    logit_mean=-0.4,  # Official iMF logit-normal mean.
                    logit_std=1.0,  # Official iMF logit-normal std.
                    adaptive_loss_eps=0.01,  # Official iMF adaptive loss epsilon.
                    adaptive_loss_p=1.0,  # Official iMF adaptive loss power.
                    head_hidden_dims=None,  # Optional per-head MLP dims for (u, v) heads.
                    use_resmlp=False,  # Use ResMLP architecture for subgoal actor.
                    resmlp_hidden_dim=256,  # ResMLP hidden dimension.
                    resmlp_num_blocks=8,  # ResMLP number of residual blocks.
                    time_embed_num_freqs=32,
                    time_embed_hidden_dim=128,
                    time_embed_dim=128,
                    time_embed_max_period=10000.0,
                    time_embed_scale=1.0,
                )
            ),
            subgoal_actor_bc_coef=1.0,
            subgoal_actor_value_coef=0.0,
            low_actor_bc_coef=1.0,
            low_actor_goal_conditioning='actual',  # 'actual', 'latent', or 'both'
            low_actor_goal_conditioned=False,  # Deprecated: use low_actor_goal_conditioning.
            low_actor_true_goal_conditioned=False,  # Deprecated: use low_actor_goal_conditioning.
            use_high_policy_inference=False,  # Ignored when low_actor_goal_conditioning is set.
            pe_type='frs',  # frs (flow rejection sampling), rpg (reparameterized grads), discrete
            frs=ml_collections.ConfigDict(dict(flow_steps=10, num_samples=32)),
            rpg=ml_collections.ConfigDict(dict(alpha=0.03, const_std=True)),
            pe_discrete=ml_collections.ConfigDict(dict(alpha=0.03, action_ct=0)),
            discrete=False,  # Set True for discrete-action environments.
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
