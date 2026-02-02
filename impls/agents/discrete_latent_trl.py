import copy
from typing import Any, Sequence

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import optax
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import (
    ActorVectorField,
    GCActor,
    GCDiscreteActor,
    GCValue,
    MLP,
    default_init,
)

def gumbel_softmax(logits, rng, temperature=1.0, hard=False):
    """Sample from the Gumbel-Softmax distribution with optional straight-through."""
    temperature = jnp.maximum(temperature, 1e-6)
    uniform = jax.random.uniform(rng, logits.shape, minval=1e-6, maxval=1.0, dtype=logits.dtype)
    gumbels = -jnp.log(-jnp.log(uniform))
    y_soft = jax.nn.softmax((logits + gumbels) / temperature, axis=-1)
    if not hard:
        return y_soft
    y_hard = jax.nn.one_hot(jnp.argmax(y_soft, axis=-1), y_soft.shape[-1], dtype=y_soft.dtype)
    return y_soft + jax.lax.stop_gradient(y_hard - y_soft)


def factorized_gumbel_softmax(logits, rng, n_factors, n_codes, temperature=1.0, hard=False):
    """Sample from factorized Gumbel-Softmax: n_factors independent categoricals with n_codes each.

    Args:
        logits: shape (..., n_factors * n_codes)
        rng: random key
        n_factors: number of independent categorical factors
        n_codes: number of codes per factor
        temperature: Gumbel-Softmax temperature
        hard: whether to use straight-through estimator

    Returns:
        shape (..., n_factors * n_codes) - concatenated one-hot vectors for each factor
    """
    batch_shape = logits.shape[:-1]
    logits = logits.reshape(*batch_shape, n_factors, n_codes)

    temperature = jnp.maximum(temperature, 1e-6)
    uniform = jax.random.uniform(rng, logits.shape, minval=1e-6, maxval=1.0, dtype=logits.dtype)
    gumbels = -jnp.log(-jnp.log(uniform))
    y_soft = jax.nn.softmax((logits + gumbels) / temperature, axis=-1)

    if hard:
        y_hard = jax.nn.one_hot(jnp.argmax(y_soft, axis=-1), n_codes, dtype=y_soft.dtype)
        y = y_soft + jax.lax.stop_gradient(y_hard - y_soft)
    else:
        y = y_soft

    return y.reshape(*batch_shape, n_factors * n_codes)


class SubgoalEncoder(nn.Module):
    """Discrete subgoal encoder: s_j -> z_j (Gumbel-Softmax)."""

    hidden_dims: Sequence[int]
    layer_norm: bool
    z_dim: int
    gumbel_temperature: float = 1.0
    gumbel_hard: bool = True

    @nn.compact
    def __call__(self, x):
        logits = MLP((*self.hidden_dims, self.z_dim), activate_final=False, layer_norm=self.layer_norm)(x)
        rng = self.make_rng('gumbel')
        return gumbel_softmax(logits, rng, temperature=self.gumbel_temperature, hard=self.gumbel_hard)


class FactorizedSubgoalEncoder(nn.Module):
    """Factorized discrete subgoal encoder using Gumbel-Softmax."""

    hidden_dims: Sequence[int]
    layer_norm: bool
    n_factors: int
    n_codes: int
    gumbel_temperature: float = 1.0
    gumbel_hard: bool = True

    @nn.compact
    def __call__(self, x):
        output_dim = self.n_factors * self.n_codes
        logits = MLP((*self.hidden_dims, output_dim), activate_final=False, layer_norm=self.layer_norm)(x)
        rng = self.make_rng('gumbel')
        return factorized_gumbel_softmax(
            logits,
            rng,
            self.n_factors,
            self.n_codes,
            temperature=self.gumbel_temperature,
            hard=self.gumbel_hard,
        )


class SubgoalActor(nn.Module):
    """Noise-conditional categorical subgoal generator (non-factorized)."""

    hidden_dims: Sequence[int]
    z_dim: int
    layer_norm: bool = False
    final_fc_init_scale: float = 1e-2

    def setup(self):
        self.actor_net = MLP(self.hidden_dims, activate_final=True, layer_norm=self.layer_norm)
        self.logit_net = nn.Dense(self.z_dim, kernel_init=default_init(self.final_fc_init_scale))

    def __call__(self, observations, goals, noise):
        inputs = [observations, goals, noise]
        x = jnp.concatenate(inputs, axis=-1)
        x = self.actor_net(x)
        return self.logit_net(x)


class FactorizedSubgoalActor(nn.Module):
    """Noise-conditional factorized categorical subgoal generator."""

    hidden_dims: Sequence[int]
    n_factors: int
    n_codes: int
    layer_norm: bool = False
    final_fc_init_scale: float = 1e-2

    def setup(self):
        self.actor_net = MLP(self.hidden_dims, activate_final=True, layer_norm=self.layer_norm)
        self.logit_net = nn.Dense(
            self.n_factors * self.n_codes,
            kernel_init=default_init(self.final_fc_init_scale),
        )

    def __call__(self, observations, goals, noise):
        inputs = [observations, goals, noise]
        x = jnp.concatenate(inputs, axis=-1)
        x = self.actor_net(x)
        return self.logit_net(x)


class DiscreteLatentTRLAgent(flax.struct.PyTreeNode):
    """Discrete latent TRL agent with factorized subgoals and noise-conditional subgoal generator."""

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

    def _encode_subgoal(self, midpoint_observations, grad_params=None, rng=None):
        rng = self.rng if rng is None else rng
        if grad_params is None:
            return self.network.select('subgoal_encoder')(midpoint_observations, rngs={'gumbel': rng})
        return self.network.select('subgoal_encoder')(
            midpoint_observations,
            params=grad_params,
            rngs={'gumbel': rng},
        )

    def _sample_subgoals(self, observations, goals, rng, num_samples=None):
        """Sample latent subgoals from the high-level policy."""
        if num_samples is None:
            num_samples = self.config['subgoal_num_samples']

        rng, noise_rng, gumbel_rng = jax.random.split(rng, 3)
        n_observations = jnp.repeat(jnp.expand_dims(observations, 0), num_samples, axis=0)
        n_goals = jnp.repeat(jnp.expand_dims(goals, 0), num_samples, axis=0)

        noise = jax.random.normal(
            noise_rng,
            (num_samples, *observations.shape[:-1], self.config['z_noise_dim']),
        )
        logits = self.network.select('subgoal_actor')(n_observations, n_goals, noise)

        if self.config['factorized']:
            return factorized_gumbel_softmax(
                logits,
                gumbel_rng,
                self.config['n_factors'],
                self.config['n_codes'],
                temperature=self.config['gumbel_temperature'],
                hard=self.config['gumbel_hard'],
            )
        return gumbel_softmax(
            logits,
            gumbel_rng,
            temperature=self.config['gumbel_temperature'],
            hard=self.config['gumbel_hard'],
        )

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

    def _subgoal_log_prob(self, logits, z_target):
        """Log-probability of z_target under factorized or flat categorical logits."""
        batch_shape = logits.shape[:-1]
        if self.config['factorized']:
            n_factors = self.config['n_factors']
            n_codes = self.config['n_codes']
            logits = logits.reshape(*batch_shape, n_factors, n_codes)
            z_target = z_target.reshape(*batch_shape, n_factors, n_codes)
            target_idx = jnp.argmax(z_target, axis=-1)
            log_probs = jax.nn.log_softmax(logits, axis=-1)
            chosen = jnp.take_along_axis(log_probs, target_idx[..., None], axis=-1).squeeze(-1)
            return chosen.sum(axis=-1)

        target_idx = jnp.argmax(z_target, axis=-1)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        return jnp.take_along_axis(log_probs, target_idx[..., None], axis=-1).squeeze(-1)

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

    def q_loss(self, batch, grad_params, rng):
        """Compute Q(s, z_j, g) loss as BCE to the triangle inequality target."""
        goal_key = 'value_goal_observations' if self.config['oracle_distill'] else 'value_goals'
        midpoint_goal_key = 'value_midpoint_observations' if self.config['oracle_distill'] else 'value_midpoint_goals'

        z_mid = self._encode_subgoal(
            batch['value_midpoint_observations'],
            grad_params=grad_params,
            rng=rng,
        )
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
        rng = rng if rng is not None else self.rng
        rng, enc_rng, gen_rng = jax.random.split(rng, 3)
        goal_key = 'value_goal_observations' if self.config['oracle_distill'] else 'value_goals'
        v_logits = self.network.select('value')(
            batch['observations'],
            goals=batch[goal_key],
            params=grad_params,
        )
        vs = jax.nn.sigmoid(v_logits)

        def compute_in_traj_target():
            z_mid = self._encode_subgoal(batch['value_midpoint_observations'], rng=enc_rng)
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
            goal_key = 'value_goal_observations' if self.config['oracle_distill'] else 'value_goals'
            z_samples = self._sample_subgoals(
                batch['observations'],
                batch[goal_key],
                gen_rng,
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

        return total_loss, {
            'total_loss': total_loss,
            'v_loss': v_loss.mean(),
            'v_mean': vs.mean(),
            'v_max': vs.max(),
            'v_min': vs.min(),
            **info,
        }

    def subgoal_actor_loss(self, batch, grad_params, rng):
        """High-level subgoal proposer loss (categorical BC + optional value maximization)."""
        rng, enc_rng, noise_rng, sample_rng = jax.random.split(rng, 4)
        z_target = self._encode_subgoal(batch['value_midpoint_observations'], rng=enc_rng)
        z_target = jax.lax.stop_gradient(z_target)

        goal_key = 'value_goal_observations' if self.config['oracle_distill'] else 'value_goals'
        batch_size = batch['observations'].shape[0]
        noise = jax.random.normal(noise_rng, (batch_size, self.config['z_noise_dim']))
        logits = self.network.select('subgoal_actor')(
            batch['observations'],
            batch[goal_key],
            noise,
            params=grad_params,
        )
        bc_loss = -self._subgoal_log_prob(logits, z_target).mean()

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

        return total_loss, {
            'actor_loss': total_loss,
            'bc_loss': bc_loss,
        }

    def _low_actor_goals(self, z, goals, true_goals=None):
        """Build low actor goal input: z, [z; goal], [z; true_goal], or [z; goal; true_goal]."""
        parts = [z]
        if self.config['low_actor_goal_conditioned']:
            parts.append(goals)
        if self.config['low_actor_true_goal_conditioned'] and true_goals is not None:
            parts.append(true_goals)
        if len(parts) == 1:
            return z
        return jnp.concatenate(parts, axis=-1)

    def low_actor_loss(self, batch, grad_params, rng):
        """Low-level actor loss conditioned on latent subgoals."""
        rng, enc_rng, flow_rng = jax.random.split(rng, 3)
        z_target = self._encode_subgoal(batch['value_midpoint_observations'], rng=enc_rng)
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

        total_loss = self.config['low_actor_bc_coef'] * bc_loss
        return total_loss, {
            'actor_loss': total_loss,
            'bc_loss': bc_loss,
        }

    def actor_loss(self, batch, grad_params, rng=None):
        """Compute the actor loss."""
        pe_info = self._get_pe_info()

        if self.config['pe_type'] == 'rpg':
            dist = self.network.select('actor')(
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
            dist = self.network.select('actor')(
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

            pred = self.network.select('actor')(
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

        rng, q_rng, value_rng, actor_rng, subgoal_actor_rng, low_actor_rng = jax.random.split(rng, 6)

        value_loss, value_info = self.value_loss(batch, grad_params, rng=value_rng)
        for k, v in value_info.items():
            info[f'value/{k}'] = v

        q_loss, q_info = self.q_loss(batch, grad_params, q_rng)
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

        if self.config['use_high_policy_inference']:
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
                vels = self.network.select('actor')(n_observations, n_goals, n_actions, t)
                n_actions = n_actions + vels / pe_info['flow_steps']
            n_actions = jnp.clip(n_actions, -1, 1)

            # Use Q_short(s, g, a) for rejection sampling
            # Q_short was distilled from V(s', g), so it scores actions without needing dynamics
            q_short = self.network.select('q_short')(n_observations, goals=n_goals, actions=n_actions)

            if len(observations.shape) == 2:
                actions = n_actions[jnp.argmax(q_short, axis=0), jnp.arange(observations.shape[0])]
            else:
                actions = n_actions[jnp.argmax(q_short)]

            return actions

        else:
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
        if config.get('factorized', False):
            z_dim = config['n_factors'] * config['n_codes']
            config['z_dim'] = z_dim
        else:
            z_dim = config['z_dim']
        ex_z = jnp.zeros((ex_observations.shape[0], z_dim), dtype=ex_observations.dtype)
        ex_noise = jnp.zeros((ex_observations.shape[0], config['z_noise_dim']), dtype=ex_observations.dtype)

        # Build example low actor goal input: z, [z; value_goal], [z; true_goal], or [z; value_goal; true_goal]
        ex_low_actor_goal_parts = [ex_z]
        if config['low_actor_goal_conditioned']:
            ex_low_actor_goal_parts.append(ex_value_goals)
        if config['low_actor_true_goal_conditioned']:
            ex_low_actor_goal_parts.append(ex_goals)
        ex_low_actor_z = jnp.concatenate(ex_low_actor_goal_parts, axis=-1) if len(ex_low_actor_goal_parts) > 1 else ex_z

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
        if config.get('factorized', False):
            subgoal_encoder_def = FactorizedSubgoalEncoder(
                hidden_dims=config['value_hidden_dims'],
                layer_norm=config['layer_norm'],
                n_factors=config['n_factors'],
                n_codes=config['n_codes'],
                gumbel_temperature=config['gumbel_temperature'],
                gumbel_hard=config['gumbel_hard'],
            )
        else:
            subgoal_encoder_def = SubgoalEncoder(
                hidden_dims=config['value_hidden_dims'],
                layer_norm=config['layer_norm'],
                z_dim=config['z_dim'],
                gumbel_temperature=config['gumbel_temperature'],
                gumbel_hard=config['gumbel_hard'],
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
        elif config['pe_type'] == 'discrete':
            actor_def = GCDiscreteActor(
                hidden_dims=config['actor_hidden_dims'],
                action_dim=config['pe_discrete']['action_ct'],
                layer_norm=config['layer_norm'],
            )
            ex_actor_in = (ex_observations, ex_goals, ex_actions)
        else:
            actor_def = GCActor(
                hidden_dims=config['actor_hidden_dims'],
                action_dim=action_dim,
                layer_norm=config['layer_norm'],
                state_dependent_std=False,
                const_std=pe_info['const_std'],
            )
            ex_actor_in = (ex_observations, ex_goals, ex_actions)

        if config.get('factorized', False):
            subgoal_actor_def = FactorizedSubgoalActor(
                hidden_dims=config['subgoal_actor_hidden_dims'],
                n_factors=config['n_factors'],
                n_codes=config['n_codes'],
                layer_norm=config['layer_norm'],
            )
        else:
            subgoal_actor_def = SubgoalActor(
                hidden_dims=config['subgoal_actor_hidden_dims'],
                z_dim=config['z_dim'],
                layer_norm=config['layer_norm'],
            )
        ex_subgoal_actor_in = (ex_observations, ex_value_goals, ex_noise)

        low_actor_hidden_dims = config.get('low_actor_hidden_dims', config['high_actor_hidden_dims'])
        if config['pe_type'] == 'frs':
            low_actor_def = ActorVectorField(
                hidden_dims=low_actor_hidden_dims,
                action_dim=action_dim,
                layer_norm=config['layer_norm'],
            )
            ex_low_actor_in = (ex_observations, ex_low_actor_z, ex_actions, ex_times)
        elif config['pe_type'] == 'discrete':
            low_actor_def = GCDiscreteActor(
                hidden_dims=low_actor_hidden_dims,
                action_dim=config['pe_discrete']['action_ct'],
                layer_norm=config['layer_norm'],
            )
            ex_low_actor_in = (ex_observations, ex_low_actor_z, ex_actions)
        else:
            low_actor_def = GCActor(
                hidden_dims=low_actor_hidden_dims,
                action_dim=action_dim,
                layer_norm=config['layer_norm'],
                state_dependent_std=False,
                const_std=pe_info['const_std'],
            )
            ex_low_actor_in = (ex_observations, ex_low_actor_z, ex_actions)

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
        rng, gumbel_rng = jax.random.split(rng)
        network_params = network_def.init({'params': init_rng, 'gumbel': gumbel_rng}, **network_args)['params']

        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network_params
        params['modules_target_value'] = params['modules_value']
        params['modules_target_q'] = params['modules_q']

        config['action_dim'] = action_dim
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='discrete_latent_trl',
            lr=3e-4,
            batch_size=1024,
            actor_hidden_dims=(1024,) * 4,
            value_hidden_dims=(1024,) * 4,
            high_actor_hidden_dims=(256,) * 4,  # Low-level actor hidden dims (fallback).
            subgoal_actor_hidden_dims=(),  # Logistic regression by default (no hidden layers).
            low_actor_hidden_dims=(256,) * 4,
            layer_norm=True,
            discount=0.999,
            tau=0.005,
            lam=0.0,
            expectile=0.7,
            oracle_distill=False,
            q_agg='min',
            z_dim=8,  # Used when factorized=False.
            factorized=True,  # Use factorized discrete subgoals by default.
            n_factors=8,
            n_codes=8,
            z_noise_dim=8,
            gumbel_temperature=1.0,
            gumbel_hard=True,
            value_maximization='in-trajectory',  # in-trajectory or generative
            value_maximization_agg='max',  # max or expectile
            value_maximization_interval=1,
            value_maximization_fallback='none',  # none or in-trajectory
            value_maximization_weight=1.0,
            value_maximization_ramp_steps=0,  # Linear ramp from in-trajectory to generative over this many steps (0 = pure generative).
            value_sample_expectile=0.7,
            value_sample_expectile_iters=5,
            subgoal_num_samples=32,
            subgoal_actor_bc_coef=1.0,
            subgoal_actor_value_coef=0.0,
            low_actor_bc_coef=1.0,
            low_actor_goal_conditioned=False,  # Whether low actor sees the subgoal goal (value goal) in addition to z.
            low_actor_true_goal_conditioned=False,  # Whether low actor sees the true task goal in addition to z.
            use_high_policy_inference=False,
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
