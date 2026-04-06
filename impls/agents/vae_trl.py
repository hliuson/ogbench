import copy
from typing import Any, Sequence

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import optax
from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.flows import imf_loss, imf_one_shot_sample
from utils.networks import ActorVectorField, GCActor, GCDiscreteActor, GCValue, MLP
from utils.pretraining import ImpalaDecoder


class VAEEncoder(nn.Module):
    """MLP encoder producing (mu, log_var)."""

    hidden_dims: Sequence[int]
    z_dim: int
    layer_norm: bool

    @nn.compact
    def __call__(self, x):
        h = MLP(self.hidden_dims, activate_final=True, layer_norm=self.layer_norm)(x)
        mu = nn.Dense(self.z_dim)(h)
        log_var = nn.Dense(self.z_dim)(h)
        return mu, log_var


class VAEDecoder(nn.Module):
    """MLP decoder for reconstruction."""

    hidden_dims: Sequence[int]
    obs_dim: int
    layer_norm: bool

    @nn.compact
    def __call__(self, z):
        return MLP((*self.hidden_dims, self.obs_dim), activate_final=False, layer_norm=self.layer_norm)(z)


class VaeTRLAgent(flax.struct.PyTreeNode):
    """Latent TRL with unified VAE encoding.

    All observations are encoded through a single VAE encoder so that value,
    Q, actor, and z-proposal networks all operate in the same latent space.
    """

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

    def _get_reg_coef(self):
        return self.config.get('reg_coef', self.config.get('vae_coef', 1.0))

    def _has_visual_encoder(self):
        encoder_name = self.config.get('encoder', '')
        return encoder_name not in (None, '')

    def _raw_observation_shape(self):
        raw_shape = self.config.get('raw_observation_shape', ())
        return tuple(raw_shape) if raw_shape is not None else ()

    def _matches_raw_observation_shape(self, x):
        raw_shape = self._raw_observation_shape()
        if not raw_shape or x.ndim < len(raw_shape) + 1:
            return False
        return tuple(x.shape[-len(raw_shape):]) == raw_shape

    def _encode_visual(self, observations, grad_params=None):
        if not self._has_visual_encoder():
            return observations

        if grad_params is None or self.config.get('freeze_encoder', False):
            encoded = self.network.select('encoder')(observations, train=False)
            if self.config.get('freeze_encoder', False):
                encoded = jax.lax.stop_gradient(encoded)
            return encoded

        return self.network.select('encoder')(observations, params=grad_params, train=True)

    def _encode_policy_goals(self, goals, grad_params=None):
        if goals is None:
            return None
        if self._has_visual_encoder() and (
            self._matches_raw_observation_shape(goals) or goals.ndim > 2
        ):
            return self._encode_visual(goals, grad_params=grad_params)
        return goals

    def _encode(self, observations, grad_params=None, rng=None, sample=False):
        """Encode observations through VAE encoder.

        When grad_params is None: stop-gradient (standard TrainState stored params).
        When grad_params is provided: gradients flow through encoder.
        """
        observations = self._encode_visual(observations, grad_params=grad_params)
        if grad_params is not None:
            mu, log_var = self.network.select('vae_encoder')(observations, params=grad_params)
        else:
            mu, log_var = self.network.select('vae_encoder')(observations)

        if sample:
            if rng is None:
                raise ValueError('Sampling latents requires rng.')
            std = jnp.exp(0.5 * log_var)
            eps = jax.random.normal(rng, mu.shape)
            return mu + std * eps
        return mu

    def _encode_rl(self, observations, grad_params, role, rng=None):
        """Encode observations for an RL role with gradients through the encoder."""
        del role
        return self._encode(
            observations,
            grad_params,
            rng=rng,
            sample=self.config.get('sample_latent_for_rl', False),
        )

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

    def _aggregate_value_ensembles(self, values):
        return self._aggregate_q_ensembles(values)

    def _local_step_consistency_metrics(self, v_cur, v_next, mask=None):
        discount = self.config['discount']
        d_cur = -jnp.log(jnp.clip(v_cur, 1e-8, 1.0)) / (-jnp.log(discount))
        d_next = -jnp.log(jnp.clip(v_next, 1e-8, 1.0)) / (-jnp.log(discount))
        delta = jnp.abs(d_cur - d_next)
        violation = jnp.maximum(delta - 1.0, 0.0)
        if mask is None:
            mask = jnp.ones_like(delta)
        return {
            'local_step_delta_mean': self._masked_mean(delta, mask),
            'local_step_violation_mean': self._masked_mean(violation, mask),
            'local_step_violation_frac': self._masked_mean(
                (violation > 0).astype(violation.dtype),
                mask,
            ),
        }

    @staticmethod
    def _masked_mean(values, mask):
        mask = mask.astype(values.dtype)
        denom = jnp.maximum(mask.sum(), 1.0)
        return (values * mask).sum() / denom

    @staticmethod
    def _local_value_diff_metrics(v_cur, v_next, threshold, mask=None):
        delta = jnp.abs(v_cur - v_next)
        violation = jnp.maximum(delta - threshold, 0.0)
        if mask is None:
            mask = jnp.ones_like(delta)
        return {
            'local_value_delta_mean': VaeTRLAgent._masked_mean(delta, mask),
            'local_value_violation_mean': VaeTRLAgent._masked_mean(violation, mask),
            'local_value_violation_frac': VaeTRLAgent._masked_mean(
                (violation > 0).astype(violation.dtype),
                mask,
            ),
        }

    def _has_learned_z_proposal(self):
        return self.config['z_proposal_coef'] > 0

    def _has_cf_midpoint_stitching(self, batch):
        return 'value_goals_is_intraj' in batch and self.config['z_proposal_coef'] > 0

    def vae_loss(self, batch, grad_params, rng):
        """Variational bottleneck loss over observations and midpoints."""
        raw_obs = batch['observations']
        raw_midpoints = batch['value_midpoint_observations']
        obs = self._encode_visual(raw_obs, grad_params=grad_params)
        midpoints = self._encode_visual(raw_midpoints, grad_params=grad_params)
        # Combine observations and midpoints for VAE training.
        all_obs = jnp.concatenate([obs, midpoints], axis=0)
        if self._has_visual_encoder():
            all_targets = jnp.concatenate([raw_obs, raw_midpoints], axis=0).astype(jnp.float32) / 255.0
        else:
            all_targets = all_obs

        mu, log_var = self.network.select('vae_encoder')(all_obs, params=grad_params)

        # Reparameterization trick.
        std = jnp.exp(0.5 * log_var)
        eps = jax.random.normal(rng, mu.shape)
        z = mu + std * eps

        recon_coef = self.config.get('vae_recon_coef', 1.0)
        if recon_coef > 0:
            recon = self.network.select('vae_decoder')(z, params=grad_params)
            if self._has_visual_encoder():
                recon = jax.nn.sigmoid(recon)
            recon_loss = jnp.mean((recon - all_targets) ** 2)
        else:
            recon_loss = jnp.asarray(0.0, dtype=mu.dtype)
        kl_loss = -0.5 * jnp.mean(1 + log_var - mu ** 2 - jnp.exp(log_var))

        total_loss = recon_coef * recon_loss + self.config['vae_beta'] * kl_loss

        return total_loss, {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'z_mean': mu.mean(),
            'z_std': std.mean(),
        }

    def representation_loss(self, batch, grad_params, rng):
        return self.vae_loss(batch, grad_params, rng)

    def q_loss(self, batch, grad_params, rng=None, cf_weight=1.0):
        """Train Q(z_s, z_g, z_mid) with factored backup using single value network."""
        has_cf = self._has_cf_midpoint_stitching(batch)
        z_s_rng, z_g_rng, z_mid_rng, cf_rng = jax.random.split(rng, 4)

        goal_key = 'value_goal_observations'
        z_s = self._encode_rl(batch['observations'], grad_params, 'observation', rng=z_s_rng)
        z_g = self._encode_rl(batch[goal_key], grad_params, 'goal', rng=z_g_rng)
        z_mid = self._encode_rl(batch['value_midpoint_observations'], grad_params, 'midpoint', rng=z_mid_rng)

        q_logits = self.network.select('q')(
            z_s, goals=z_g, actions=z_mid, params=grad_params,
        )

        second_offset = batch['value_offsets'][None, ...] - batch['value_midpoint_offsets']

        # Factored target: V(z_s, z_mid) * V(z_mid, z_g) using SAME value network.
        first_v_logits = self.network.select('target_value')(z_s, goals=z_mid)
        second_v_logits = self.network.select('target_value')(z_mid, goals=z_g)

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
        intraj_target = first_v * second_v

        # --- Counterfactual path ---
        if has_cf:
            is_intraj = batch['value_goals_is_intraj']
            is_intraj_2d = is_intraj[None, ...]

            z_cf = self._sample_cf_midpoints(batch, cf_rng, num_samples=1)
            z_cf = jax.lax.stop_gradient(z_cf)

            q_cf_logits = self.network.select('q')(
                z_s, goals=z_g, actions=z_cf, params=grad_params,
            )

            cf_first_v_logits = self.network.select('target_value')(z_s, goals=z_cf)
            cf_second_v_logits = self.network.select('target_value')(z_cf, goals=z_g)
            cf_target = jax.nn.sigmoid(cf_first_v_logits) * jax.nn.sigmoid(cf_second_v_logits)

            q_logits = jnp.where(is_intraj_2d, q_logits, q_cf_logits)
            target = jnp.where(is_intraj_2d, intraj_target, cf_target)
        else:
            target = intraj_target

        qs = jax.nn.sigmoid(q_logits)

        dist = jax.lax.stop_gradient(jnp.log(target) / jnp.log(self.config['discount']))
        dist_weight = (1 / (1 + dist)) ** self.config['lam']
        q_loss = dist_weight * self.bce_loss(q_logits, jax.lax.stop_gradient(target))
        if has_cf:
            sample_weight = is_intraj_2d + (1.0 - is_intraj_2d) * cf_weight
            q_loss = q_loss * sample_weight
        total_loss = q_loss.mean()

        info = {
            'total_loss': total_loss,
            'q_loss': q_loss.mean(),
            'q_mean': qs.mean(),
            'q_max': qs.max(),
            'q_min': qs.min(),
            'first_factor_mean': first_v.mean(),
            'second_factor_mean': second_v.mean(),
        }
        if has_cf:
            info['cf_target_mean'] = cf_target.mean()
            info['cf_frac'] = (1.0 - is_intraj).mean()

        return total_loss, info

    def value_loss(self, batch, grad_params, rng=None, cf_weight=1.0):
        """Value loss with expectile regression, all in latent space."""
        has_cf = self._has_cf_midpoint_stitching(batch)
        z_s_rng, z_g_rng, proposal_rng, z_next_rng = jax.random.split(rng, 4)

        goal_key = 'value_goal_observations'
        z_s = self._encode_rl(batch['observations'], grad_params, 'observation', rng=z_s_rng)
        z_g = self._encode_rl(batch[goal_key], grad_params, 'goal', rng=z_g_rng)
        z_next = self._encode_rl(batch['next_observations'], grad_params, 'observation', rng=z_next_rng)

        v_logits = self.network.select('value')(z_s, goals=z_g, params=grad_params)
        vs = jax.nn.sigmoid(v_logits)
        next_v_logits = self.network.select('value')(z_next, goals=z_g, params=grad_params)
        next_vs = jax.nn.sigmoid(next_v_logits)

        backup_mode = self.config.get('value_backup_mode', 'q')
        z_mid = self._encode(batch['value_midpoint_observations'])
        z_mid = jax.lax.stop_gradient(z_mid)
        second_offset = batch['value_offsets'] - batch['value_midpoint_offsets']

        if backup_mode == 'q':
            q_logits = self.network.select('target_q')(z_s, goals=z_g, actions=z_mid)
            q_traj = jax.nn.sigmoid(q_logits)
            target_traj = self._aggregate_q_ensembles(q_traj)
        elif backup_mode == 'factorized_v':
            first_v_logits = self.network.select('target_value')(z_s, goals=z_mid)
            second_v_logits = self.network.select('target_value')(z_mid, goals=z_g)
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
            target_traj = self._aggregate_value_ensembles(first_v) * self._aggregate_value_ensembles(second_v)
        else:
            raise ValueError(f'Unsupported value_backup_mode: {backup_mode}')

        # --- Proposed z target ---
        use_cf_midpoints = has_cf
        if use_cf_midpoints and has_cf:
            is_intraj = batch['value_goals_is_intraj']
            num_cf_samples = max(1, int(self.config.get('cf_num_z_proposals', 1)))

            z_proposed = self._sample_cf_midpoints(batch, proposal_rng, num_samples=num_cf_samples)
            z_proposed = jax.lax.stop_gradient(z_proposed)

            if backup_mode == 'q':
                if num_cf_samples == 1:
                    q_proposed_logits = self.network.select('target_q')(z_s, goals=z_g, actions=z_proposed)
                    q_proposed = jax.nn.sigmoid(q_proposed_logits)
                    target_proposed = self._aggregate_q_ensembles(q_proposed)
                else:
                    z_s_bc = jnp.broadcast_to(z_s[None], (num_cf_samples, *z_s.shape))
                    z_g_bc = jnp.broadcast_to(z_g[None], (num_cf_samples, *z_g.shape))
                    q_proposed_logits = self.network.select('target_q')(
                        z_s_bc.reshape(-1, z_s.shape[-1]),
                        goals=z_g_bc.reshape(-1, z_g.shape[-1]),
                        actions=z_proposed.reshape(-1, z_proposed.shape[-1]),
                    )
                    q_proposed = jax.nn.sigmoid(q_proposed_logits)
                    target_proposed = self._aggregate_q_ensembles(q_proposed).reshape(num_cf_samples, -1).max(axis=0)
            elif backup_mode == 'factorized_v':
                if num_cf_samples == 1:
                    first_v_logits = self.network.select('target_value')(z_s, goals=z_proposed)
                    second_v_logits = self.network.select('target_value')(z_proposed, goals=z_g)
                    target_proposed = (
                        self._aggregate_value_ensembles(jax.nn.sigmoid(first_v_logits))
                        * self._aggregate_value_ensembles(jax.nn.sigmoid(second_v_logits))
                    )
                else:
                    z_s_bc = jnp.broadcast_to(z_s[None], (num_cf_samples, *z_s.shape))
                    z_g_bc = jnp.broadcast_to(z_g[None], (num_cf_samples, *z_g.shape))
                    first_v_logits = self.network.select('target_value')(
                        z_s_bc.reshape(-1, z_s.shape[-1]),
                        goals=z_proposed.reshape(-1, z_proposed.shape[-1]),
                    )
                    second_v_logits = self.network.select('target_value')(
                        z_proposed.reshape(-1, z_proposed.shape[-1]),
                        goals=z_g_bc.reshape(-1, z_g.shape[-1]),
                    )
                    first_v = self._aggregate_value_ensembles(jax.nn.sigmoid(first_v_logits)).reshape(num_cf_samples, -1)
                    second_v = self._aggregate_value_ensembles(jax.nn.sigmoid(second_v_logits)).reshape(num_cf_samples, -1)
                    target_proposed = (first_v * second_v).max(axis=0)
            else:
                raise ValueError(f'Unsupported value_backup_mode: {backup_mode}')

            if self.config.get('intraj_cf_max_target', True):
                intraj_target = jnp.maximum(target_traj, cf_weight * target_proposed)
            else:
                intraj_target = target_traj
            cf_target_val = cf_weight * target_proposed
            target = jnp.where(is_intraj, intraj_target, cf_target_val)
        elif use_cf_midpoints:
            num_cf_samples = max(1, int(self.config.get('cf_num_z_proposals', 1)))
            z_proposed = self._sample_cf_midpoints(batch, proposal_rng, num_samples=num_cf_samples)
            z_proposed = jax.lax.stop_gradient(z_proposed)
            if backup_mode == 'q':
                if num_cf_samples == 1:
                    q_proposed_logits = self.network.select('target_q')(z_s, goals=z_g, actions=z_proposed)
                    q_proposed = jax.nn.sigmoid(q_proposed_logits)
                    target_proposed = self._aggregate_q_ensembles(q_proposed)
                else:
                    z_s_bc = jnp.broadcast_to(z_s[None], (num_cf_samples, *z_s.shape))
                    z_g_bc = jnp.broadcast_to(z_g[None], (num_cf_samples, *z_g.shape))
                    q_proposed_logits = self.network.select('target_q')(
                        z_s_bc.reshape(-1, z_s.shape[-1]),
                        goals=z_g_bc.reshape(-1, z_g.shape[-1]),
                        actions=z_proposed.reshape(-1, z_proposed.shape[-1]),
                    )
                    q_proposed = jax.nn.sigmoid(q_proposed_logits)
                    target_proposed = self._aggregate_q_ensembles(q_proposed).reshape(num_cf_samples, -1).max(axis=0)
            elif backup_mode == 'factorized_v':
                if num_cf_samples == 1:
                    first_v_logits = self.network.select('target_value')(z_s, goals=z_proposed)
                    second_v_logits = self.network.select('target_value')(z_proposed, goals=z_g)
                    target_proposed = (
                        self._aggregate_value_ensembles(jax.nn.sigmoid(first_v_logits))
                        * self._aggregate_value_ensembles(jax.nn.sigmoid(second_v_logits))
                    )
                else:
                    z_s_bc = jnp.broadcast_to(z_s[None], (num_cf_samples, *z_s.shape))
                    z_g_bc = jnp.broadcast_to(z_g[None], (num_cf_samples, *z_g.shape))
                    first_v_logits = self.network.select('target_value')(
                        z_s_bc.reshape(-1, z_s.shape[-1]),
                        goals=z_proposed.reshape(-1, z_proposed.shape[-1]),
                    )
                    second_v_logits = self.network.select('target_value')(
                        z_proposed.reshape(-1, z_proposed.shape[-1]),
                        goals=z_g_bc.reshape(-1, z_g.shape[-1]),
                    )
                    first_v = self._aggregate_value_ensembles(jax.nn.sigmoid(first_v_logits)).reshape(num_cf_samples, -1)
                    second_v = self._aggregate_value_ensembles(jax.nn.sigmoid(second_v_logits)).reshape(num_cf_samples, -1)
                    target_proposed = (first_v * second_v).max(axis=0)
            else:
                raise ValueError(f'Unsupported value_backup_mode: {backup_mode}')
            target = jnp.maximum(target_traj, target_proposed)
        else:
            target = target_traj

        if use_cf_midpoints and has_cf:
            tau = jnp.where(
                is_intraj,
                self.config['expectile'],
                self.config.get('cf_expectile', self.config['expectile']),
            )
        else:
            tau = jnp.full_like(target, self.config['expectile'])
        expectile_weight = jnp.where(target >= vs, tau, (1 - tau))
        dist = jax.lax.stop_gradient(jnp.log(target) / jnp.log(self.config['discount']))
        dist_weight = (1 / (1 + dist)) ** self.config['lam']
        v_loss = expectile_weight * dist_weight * self.bce_loss(v_logits, jax.lax.stop_gradient(target))
        total_loss = v_loss.mean()

        cur_v_pred = self._aggregate_value_ensembles(vs)
        next_v_pred = self._aggregate_value_ensembles(next_vs)
        discount = self.config['discount']
        d_cur = -jnp.log(jnp.clip(cur_v_pred, 1e-8, 1.0)) / (-jnp.log(discount))
        d_next = -jnp.log(jnp.clip(next_v_pred, 1e-8, 1.0)) / (-jnp.log(discount))
        local_step_violation = jax.nn.relu(jnp.abs(d_cur - d_next) - 1.0)
        local_step_mask = jnp.ones_like(local_step_violation)
        if has_cf and self.config.get('local_step_hinge_only_cf', True):
            local_step_mask = 1.0 - batch['value_goals_is_intraj']
        local_step_hinge_loss = self._masked_mean(local_step_violation, local_step_mask)
        total_loss = total_loss + self.config.get('local_step_hinge_coef', 0.0) * local_step_hinge_loss

        local_value_diff = jnp.abs(cur_v_pred - next_v_pred)
        local_value_mask = jnp.ones_like(local_value_diff)
        if has_cf and self.config.get('local_value_smoothness_only_cf', True):
            local_value_mask = 1.0 - batch['value_goals_is_intraj']
        local_value_threshold = self.config.get('local_value_smoothness_threshold', 0.01)
        local_value_hinge = jax.nn.relu(local_value_diff - local_value_threshold)
        local_value_smoothness_loss = self._masked_mean(local_value_hinge, local_value_mask)
        total_loss = total_loss + self.config.get('local_value_smoothness_coef', 0.0) * local_value_smoothness_loss

        predicted_steps = jnp.log(vs + 1e-8) / jnp.log(self.config['discount'])
        actual_steps = batch['value_offsets']
        if has_cf:
            calib_mask = batch['value_goals_is_intraj'] * (actual_steps > 0).astype(jnp.float32)
            safe_actual = jnp.maximum(actual_steps, 1.0)
            relative_gap = (predicted_steps - safe_actual) / (safe_actual + 1e-8)
            relative_gap = relative_gap * calib_mask[None, ...]
            calib_denom = jnp.maximum(calib_mask.sum() * relative_gap.shape[0], 1.0)
            calib_mean = relative_gap.sum() / calib_denom
            calib_max = jnp.where(
                calib_mask[None, ...] > 0,
                relative_gap,
                -jnp.inf,
            ).max()
            calib_max = jnp.where(jnp.isfinite(calib_max), calib_max, 0.0)
        else:
            calib_mask = (actual_steps > 0).astype(jnp.float32)
            safe_actual = jnp.maximum(actual_steps, 1.0)
            relative_gap = (predicted_steps - safe_actual) / (safe_actual + 1e-8)
            relative_gap = relative_gap * calib_mask[None, ...]
            calib_denom = jnp.maximum(calib_mask.sum() * relative_gap.shape[0], 1.0)
            calib_mean = relative_gap.sum() / calib_denom
            calib_max = jnp.where(
                calib_mask[None, ...] > 0,
                relative_gap,
                -jnp.inf,
            ).max()
            calib_max = jnp.where(jnp.isfinite(calib_max), calib_max, 0.0)

        info = {
            'total_loss': total_loss,
            'v_loss': total_loss,
            'bootstrapped_v_loss': v_loss.mean(),
            'v_mean': vs.mean(),
            'v_max': vs.max(),
            'v_min': vs.min(),
            'calibration_rel_gap_mean': calib_mean,
            'calibration_rel_gap_max': calib_max,
        }
        info.update(self._local_step_consistency_metrics(cur_v_pred, next_v_pred, local_step_mask))
        info.update(self._local_value_diff_metrics(cur_v_pred, next_v_pred, local_value_threshold, local_value_mask))
        if use_cf_midpoints:
            info['backup_proposed_mean'] = target_proposed.mean()
            info['backup_traj_mean'] = target_traj.mean()
            if backup_mode == 'q':
                info['q_proposed_mean'] = target_proposed.mean()
                info['q_traj_mean'] = target_traj.mean()
        return total_loss, info

    def z_proposal_loss(self, batch, grad_params, rng):
        """Train iMF z-proposal: f(noise; z_s, z_g) -> z_mid, all in latent space."""
        goal_key = 'value_goal_observations'
        z_s_rng, z_g_rng, imf_rng = jax.random.split(rng, 3)
        z_target = self._encode(batch['value_midpoint_observations'])
        z_target = jax.lax.stop_gradient(z_target)

        z_s = self._encode_rl(batch['observations'], grad_params, 'observation', rng=z_s_rng)
        z_g = self._encode_rl(batch[goal_key], grad_params, 'goal', rng=z_g_rng)

        def vector_field_fn(noise, times):
            return self.network.select('z_proposal')(
                z_s, goals=z_g, actions=noise, times=times,
                params=grad_params,
            )

        intraj_mask = batch.get('value_goals_is_intraj', None)

        loss, flow_info = imf_loss(
            imf_rng, z_target, vector_field_fn,
            r_equals_t_prob=0.5,
            mask=intraj_mask,
        )

        info = {'loss': loss}
        if intraj_mask is not None:
            info['intraj_frac'] = intraj_mask.mean()

        for k, v in flow_info.items():
            if k != 'loss':
                info[k] = v

        return loss, info

    def _sample_z_proposal(self, batch, rng, num_samples=1):
        """Sample z candidates from z-proposal via iMF one-shot, all in latent space."""
        goal_key = 'value_goal_observations'
        z_s = self._encode(batch['observations'])
        z_g = self._encode(batch[goal_key])
        z_dim = self.config['z_dim']

        if num_samples == 1:
            sample_shape = (z_s.shape[0], z_dim)
        else:
            sample_shape = (num_samples, z_s.shape[0], z_dim)

        def vector_field_fn(noise, times):
            if num_samples == 1:
                return self.network.select('z_proposal')(
                    z_s, goals=z_g, actions=noise, times=times,
                )
            else:
                obs_bc = jnp.broadcast_to(z_s[None], (num_samples, *z_s.shape))
                goals_bc = jnp.broadcast_to(z_g[None], (num_samples, *z_g.shape))
                obs_flat = obs_bc.reshape(-1, z_s.shape[-1])
                goals_flat = goals_bc.reshape(-1, z_g.shape[-1])
                noise_flat = noise.reshape(-1, z_dim)
                times_flat = times.reshape(-1, times.shape[-1])
                out = self.network.select('z_proposal')(
                    obs_flat, goals=goals_flat, actions=noise_flat, times=times_flat,
                )
                return out.reshape(num_samples, z_s.shape[0], z_dim)

        z_samples = imf_one_shot_sample(rng, sample_shape, vector_field_fn)
        return z_samples

    def _sample_cf_midpoints(self, batch, rng, num_samples=1):
        return self._sample_z_proposal(batch, rng, num_samples=num_samples)

    def q_action_loss(self, batch, grad_params, rng):
        """Distill Q_short from an n-step bootstrap into V for action selection."""
        z_s_rng, z_next_rng, z_g_rng = jax.random.split(rng, 3)
        q_action_goal_key = 'actor_goals'
        target_goal_key = 'actor_goal_observations'
        q_action_n_step = int(self.config.get('q_action_n_step', 1))
        next_obs_key = 'actor_nstep_observations' if q_action_n_step > 1 and 'actor_nstep_observations' in batch else 'next_observations'
        z_s = self._encode_rl(batch['observations'], grad_params, 'observation', rng=z_s_rng)
        z_next = self._encode_rl(batch[next_obs_key], grad_params, 'observation', rng=z_next_rng)
        z_g_encoded = self._encode_rl(batch[target_goal_key], grad_params, 'goal', rng=z_g_rng)
        if self.config.get('q_action_stopgrad_encoder', False):
            z_s = jax.lax.stop_gradient(z_s)
            z_next = jax.lax.stop_gradient(z_next)
            z_g_encoded = jax.lax.stop_gradient(z_g_encoded)

        v_next_logits = self.network.select('target_value')(z_next, goals=z_g_encoded)
        v_next = jax.nn.sigmoid(v_next_logits)
        v_next_min = jnp.minimum(v_next[0], v_next[1])
        if q_action_n_step > 1 and 'actor_nstep_steps' in batch:
            n_steps = batch['actor_nstep_steps']
        else:
            n_steps = jnp.ones_like(batch['actions'][..., 0], dtype=jnp.int32)
        bootstrap_target = (self.config['discount'] ** n_steps) * v_next_min

        if q_action_n_step > 1 and 'actor_goals_is_intraj' in batch and 'actor_goal_offsets' in batch:
            actor_is_intraj = batch['actor_goals_is_intraj']
            actor_goal_offsets = batch['actor_goal_offsets']
            exact_reached = actor_is_intraj * (actor_goal_offsets <= n_steps)
            exact_target = self.config['discount'] ** actor_goal_offsets
            target = jnp.where(exact_reached > 0, exact_target, bootstrap_target)
        else:
            exact_reached = jnp.zeros_like(bootstrap_target)
            target = bootstrap_target

        q_action_goals = self._encode_policy_goals(batch[q_action_goal_key], grad_params=grad_params)

        q_action_logits = self.network.select('q_action')(
            z_s, goals=q_action_goals, actions=batch['actions'], params=grad_params,
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
        """Actor loss. Actor takes latent obs and encoded/raw goals consistently."""
        pe_info = self._get_pe_info()
        rngs = jax.random.split(rng, 5)

        z_s = self._encode_rl(batch['observations'], grad_params, 'observation', rng=rngs[0])
        policy_goals = self._encode_policy_goals(batch['actor_goals'], grad_params=grad_params)

        if self.config['pe_type'] == 'rpg':
            dist = self.network.select('actor')(z_s, policy_goals, params=grad_params)

            z_next = self._encode_rl(batch['next_observations'], grad_params, 'observation', rng=rngs[1])
            z_actor_g = self._encode_rl(batch['actor_goal_observations'], grad_params, 'goal', rng=rngs[2])
            v_next = self.network.select('value')(z_next, goals=z_actor_g)
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
            dist = self.network.select('actor')(z_s, policy_goals, params=grad_params)

            z_next = self._encode_rl(batch['next_observations'], grad_params, 'observation', rng=rngs[1])
            z_actor_g = self._encode_rl(batch['actor_goal_observations'], grad_params, 'goal', rng=rngs[2])
            v = self.network.select('value')(z_next, goals=z_actor_g).mean(axis=0)
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
            x_rng, t_rng = rngs[3], rngs[4]

            x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
            x_1 = batch['actions']
            t = jax.random.uniform(t_rng, (batch_size, 1))
            x_t = (1 - t) * x_0 + t * x_1
            y = x_1 - x_0

            pred = self.network.select('actor')(
                z_s, policy_goals, x_t, t, params=grad_params,
            )
            actor_loss = jnp.mean((pred - y) ** 2)

            return actor_loss, {
                'actor_loss': actor_loss,
            }

        raise ValueError(f"Unsupported pe_type: {self.config['pe_type']}")

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None, step=0):
        info = {}
        rng = rng if rng is not None else self.rng
        rng, vae_rng, value_rng, q_rng, q_action_rng, actor_rng, z_proposal_rng = (
            jax.random.split(rng, 7)
        )

        # CF warmup weight.
        cf_burnin_steps = self.config.get('cf_burnin_steps', 0)
        cf_warmup_steps = self.config.get('cf_warmup_steps', 0)
        if cf_burnin_steps > 0 or cf_warmup_steps > 0:
            ramp_progress = jnp.maximum(step - cf_burnin_steps, 0)
            cf_weight = jnp.where(
                cf_warmup_steps > 0,
                jnp.minimum(ramp_progress / cf_warmup_steps, 1.0),
                jnp.where(step >= cf_burnin_steps, 1.0, 0.0),
            )
        else:
            cf_weight = 1.0
        info['cf_weight'] = jnp.asarray(cf_weight, dtype=jnp.float32)

        reg_coef = self._get_reg_coef()
        if reg_coef > 0:
            reg_loss, reg_info = self.representation_loss(batch, grad_params, rng=vae_rng)
            for k, v in reg_info.items():
                info[f'vae/{k}'] = v
        else:
            reg_loss = jnp.asarray(0.0, dtype=jnp.float32)
            info['vae/disabled'] = jnp.asarray(1.0, dtype=jnp.float32)

        value_loss, value_info = self.value_loss(batch, grad_params, rng=value_rng, cf_weight=cf_weight)
        for k, v in value_info.items():
            info[f'value/{k}'] = v

        q_loss, q_info = self.q_loss(batch, grad_params, rng=q_rng, cf_weight=cf_weight)
        for k, v in q_info.items():
            info[f'q/{k}'] = v

        z_proposal_loss = 0.0
        if self._has_learned_z_proposal():
            z_proposal_loss, z_proposal_info = self.z_proposal_loss(batch, grad_params, rng=z_proposal_rng)
            for k, v in z_proposal_info.items():
                info[f'z_proposal/{k}'] = v

        q_action_loss, q_action_info = self.q_action_loss(batch, grad_params, rng=q_action_rng)
        for k, v in q_action_info.items():
            info[f'q_action/{k}'] = v

        actor_loss, actor_info = self.actor_loss(batch, grad_params, rng=actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = (
            reg_coef * reg_loss
            + value_loss
            + q_loss
            + self.config['z_proposal_coef'] * z_proposal_loss
            + q_action_loss
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
    def update(self, batch, step=0):
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng, step=step)

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

        z_s = self._encode(observations)
        goals = self._encode_policy_goals(goals)

        if self.config['pe_type'] == 'frs':
            leading_shape = z_s.shape[:-1]
            n_z_s = jnp.repeat(jnp.expand_dims(z_s, 0), pe_info['num_samples'], axis=0)
            n_goals = jnp.repeat(jnp.expand_dims(goals, 0), pe_info['num_samples'], axis=0)

            n_actions = jax.random.normal(
                seed,
                (
                    pe_info['num_samples'],
                    *leading_shape,
                    self.config['action_dim'],
                ),
            )
            for i in range(pe_info['flow_steps']):
                t = jnp.full(
                    (pe_info['num_samples'], *leading_shape, 1),
                    i / pe_info['flow_steps'],
                )
                vels = self.network.select('actor')(n_z_s, n_goals, n_actions, t)
                n_actions = n_actions + vels / pe_info['flow_steps']
            n_actions = jnp.clip(n_actions, -1, 1)

            q_action = self.network.select('q_action')(
                n_z_s, goals=n_goals, actions=n_actions,
            )

            if len(z_s.shape) == 2:
                actions = n_actions[jnp.argmax(q_action, axis=0), jnp.arange(z_s.shape[0])]
            else:
                actions = n_actions[jnp.argmax(q_action)]

            return actions

        dist = self.network.select('actor')(z_s, goals, temperature=temperature)
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
        raw_observation_shape = tuple(ex_observations.shape[1:])
        action_dim = ex_actions.shape[-1]
        pe_info = cls._get_pe_info_from_config(config)

        encoder_name = config.get('encoder', None)
        if encoder_name in (None, ''):
            encoder_name = None
        if encoder_name is not None:
            encoder_module = encoder_modules[encoder_name]
            encoder_def = encoder_module()
            rng, encoder_init_rng = jax.random.split(rng)
            encoder_params = encoder_def.init(encoder_init_rng, ex_observations, train=False)['params']
            ex_encoded_observations = encoder_def.apply({'params': encoder_params}, ex_observations, train=False)
            if tuple(ex_goals.shape[1:]) == raw_observation_shape or ex_goals.ndim > 2:
                ex_encoded_goals = encoder_def.apply({'params': encoder_params}, ex_goals, train=False)
            else:
                ex_encoded_goals = ex_goals
        else:
            encoder_def = None
            ex_encoded_observations = ex_observations
            ex_encoded_goals = ex_goals
        obs_dim = ex_encoded_observations.shape[-1]
        latent_dtype = ex_encoded_observations.dtype

        ex_z = jnp.zeros((ex_observations.shape[0], config['z_dim']), dtype=latent_dtype)
        ex_times = ex_actions[..., :1]

        # VAE encoder/decoder.
        encoder_hidden_dims = config.get('vae_encoder_hidden_dims', None)
        if encoder_hidden_dims is None:
            encoder_hidden_dims = config.get('vae_hidden_dims', config['value_hidden_dims'])
        decoder_hidden_dims = config.get('vae_decoder_hidden_dims', None)
        if decoder_hidden_dims is None:
            decoder_hidden_dims = config.get('vae_hidden_dims', config['value_hidden_dims'])

        vae_encoder_def = VAEEncoder(
            hidden_dims=encoder_hidden_dims,
            z_dim=config['z_dim'],
            layer_norm=config['layer_norm'],
        )
        if encoder_def is not None and len(raw_observation_shape) == 3:
            vae_decoder_def = ImpalaDecoder(
                output_channels=raw_observation_shape[-1],
                mlp_hidden_dims=decoder_hidden_dims,
            )
        else:
            vae_decoder_def = VAEDecoder(
                hidden_dims=decoder_hidden_dims,
                obs_dim=obs_dim,
                layer_norm=config['layer_norm'],
            )

        # Value and Q networks operate on latent inputs.
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
        # Q_short: latent obs, encoded/raw goal, raw action.
        q_action_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=1,
        )

        # z-proposal: iMF flow in latent space.
        ex_z_times = jnp.zeros((ex_observations.shape[0], 5), dtype=latent_dtype)
        z_proposal_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=config['z_dim'],
            layer_norm=config['layer_norm'],
        )

        # Actor: latent obs, encoded/raw goal, raw action output.
        if config['pe_type'] == 'frs':
            actor_def = ActorVectorField(
                hidden_dims=config['actor_hidden_dims'],
                action_dim=action_dim,
                layer_norm=config['layer_norm'],
            )
            ex_actor_in = (ex_z, ex_encoded_goals, ex_actions, ex_times)
        elif config['pe_type'] == 'discrete':
            actor_def = GCDiscreteActor(
                hidden_dims=config['actor_hidden_dims'],
                action_dim=config['pe_discrete']['action_ct'],
                layer_norm=config['layer_norm'],
            )
            ex_actor_in = (ex_z, ex_encoded_goals, ex_actions)
        elif config['pe_type'] == 'rpg':
            actor_def = GCActor(
                hidden_dims=config['actor_hidden_dims'],
                action_dim=action_dim,
                layer_norm=config['layer_norm'],
                state_dependent_std=False,
                const_std=pe_info['const_std'],
            )
            ex_actor_in = (ex_z, ex_encoded_goals, ex_actions)
        else:
            raise ValueError(f"Unsupported pe_type: {config['pe_type']}")

        network_info = dict(
            vae_encoder=(vae_encoder_def, (ex_encoded_observations,)),
            vae_decoder=(vae_decoder_def, (ex_z,)),
            value=(value_def, (ex_z, ex_z)),
            target_value=(copy.deepcopy(value_def), (ex_z, ex_z)),
            q=(q_def, (ex_z, ex_z, ex_z)),
            target_q=(copy.deepcopy(q_def), (ex_z, ex_z, ex_z)),
            q_action=(q_action_def, (ex_z, ex_encoded_goals, ex_actions)),
            z_proposal=(z_proposal_def, (ex_z, ex_z, ex_z, ex_z_times)),
            actor=(actor_def, ex_actor_in),
        )
        if encoder_def is not None:
            network_info['encoder'] = (encoder_def, (ex_observations,))

        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']

        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network_params
        params['modules_target_value'] = params['modules_value']
        params['modules_target_q'] = params['modules_q']

        config['action_dim'] = action_dim
        config['raw_observation_shape'] = raw_observation_shape
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='vae_trl',
            lr=3e-4,
            batch_size=1024,
            actor_hidden_dims=(1024,) * 4,
            value_hidden_dims=(1024,) * 4,
            vae_hidden_dims=(512, 512),
            vae_encoder_hidden_dims=(1024,) * 4,
            vae_decoder_hidden_dims=(512, 512),
            layer_norm=True,
            discount=0.999,
            tau=0.005,
            lam=0.0,
            expectile=0.7,
            cf_expectile=0.7,
            q_agg='min',
            value_backup_mode='q',
            z_dim=32,
            vae_beta=0.01,
            reg_coef=1.0,
            vae_recon_coef=0.25,
            sample_latent_for_rl=False,
            encoder='',
            freeze_encoder=False,
            z_proposal_coef=1.0,
            cf_num_z_proposals=1,
            cf_burnin_steps=50000,
            cf_warmup_steps=50000,
            intraj_cf_max_target=True,
            local_step_hinge_coef=0.0,
            local_step_hinge_only_cf=True,
            local_value_smoothness_coef=0.0,
            local_value_smoothness_threshold=0.01,
            local_value_smoothness_only_cf=True,
            q_action_stopgrad_encoder=False,
            q_action_n_step=1,
            pe_type='frs',
            frs=ml_collections.ConfigDict(dict(flow_steps=10, num_samples=32)),
            rpg=ml_collections.ConfigDict(dict(alpha=0.03, const_std=True)),
            pe_discrete=ml_collections.ConfigDict(dict(alpha=0.03, action_ct=0)),
            discrete=False,
            dataset_class='GCDataset',
            value_p_curgoal=0.0,
            value_p_trajgoal=0.8,
            value_p_randomgoal=0.2,
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
