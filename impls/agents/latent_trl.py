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
from utils.flows import imf_cfg_loss, imf_loss, imf_one_shot_sample
from utils.latent_regularizers import sliced_epps_pulley_loss
from utils.networks import ActorVectorField, GCActor, GCDiscreteActor, GCValue, MLP


class SubgoalEncoder(nn.Module):
    """Midpoint encoder: s_mid -> z_mid."""

    hidden_dims: Sequence[int]
    layer_norm: bool
    z_dim: int
    @nn.compact
    def __call__(self, x):
        return MLP((*self.hidden_dims, self.z_dim), activate_final=False, layer_norm=self.layer_norm)(x)


class VAEEncoder(nn.Module):
    """MLP encoder producing (mu, log_var) for state latents."""

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
    """MLP decoder for reconstructing state latents back to observations."""

    hidden_dims: Sequence[int]
    obs_dim: int
    layer_norm: bool

    @nn.compact
    def __call__(self, z):
        return MLP((*self.hidden_dims, self.obs_dim), activate_final=False, layer_norm=self.layer_norm)(z)


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

    def _get_reg_coef(self):
        return self.config.get('reg_coef', self.config.get('vae_coef', 1.0))

    @staticmethod
    def _get_stitch_mode_from_config(config):
        """Resolve the mutually exclusive midpoint-stitching mode.

        Preferred config is the string-valued `stitch_mode`. Legacy boolean
        flags remain supported so older launchers and historical configs still
        behave the same.
        """
        stitch_mode = config.get('stitch_mode', '')
        valid_modes = ('latent_q', 'all_latent_q', 'latent_value', 'direct_midpoint')

        legacy_modes = []
        if config.get('direct_midpoint_proposal', False):
            legacy_modes.append('direct_midpoint')
        if config.get('latent_value_stitching', False):
            legacy_modes.append('latent_value')

        if len(legacy_modes) > 1:
            raise ValueError(
                'direct_midpoint_proposal and latent_value_stitching are mutually exclusive. '
                'Use stitch_mode to select a single stitching branch.'
            )

        if stitch_mode not in (None, ''):
            if stitch_mode not in valid_modes:
                raise ValueError(f'Unsupported stitch_mode: {stitch_mode}')
            if legacy_modes and stitch_mode != legacy_modes[0]:
                raise ValueError(
                    f'Conflicting stitch configuration: stitch_mode={stitch_mode} '
                    f'but legacy flag requested {legacy_modes[0]}.'
                )
            return stitch_mode

        if legacy_modes:
            return legacy_modes[0]
        return 'latent_q'

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

    def _encode_policy_inputs(self, observations, goals=None, grad_params=None):
        encoded_observations = self._encode_visual(observations, grad_params=grad_params)
        if goals is None:
            return encoded_observations, None
        # Eval can hand us raw rendered goals with slightly different leading
        # dimensions than the training batch. For visual runs, any non-flat goal
        # tensor should go through the shared encoder.
        if self._has_visual_encoder() and (
            self._matches_raw_observation_shape(goals) or goals.ndim > 2
        ):
            encoded_goals = self._encode_visual(goals, grad_params=grad_params)
        else:
            encoded_goals = goals
        return encoded_observations, encoded_goals

    def _encode_state(self, observations, grad_params=None, rng=None, sample=False):
        """Encode raw states into the VAE state latent space."""
        observations = self._encode_visual(observations, grad_params=grad_params)
        if grad_params is None:
            mu, log_var = self.network.select('vae_encoder')(observations)
        else:
            mu, log_var = self.network.select('vae_encoder')(observations, params=grad_params)

        if sample:
            if rng is None:
                raise ValueError('Sampling VAE state latents requires rng.')
            std = jnp.exp(0.5 * log_var)
            eps = jax.random.normal(rng, mu.shape)
            return mu + std * eps
        return mu

    def _state_vae_loss(self, batch, grad_params, rng):
        """Train a lightweight VAE over state observations used by value learning."""
        state_tensors = [
            self._encode_visual(batch['observations'], grad_params=grad_params),
            self._encode_visual(batch['next_observations'], grad_params=grad_params),
            self._encode_visual(batch['value_midpoint_observations'], grad_params=grad_params),
            self._encode_visual(batch[self._state_goal_key()], grad_params=grad_params),
        ]
        vae_inputs = jnp.concatenate(state_tensors, axis=0)

        mu, log_var = self.network.select('vae_encoder')(vae_inputs, params=grad_params)
        std = jnp.exp(0.5 * log_var)
        eps = jax.random.normal(rng, mu.shape)
        z = mu + std * eps
        recon = self.network.select('vae_decoder')(z, params=grad_params)

        recon_loss = jnp.mean((recon - vae_inputs) ** 2)
        kl_loss = -0.5 * jnp.mean(1.0 + log_var - mu**2 - jnp.exp(log_var))
        total_loss = self.config['vae_recon_coef'] * recon_loss + self.config['vae_beta'] * kl_loss

        return total_loss, {
            'loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'z_mean': mu.mean(),
            'z_std': std.mean(),
        }

    def _latent_value_aux_loss(self, u_s, u_g, target, tau, dist_weight, grad_params):
        """Auxiliary value prediction on midpoint-encoder latents."""
        z_s = self._encode_subgoal(u_s, grad_params=grad_params)
        z_g = self._encode_subgoal(u_g, grad_params=grad_params)
        latent_v_logits = self.network.select('latent_value')(
            z_s,
            goals=z_g,
            params=grad_params,
        )
        latent_vs = jax.nn.sigmoid(latent_v_logits)
        expectile_weight = jnp.where(target >= latent_vs, tau, (1 - tau))
        latent_value_loss = expectile_weight * dist_weight * self.bce_loss(
            latent_v_logits,
            jax.lax.stop_gradient(target),
        )
        return latent_value_loss.mean(), {
            'loss': latent_value_loss.mean(),
            'v_mean': latent_vs.mean(),
            'v_max': latent_vs.max(),
            'v_min': latent_vs.min(),
        }

    def _latent_stitch_targets(self, z_s, z_g, z_mid):
        """Factorized latent-space midpoint backup V_latent(z_s, z_mid) * V_latent(z_mid, z_g)."""
        first_logits = self.network.select('target_latent_value')(z_s, goals=z_mid)
        second_logits = self.network.select('target_latent_value')(z_mid, goals=z_g)
        first_v = self._aggregate_q_ensembles(jax.nn.sigmoid(first_logits))
        second_v = self._aggregate_q_ensembles(jax.nn.sigmoid(second_logits))
        return first_v * second_v

    def _encode_subgoal(self, midpoint_observations, grad_params=None):
        if grad_params is None:
            return self.network.select('subgoal_encoder')(midpoint_observations)
        return self.network.select('subgoal_encoder')(midpoint_observations, params=grad_params)

    def _stitch_mode(self):
        return self._get_stitch_mode_from_config(self.config)

    def _use_latent_value_stitching(self):
        return self._stitch_mode() == 'latent_value'

    def _use_all_latent_q(self):
        return self._stitch_mode() == 'all_latent_q'

    def _q_context_inputs(self, u_s, u_g, grad_params=None):
        if not self._use_all_latent_q():
            return u_s, u_g
        z_s = self._encode_subgoal(u_s, grad_params=grad_params)
        z_g = self._encode_subgoal(u_g, grad_params=grad_params)
        return z_s, z_g

    def _apply_q_network(self, module_name, u_s, u_g, z_mid, grad_params=None):
        q_obs, q_goals = self._q_context_inputs(u_s, u_g, grad_params=grad_params)
        if grad_params is None:
            return self.network.select(module_name)(q_obs, goals=q_goals, actions=z_mid)
        return self.network.select(module_name)(
            q_obs,
            goals=q_goals,
            actions=z_mid,
            params=grad_params,
        )

    def _use_hybrid_state_midpoint(self):
        cutoff = self.config['hybrid_state_midpoint_max_offset']
        return cutoff is not None and cutoff >= 0

    def _use_direct_midpoint_proposal(self):
        return self._stitch_mode() == 'direct_midpoint'

    def _zero_midpoint_latent(self, observations, num_samples=1):
        if num_samples == 1:
            return jnp.zeros((observations.shape[0], self.config['z_dim']), dtype=observations.dtype)
        return jnp.zeros((num_samples, observations.shape[0], self.config['z_dim']), dtype=observations.dtype)

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

    def _has_cf_z_stitching(self, batch):
        if self._use_direct_midpoint_proposal():
            return 'value_goals_is_intraj' in batch and self.config.get('midpoint_decoder_coef', 0.0) > 0
        return 'value_goals_is_intraj' in batch and self.config['z_proposal_coef'] > 0

    def _grounded_midpoint_targets_enabled(self):
        return self.config.get('grounded_midpoint_targets', False)

    def _state_goal_key(self):
        return 'value_goal_observations'

    def _midpoint_goal_key(self):
        return 'value_midpoint_observations'

    def _get_cf_num_z_proposals(self, split=None):
        base = max(1, int(self.config.get('cf_num_z_proposals', 1)))
        if split is None:
            return base

        key = f'cf_num_z_proposals_{split}'
        value = int(self.config.get(key, -1))
        if value < 1:
            return base
        return value

    def _select_value_training_batch(self, batch, step, rng=None):
        """Optionally override the value-side goal source during early warmup.

        When dual value goals are available, this ramps from explicit
        in-trajectory goals toward the dataset's configured mixed value-goal
        distribution, while leaving actor-side goals unchanged.
        """
        warmup_steps = int(self.config.get('value_goal_warmup_steps', 0))
        if warmup_steps <= 0:
            return batch, False, 1.0
        if 'value_goal_observations_intraj' not in batch:
            return batch, False, 1.0

        progress = jnp.clip(step / float(warmup_steps), 0.0, 1.0)
        using_intraj_warmup = progress < 1.0

        selected_batch = dict(batch)

        if 'value_goals_is_intraj' not in batch:
            selected_batch['value_goals'] = jnp.where(
                using_intraj_warmup,
                batch['value_goals_intraj'],
                batch['value_goals'],
            )
            selected_batch['value_goal_observations'] = jnp.where(
                using_intraj_warmup,
                batch['value_goal_observations_intraj'],
                batch['value_goal_observations'],
            )
            if 'value_offsets_intraj' in batch:
                selected_batch['value_offsets'] = jnp.where(
                    using_intraj_warmup,
                    batch['value_offsets_intraj'],
                    batch['value_offsets'],
                )
            return selected_batch, using_intraj_warmup, progress

        rng = self.rng if rng is None else rng
        ramp_rng = jax.random.fold_in(rng, step)
        keep_mixed_random = jax.random.bernoulli(
            ramp_rng, p=progress, shape=batch['value_goals_is_intraj'].shape
        )
        use_mixed_goal = jnp.logical_or(
            batch['value_goals_is_intraj'] > 0.5,
            keep_mixed_random,
        )

        goal_mask = use_mixed_goal[:, None]
        selected_batch['value_goals'] = jnp.where(
            goal_mask,
            batch['value_goals'],
            batch['value_goals_intraj'],
        )
        selected_batch['value_goal_observations'] = jnp.where(
            goal_mask,
            batch['value_goal_observations'],
            batch['value_goal_observations_intraj'],
        )
        if 'value_midpoint_observations_intraj' in batch:
            midpoint_mask = use_mixed_goal[:, None]
            selected_batch['value_midpoint_observations'] = jnp.where(
                midpoint_mask,
                batch['value_midpoint_observations'],
                batch['value_midpoint_observations_intraj'],
            )
        if 'value_midpoint_goals_intraj' in batch:
            midpoint_goal_mask = use_mixed_goal[:, None]
            selected_batch['value_midpoint_goals'] = jnp.where(
                midpoint_goal_mask,
                batch['value_midpoint_goals'],
                batch['value_midpoint_goals_intraj'],
            )
        if 'value_midpoint_actions_intraj' in batch:
            action_mask = use_mixed_goal[:, None]
            selected_batch['value_midpoint_actions'] = jnp.where(
                action_mask,
                batch['value_midpoint_actions'],
                batch['value_midpoint_actions_intraj'],
            )
        if 'value_offsets_intraj' in batch:
            selected_batch['value_offsets'] = jnp.where(
                use_mixed_goal,
                batch['value_offsets'],
                batch['value_offsets_intraj'],
            )
        if 'value_midpoint_offsets_intraj' in batch:
            selected_batch['value_midpoint_offsets'] = jnp.where(
                use_mixed_goal,
                batch['value_midpoint_offsets'],
                batch['value_midpoint_offsets_intraj'],
            )
        selected_batch['value_goals_is_intraj'] = jnp.where(
            use_mixed_goal,
            batch['value_goals_is_intraj'],
            jnp.ones_like(batch['value_goals_is_intraj']),
        )
        return selected_batch, using_intraj_warmup, progress

    def _assert_grounded_goal_space(self, batch, goal_key):
        if batch[goal_key].shape[-1] != batch['observations'].shape[-1]:
            raise ValueError(
                'This path expects raw-state goal inputs for value learning.'
            )

    def _grounded_midpoint_targets(
        self,
        observations,
        goals,
        midpoint_observations,
        midpoint_goals,
        midpoint_offsets,
        goal_offsets,
    ):
        second_offset = goal_offsets[None, ...] - midpoint_offsets

        first_v_logits = self.network.select('target_value')(observations, goals=midpoint_goals)
        second_v_logits = self.network.select('target_value')(midpoint_observations, goals=goals)

        first_v = jnp.where(
            (midpoint_offsets <= 1)[None, ...],
            self.config['discount'] ** midpoint_offsets[None, ...],
            jax.nn.sigmoid(first_v_logits),
        )
        second_v = jnp.where(
            (second_offset <= 1)[None, ...],
            self.config['discount'] ** second_offset[None, ...],
            jax.nn.sigmoid(second_v_logits),
        )
        return first_v * second_v

    def _sample_midpoint_decoder(self, observations, goals, z, rng, num_samples=1):
        """Sample midpoint state latents from the grounded midpoint decoder."""
        del goals
        state_latent_dim = observations.shape[-1]
        observations = jnp.zeros_like(observations)
        decoder_goals = z if num_samples == 1 else None

        if num_samples == 1:
            sample_shape = (observations.shape[0], state_latent_dim)
        else:
            sample_shape = (num_samples, observations.shape[0], state_latent_dim)

        def vector_field_fn(noise, times):
            if num_samples == 1:
                return self.network.select('midpoint_decoder')(
                    observations,
                    goals=decoder_goals,
                    actions=noise,
                    times=times,
                )
            obs_bc = jnp.broadcast_to(observations[None], (num_samples, *observations.shape))
            decoder_goals_bc = z
            obs_flat = obs_bc.reshape(-1, observations.shape[-1])
            goals_flat = decoder_goals_bc.reshape(-1, decoder_goals_bc.shape[-1])
            noise_flat = noise.reshape(-1, state_latent_dim)
            times_flat = times.reshape(-1, times.shape[-1])
            out = self.network.select('midpoint_decoder')(
                obs_flat,
                goals=goals_flat,
                actions=noise_flat,
                times=times_flat,
            )
            return out.reshape(num_samples, observations.shape[0], state_latent_dim)

        return imf_one_shot_sample(rng, sample_shape, vector_field_fn)

    def _proposal_context(self, observations, goals):
        """Optionally drop conditioning so z_proposal becomes an unconditional prior."""
        if self.config.get('unconditional_z_proposal', False):
            return jnp.zeros_like(observations), jnp.zeros_like(goals)
        return observations, goals

    def _midpoint_decoder_context(self, observations, goals):
        """Midpoint generation is unconditional on (s, g) and depends only on z."""
        del goals
        return jnp.zeros_like(observations), None

    def _midpoint_decoder_cfg_enabled(self):
        return False

    def _z_proposal_cfg_enabled(self):
        if self.config.get('z_proposal_coef', 0.0) <= 0.0:
            return False
        if self.config.get('unconditional_z_proposal', False):
            return False
        return (
            self.config.get('z_proposal_cfg_w_min', 1.0) != 1.0
            or self.config.get('z_proposal_cfg_w_max', 1.0) != 1.0
            or self.config.get('z_proposal_cfg_class_dropout_prob', 0.0) > 0.0
        )

    def _actor_proposal_goals(self, goals):
        """Optionally drop goal conditioning from the behavior-cloning actor only."""
        if self.config.get('unconditional_actor_proposal', False):
            return jnp.zeros_like(goals)
        return goals

    def midpoint_decoder_loss(self, batch, grad_params, rng):
        """Train the grounded midpoint decoder: z_mid -> u_mid."""
        u_s = self._encode_state(batch['observations'])
        u_s = jnp.zeros_like(u_s)
        u_target = self._encode_state(batch['value_midpoint_observations'])
        if self._use_direct_midpoint_proposal():
            z_target = self._zero_midpoint_latent(u_target)
        else:
            z_target = self._encode_subgoal(u_target)
        u_target = jax.lax.stop_gradient(u_target)
        z_target = jax.lax.stop_gradient(z_target)
        decoder_goals = z_target
        intraj_mask = batch.get('value_goals_is_intraj', None)

        def cond_vector_field_fn(noise, times):
            return self.network.select('midpoint_decoder')(
                u_s,
                goals=decoder_goals,
                actions=noise,
                times=times,
                params=grad_params,
            )

        loss, flow_info = imf_loss(
            rng,
            u_target,
            cond_vector_field_fn,
            r_equals_t_prob=0.5,
            mask=intraj_mask,
        )

        info = {'loss': loss}
        if intraj_mask is not None:
            info['intraj_frac'] = intraj_mask.mean()
        info['cfg_scale'] = jnp.asarray(1.0, dtype=u_target.dtype)
        info['cfg_w_max'] = jnp.asarray(1.0, dtype=u_target.dtype)
        info['cfg_enabled'] = jnp.asarray(0.0, dtype=u_target.dtype)
        for k, v in flow_info.items():
            if k != 'loss':
                info[k] = v
        return loss, info

    def midpoint_sigreg_loss(self, batch, grad_params, rng):
        """Regularize midpoint latents toward a simple Gaussian geometry."""
        if self._use_direct_midpoint_proposal():
            zero = jnp.asarray(0.0, dtype=batch['observations'].dtype)
            return zero, {
                'loss': zero,
                'skipped': jnp.asarray(1.0, dtype=batch['observations'].dtype),
            }
        midpoint_observations = batch.get('value_midpoint_observations_intraj', batch['value_midpoint_observations'])
        u_mid = self._encode_state(midpoint_observations, grad_params=grad_params)
        z_mid = self._encode_subgoal(u_mid, grad_params=grad_params)
        loss = sliced_epps_pulley_loss(
            z_mid,
            rng,
            num_slices=int(self.config.get('sigreg_num_slices', 32)),
            reduction='mean',
        )
        return loss, {
            'loss': loss,
            'z_mean': z_mid.mean(),
            'z_std': z_mid.std(),
        }

    def q_loss(self, batch, grad_params, rng=None, cf_weight=1.0):
        if self._grounded_midpoint_targets_enabled():
            return self._q_loss_grounded(batch, grad_params, rng=rng, cf_weight=cf_weight)
        return self._q_loss_legacy(batch, grad_params, rng=rng, cf_weight=cf_weight)

    def value_loss(self, batch, grad_params, rng=None, cf_weight=1.0):
        if self._grounded_midpoint_targets_enabled():
            return self._value_loss_grounded(batch, grad_params, rng=rng, cf_weight=cf_weight)
        return self._value_loss_legacy(batch, grad_params, rng=rng, cf_weight=cf_weight)

    def _q_loss_legacy(self, batch, grad_params, rng=None, cf_weight=1.0):
        """Train Q with optional short-horizon state-midpoint fallback.

        Core latent TRL target: Q(s, z_mid, g) <- V(s, z_mid) * V(z_mid, g),
        where z_mid = enc(s_mid).

        When counterfactual z-stitching is active, counterfactual samples use
        z from the iMF proposal and a latent factored backup instead.
        """
        del rng, cf_weight
        goal_key = self._state_goal_key()
        midpoint_goal_key = self._midpoint_goal_key()
        has_cf = self._has_cf_z_stitching(batch)
        if has_cf:
            raise ValueError(
                'Counterfactual stitching in latent_trl now requires grounded_midpoint_targets=True.'
            )

        u_s = self._encode_state(batch['observations'], grad_params=grad_params)
        u_g = self._encode_state(batch[goal_key], grad_params=grad_params)
        u_mid = self._encode_state(batch['value_midpoint_observations'], grad_params=grad_params)

        # --- In-trajectory path (standard) ---
        z_mid = self._encode_subgoal(u_mid, grad_params=grad_params)
        q_latent_logits = self._apply_q_network('q', u_s, u_g, z_mid, grad_params=grad_params)
        q_logits = q_latent_logits
        q_state = None
        if self._use_hybrid_state_midpoint():
            q_state_logits = self.network.select('q_state')(
                u_s,
                goals=u_g,
                actions=u_mid,
                params=grad_params,
            )
            short_mask = self._hybrid_short_mask(batch)
            q_logits = jnp.where(short_mask, q_state_logits, q_latent_logits)
            q_state = jax.nn.sigmoid(q_state_logits)

        second_offset = batch['value_offsets'][None, ...] - batch['value_midpoint_offsets']

        # State-space factors (teacher values over concrete midpoint state s').
        state_first_v_logits = self.network.select('target_value')(
            u_s,
            goals=u_mid,
        )
        state_second_v_logits = self.network.select('target_value')(
            u_mid,
            goals=u_g,
        )

        first_v = jnp.where(
            (batch['value_midpoint_offsets'] <= 1)[None, ...],
            self.config['discount'] ** batch['value_midpoint_offsets'][None, ...],
            jax.nn.sigmoid(state_first_v_logits),
        )
        second_v = jnp.where(
            (second_offset <= 1)[None, ...],
            self.config['discount'] ** second_offset[None, ...],
            jax.nn.sigmoid(state_second_v_logits),
        )
        target = first_v * second_v

        qs = jax.nn.sigmoid(q_logits)

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
        }
        if self._use_hybrid_state_midpoint():
            short_mask = self._hybrid_short_mask(batch)
            info['short_frac'] = short_mask.mean()
            info['q_latent_mean'] = jax.nn.sigmoid(q_latent_logits).mean()
            info['q_state_mean'] = q_state.mean()

        info['loss'] = total_loss
        return total_loss, info

    def _value_loss_legacy(self, batch, grad_params, rng=None, cf_weight=1.0):
        """Value loss with optional multi-z expectile from proposed z candidates."""
        goal_key = self._state_goal_key()
        has_cf = self._has_cf_z_stitching(batch)
        if has_cf:
            raise ValueError(
                'Counterfactual stitching in latent_trl now requires grounded_midpoint_targets=True.'
            )

        u_s = self._encode_state(batch['observations'], grad_params=grad_params)
        u_g = self._encode_state(batch[goal_key], grad_params=grad_params)
        u_mid = self._encode_state(batch['value_midpoint_observations'])

        v_logits = self.network.select('value')(u_s, goals=u_g, params=grad_params)
        vs = jax.nn.sigmoid(v_logits)

        # --- Trajectory midpoint Q target (standard path) ---
        z_mid = self._encode_subgoal(u_mid)
        z_mid = jax.lax.stop_gradient(z_mid)
        q_latent_logits = self._apply_q_network('target_q', u_s, u_g, z_mid)
        q_logits = q_latent_logits
        q_state = None
        if self._use_hybrid_state_midpoint():
            q_state_logits = self.network.select('target_q_state')(
                u_s,
                goals=u_g,
                actions=u_mid,
            )
            short_mask = self._hybrid_short_mask(batch)
            q_logits = jnp.where(short_mask, q_state_logits, q_latent_logits)
            q_state = jax.nn.sigmoid(q_state_logits)
        q_traj = jax.nn.sigmoid(q_logits)
        target_traj = self._aggregate_q_ensembles(q_traj)

        # --- Proposed z target (when z_proposal is enabled) ---
        use_proposal = self.config['z_proposal_coef'] > 0
        if use_proposal:
            # No cf goals, but proposal enabled (gen-midpoint ablation).
            z_proposed = self._sample_z_proposal(batch, rng, num_samples=1)
            z_proposed = jax.lax.stop_gradient(z_proposed)

            q_proposed_logits = self._apply_q_network('target_q', u_s, u_g, z_proposed)
            q_proposed = jax.nn.sigmoid(q_proposed_logits)
            target_proposed = self._aggregate_q_ensembles(q_proposed)
            target = jnp.maximum(target_traj, target_proposed)
        else:
            target = target_traj

        if use_proposal and has_cf:
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

        predicted_steps = jnp.log(vs + 1e-8) / jnp.log(self.config['discount'])
        actual_steps = batch['value_offsets']
        relative_gap = (predicted_steps - actual_steps) / (actual_steps + 1e-8)

        info = {
            'total_loss': total_loss,
            'v_loss': total_loss,
            'v_mean': vs.mean(),
            'v_max': vs.max(),
            'v_min': vs.min(),
            'calibration_rel_gap_mean': relative_gap.mean(),
            'calibration_rel_gap_max': relative_gap.max(),
        }
        if use_proposal:
            info['q_proposed_mean'] = target_proposed.mean()
            info['q_traj_mean'] = target_traj.mean()
        if self._use_hybrid_state_midpoint():
            short_mask = self._hybrid_short_mask(batch)
            info['short_frac'] = short_mask.mean()
            info['q_target_latent_mean'] = jax.nn.sigmoid(q_latent_logits).mean()
            info['q_target_state_mean'] = q_state.mean()
        return total_loss, info

    def _q_loss_grounded(self, batch, grad_params, rng=None, cf_weight=1.0):
        """Train latent Q from grounded midpoint state factors."""
        del cf_weight  # q targets are defined directly from the chosen midpoint.
        if self._use_direct_midpoint_proposal() or self._use_latent_value_stitching():
            zero = jnp.asarray(0.0, dtype=batch['observations'].dtype)
            return zero, {
                'total_loss': zero,
                'q_loss': zero,
                'loss': zero,
                'skipped': jnp.asarray(1.0, dtype=batch['observations'].dtype),
            }
        goal_key = self._state_goal_key()
        has_cf = self._has_cf_z_stitching(batch)

        q_rng, decoder_rng = jax.random.split(rng)
        u_s = self._encode_state(batch['observations'], grad_params=grad_params)
        u_g = self._encode_state(batch[goal_key], grad_params=grad_params)
        u_mid = self._encode_state(batch['value_midpoint_observations'], grad_params=grad_params)
        z_mid = self._encode_subgoal(u_mid, grad_params=grad_params)
        q_logits = self._apply_q_network('q', u_s, u_g, z_mid, grad_params=grad_params)

        intraj_target = self._grounded_midpoint_targets(
            u_s,
            u_g,
            u_mid,
            u_mid,
            batch['value_midpoint_offsets'],
            batch['value_offsets'],
        )

        if has_cf:
            is_intraj = batch['value_goals_is_intraj']
            is_intraj_2d = is_intraj[None, ...]

            z_cf = self._sample_z_proposal(batch, q_rng, num_samples=1)
            z_cf = jax.lax.stop_gradient(z_cf)
            m_cf = self._sample_midpoint_decoder(
                u_s,
                u_g,
                z_cf,
                decoder_rng,
                num_samples=1,
            )
            m_cf = jax.lax.stop_gradient(m_cf)
            if self.config.get('latent_round_trip', False):
                z_cf_q = self._encode_subgoal(m_cf, grad_params=grad_params)
            else:
                z_cf_q = z_cf

            q_cf_logits = self._apply_q_network('q', u_s, u_g, z_cf_q, grad_params=grad_params)
            cf_target = self._grounded_midpoint_targets(
                u_s,
                u_g,
                m_cf,
                m_cf,
                jnp.full_like(batch['value_offsets'], 2),
                jnp.full_like(batch['value_offsets'], 4),
            )

            q_logits = jnp.where(is_intraj_2d, q_logits, q_cf_logits)
            target = jnp.where(is_intraj_2d, intraj_target, cf_target)
        else:
            target = intraj_target

        qs = jax.nn.sigmoid(q_logits)
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
            'grounded_target_mean': target.mean(),
            'intraj_target_mean': intraj_target.mean(),
        }
        if has_cf:
            info['cf_target_mean'] = cf_target.mean()
            info['cf_frac'] = (1.0 - is_intraj).mean()
            if self.config.get('latent_round_trip', False):
                info['cf_roundtrip_l2_mean'] = jnp.linalg.norm(z_cf_q - z_cf, axis=-1).mean()
        return total_loss, info

    def _value_loss_grounded(self, batch, grad_params, rng=None, cf_weight=1.0):
        """Use direct TRL targets locally and latent-Q stitching for longer horizons."""
        goal_key = self._state_goal_key()
        has_cf = self._has_cf_z_stitching(batch)
        intraj_direct_only_cf = self.config.get('intraj_direct_only_cf', False)
        direct_midpoint_proposal = self._use_direct_midpoint_proposal()
        latent_value_stitching = self._use_latent_value_stitching()

        u_s = self._encode_state(batch['observations'], grad_params=grad_params)
        u_g = self._encode_state(batch[goal_key], grad_params=grad_params)
        u_mid = self._encode_state(batch['value_midpoint_observations'], grad_params=grad_params)

        v_logits = self.network.select('value')(u_s, goals=u_g, params=grad_params)
        vs = jax.nn.sigmoid(v_logits)

        direct_target_ens = self._grounded_midpoint_targets(
            u_s,
            u_g,
            u_mid,
            u_mid,
            batch['value_midpoint_offsets'],
            batch['value_offsets'],
        )
        direct_target = self._aggregate_q_ensembles(direct_target_ens)

        if direct_midpoint_proposal:
            direct_mask = None
            target_traj = direct_target
            target_traj_q = None
            target_traj_latent = None
        else:
            z_mid = self._encode_subgoal(u_mid)
            z_mid = jax.lax.stop_gradient(z_mid)
            if latent_value_stitching:
                z_s = self._encode_subgoal(u_s)
                z_s = jax.lax.stop_gradient(z_s)
                z_g = self._encode_subgoal(u_g)
                z_g = jax.lax.stop_gradient(z_g)
                target_traj_latent = self._latent_stitch_targets(z_s, z_g, z_mid)
                target_traj_q = None
            else:
                q_traj_logits = self._apply_q_network('target_q', u_s, u_g, z_mid)
                q_traj = jax.nn.sigmoid(q_traj_logits)
                target_traj_q = self._aggregate_q_ensembles(q_traj)
                target_traj_latent = None

            if intraj_direct_only_cf:
                direct_mask = None
                target_traj = direct_target
            else:
                direct_cutoff = self.config.get('direct_intraj_value_max_offset', -1)
                if direct_cutoff is not None and direct_cutoff >= 0:
                    direct_mask = batch['value_offsets'] <= direct_cutoff
                    backing_target = target_traj_latent if latent_value_stitching else target_traj_q
                    target_traj = jnp.where(direct_mask, direct_target, backing_target)
                else:
                    direct_mask = None
                    target_traj = target_traj_latent if latent_value_stitching else target_traj_q

        if direct_midpoint_proposal:
            use_proposal = self._grounded_midpoint_targets_enabled() and self.config.get('midpoint_decoder_coef', 0.0) > 0
        else:
            use_proposal = self.config['z_proposal_coef'] > 0

        if direct_midpoint_proposal and use_proposal and has_cf:
            is_intraj = batch['value_goals_is_intraj']
            num_cf_samples_intraj = self._get_cf_num_z_proposals('intraj')
            num_cf_samples_random = self._get_cf_num_z_proposals('random')
            num_cf_samples = max(num_cf_samples_intraj, num_cf_samples_random)
            z_proposed = self._zero_midpoint_latent(u_s, num_cf_samples)
            w_proposed = self._sample_midpoint_decoder(u_s, u_g, z_proposed, rng, num_samples=num_cf_samples)
            w_proposed = jax.lax.stop_gradient(w_proposed)

            if num_cf_samples == 1:
                proposed_target_ens = self._grounded_midpoint_targets(
                    u_s,
                    u_g,
                    w_proposed,
                    w_proposed,
                    jnp.full_like(batch['value_offsets'], 2),
                    jnp.full_like(batch['value_offsets'], 4),
                )
                target_proposed = self._aggregate_q_ensembles(proposed_target_ens)
            else:
                obs_bc = jnp.broadcast_to(u_s[None], (num_cf_samples, *u_s.shape))
                goals_bc = jnp.broadcast_to(u_g[None], (num_cf_samples, *u_g.shape))
                proposed_target_ens = self._grounded_midpoint_targets(
                    obs_bc.reshape(-1, u_s.shape[-1]),
                    goals_bc.reshape(-1, u_g.shape[-1]),
                    w_proposed.reshape(-1, w_proposed.shape[-1]),
                    w_proposed.reshape(-1, w_proposed.shape[-1]),
                    jnp.full((num_cf_samples * u_s.shape[0],), 2, dtype=batch['value_offsets'].dtype),
                    jnp.full((num_cf_samples * u_s.shape[0],), 4, dtype=batch['value_offsets'].dtype),
                )
                proposed_values = self._aggregate_q_ensembles(proposed_target_ens).reshape(num_cf_samples, -1)
                proposal_idx = jnp.arange(num_cf_samples)[:, None]
                intraj_mask = proposal_idx < num_cf_samples_intraj
                random_mask = proposal_idx < num_cf_samples_random
                target_proposed_intraj = jnp.where(
                    intraj_mask, proposed_values, -jnp.inf
                ).max(axis=0)
                target_proposed_random = jnp.where(
                    random_mask, proposed_values, -jnp.inf
                ).max(axis=0)
                target_proposed = jnp.where(is_intraj, target_proposed_intraj, target_proposed_random)

            if self.config.get('intraj_cf_max_target', True):
                intraj_target = jnp.maximum(target_traj, cf_weight * target_proposed)
            else:
                intraj_target = target_traj
            cf_target_val = cf_weight * target_proposed
            target = jnp.where(is_intraj, intraj_target, cf_target_val)
        elif direct_midpoint_proposal and use_proposal:
            num_cf_samples = self._get_cf_num_z_proposals()
            z_proposed = self._zero_midpoint_latent(u_s, num_cf_samples)
            w_proposed = self._sample_midpoint_decoder(u_s, u_g, z_proposed, rng, num_samples=num_cf_samples)
            w_proposed = jax.lax.stop_gradient(w_proposed)
            if num_cf_samples == 1:
                proposed_target_ens = self._grounded_midpoint_targets(
                    u_s,
                    u_g,
                    w_proposed,
                    w_proposed,
                    jnp.full_like(batch['value_offsets'], 2),
                    jnp.full_like(batch['value_offsets'], 4),
                )
                target_proposed = self._aggregate_q_ensembles(proposed_target_ens)
            else:
                obs_bc = jnp.broadcast_to(u_s[None], (num_cf_samples, *u_s.shape))
                goals_bc = jnp.broadcast_to(u_g[None], (num_cf_samples, *u_g.shape))
                proposed_target_ens = self._grounded_midpoint_targets(
                    obs_bc.reshape(-1, u_s.shape[-1]),
                    goals_bc.reshape(-1, u_g.shape[-1]),
                    w_proposed.reshape(-1, w_proposed.shape[-1]),
                    w_proposed.reshape(-1, w_proposed.shape[-1]),
                    jnp.full((num_cf_samples * u_s.shape[0],), 2, dtype=batch['value_offsets'].dtype),
                    jnp.full((num_cf_samples * u_s.shape[0],), 4, dtype=batch['value_offsets'].dtype),
                )
                target_proposed = self._aggregate_q_ensembles(proposed_target_ens).reshape(num_cf_samples, -1).max(axis=0)
            target = jnp.maximum(target_traj, target_proposed)
        elif use_proposal and has_cf:
            is_intraj = batch['value_goals_is_intraj']
            num_cf_samples_intraj = self._get_cf_num_z_proposals('intraj')
            num_cf_samples_random = self._get_cf_num_z_proposals('random')
            num_cf_samples = max(num_cf_samples_intraj, num_cf_samples_random)
            z_proposed = self._sample_z_proposal(batch, rng, num_samples=num_cf_samples)
            z_proposed = jax.lax.stop_gradient(z_proposed)

            if latent_value_stitching:
                z_s = self._encode_subgoal(u_s)
                z_s = jax.lax.stop_gradient(z_s)
                z_g = self._encode_subgoal(u_g)
                z_g = jax.lax.stop_gradient(z_g)
                if num_cf_samples == 1:
                    target_proposed = self._latent_stitch_targets(z_s, z_g, z_proposed)
                else:
                    z_s_bc = jnp.broadcast_to(z_s[None], (num_cf_samples, *z_s.shape))
                    z_g_bc = jnp.broadcast_to(z_g[None], (num_cf_samples, *z_g.shape))
                    latent_values = self._latent_stitch_targets(
                        z_s_bc.reshape(-1, z_s.shape[-1]),
                        z_g_bc.reshape(-1, z_g.shape[-1]),
                        z_proposed.reshape(-1, z_proposed.shape[-1]),
                    ).reshape(num_cf_samples, -1)
                    proposal_idx = jnp.arange(num_cf_samples)[:, None]
                    intraj_mask = proposal_idx < num_cf_samples_intraj
                    random_mask = proposal_idx < num_cf_samples_random
                    target_proposed_intraj = jnp.where(
                        intraj_mask, latent_values, -jnp.inf
                    ).max(axis=0)
                    target_proposed_random = jnp.where(
                        random_mask, latent_values, -jnp.inf
                    ).max(axis=0)
                    target_proposed = jnp.where(is_intraj, target_proposed_intraj, target_proposed_random)
            else:
                if num_cf_samples == 1:
                    q_proposed_logits = self._apply_q_network('target_q', u_s, u_g, z_proposed)
                    q_proposed = jax.nn.sigmoid(q_proposed_logits)
                    target_proposed = self._aggregate_q_ensembles(q_proposed)
                else:
                    obs_bc = jnp.broadcast_to(u_s[None], (num_cf_samples, *u_s.shape))
                    goals_bc = jnp.broadcast_to(u_g[None], (num_cf_samples, *u_g.shape))
                    q_proposed_logits = self._apply_q_network(
                        'target_q',
                        obs_bc.reshape(-1, u_s.shape[-1]),
                        goals_bc.reshape(-1, u_g.shape[-1]),
                        z_proposed.reshape(-1, z_proposed.shape[-1]),
                    )
                    q_proposed = jax.nn.sigmoid(q_proposed_logits)
                    q_proposed_values = self._aggregate_q_ensembles(q_proposed).reshape(num_cf_samples, -1)
                    proposal_idx = jnp.arange(num_cf_samples)[:, None]
                    intraj_mask = proposal_idx < num_cf_samples_intraj
                    random_mask = proposal_idx < num_cf_samples_random
                    target_proposed_intraj = jnp.where(
                        intraj_mask, q_proposed_values, -jnp.inf
                    ).max(axis=0)
                    target_proposed_random = jnp.where(
                        random_mask, q_proposed_values, -jnp.inf
                    ).max(axis=0)
                    target_proposed = jnp.where(is_intraj, target_proposed_intraj, target_proposed_random)

            if self.config.get('intraj_cf_max_target', True):
                intraj_target = jnp.maximum(target_traj, cf_weight * target_proposed)
            else:
                intraj_target = target_traj
            cf_target_val = cf_weight * target_proposed
            target = jnp.where(is_intraj, intraj_target, cf_target_val)
        elif use_proposal:
            num_cf_samples = self._get_cf_num_z_proposals()
            z_proposed = self._sample_z_proposal(batch, rng, num_samples=num_cf_samples)
            z_proposed = jax.lax.stop_gradient(z_proposed)
            if latent_value_stitching:
                z_s = self._encode_subgoal(u_s)
                z_s = jax.lax.stop_gradient(z_s)
                z_g = self._encode_subgoal(u_g)
                z_g = jax.lax.stop_gradient(z_g)
                if num_cf_samples == 1:
                    target_proposed = self._latent_stitch_targets(z_s, z_g, z_proposed)
                else:
                    z_s_bc = jnp.broadcast_to(z_s[None], (num_cf_samples, *z_s.shape))
                    z_g_bc = jnp.broadcast_to(z_g[None], (num_cf_samples, *z_g.shape))
                    target_proposed = self._latent_stitch_targets(
                        z_s_bc.reshape(-1, z_s.shape[-1]),
                        z_g_bc.reshape(-1, z_g.shape[-1]),
                        z_proposed.reshape(-1, z_proposed.shape[-1]),
                    ).reshape(num_cf_samples, -1).max(axis=0)
            else:
                if num_cf_samples == 1:
                    q_proposed_logits = self._apply_q_network('target_q', u_s, u_g, z_proposed)
                    q_proposed = jax.nn.sigmoid(q_proposed_logits)
                    target_proposed = self._aggregate_q_ensembles(q_proposed)
                else:
                    obs_bc = jnp.broadcast_to(u_s[None], (num_cf_samples, *u_s.shape))
                    goals_bc = jnp.broadcast_to(u_g[None], (num_cf_samples, *u_g.shape))
                    q_proposed_logits = self._apply_q_network(
                        'target_q',
                        obs_bc.reshape(-1, u_s.shape[-1]),
                        goals_bc.reshape(-1, u_g.shape[-1]),
                        z_proposed.reshape(-1, z_proposed.shape[-1]),
                    )
                    q_proposed = jax.nn.sigmoid(q_proposed_logits)
                    target_proposed = self._aggregate_q_ensembles(q_proposed).reshape(num_cf_samples, -1).max(axis=0)
            target = jnp.maximum(target_traj, target_proposed)
        else:
            target = target_traj

        if has_cf:
            tau = jnp.where(
                batch['value_goals_is_intraj'],
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

        latent_value_loss = jnp.asarray(0.0, dtype=total_loss.dtype)
        latent_value_info = {}
        if self.config.get('latent_value_coef', 0.0) > 0:
            latent_value_loss, latent_value_info = self._latent_value_aux_loss(
                u_s,
                u_g,
                target,
                tau,
                dist_weight,
                grad_params,
            )

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
            'v_mean': vs.mean(),
            'v_max': vs.max(),
            'v_min': vs.min(),
            'calibration_rel_gap_mean': calib_mean,
            'calibration_rel_gap_max': calib_max,
            'backup_traj_mean': target_traj.mean(),
            'direct_target_mean': direct_target.mean(),
        }
        if target_traj_q is not None:
            info['q_traj_mean'] = target_traj_q.mean()
        if target_traj_latent is not None:
            info['latent_traj_mean'] = target_traj_latent.mean()
        if direct_mask is not None:
            if has_cf:
                info['direct_frac'] = (direct_mask * batch['value_goals_is_intraj']).sum() / jnp.maximum(
                    batch['value_goals_is_intraj'].sum(), 1.0
                )
            else:
                info['direct_frac'] = direct_mask.mean()
        elif intraj_direct_only_cf:
            info['direct_frac'] = 1.0
        if use_proposal:
            info['q_proposed_mean'] = target_proposed.mean()
        if self.config.get('latent_value_coef', 0.0) > 0:
            info['latent_value_loss'] = latent_value_loss
            for k, v in latent_value_info.items():
                info[f'latent_value_{k}'] = v
        return total_loss, info

    def z_proposal_loss(self, batch, grad_params, rng):
        """Train iMF z-proposal network: f(noise; u_s, u_g) -> z, supervised by enc(u_mid).

        By default this only uses in-trajectory midpoint supervision.
        Optionally adds a conservative random-goal proposal-improvement term:
        sample K proposals, score with target Q, and imitate the best proposal
        only when it has positive advantage over V(s, g).
        """
        if self._use_direct_midpoint_proposal():
            zero = jnp.asarray(0.0, dtype=batch['observations'].dtype)
            return zero, {
                'loss': zero,
                'skipped': jnp.asarray(1.0, dtype=batch['observations'].dtype),
            }
        goal_key = self._state_goal_key()
        rng, improve_rng = jax.random.split(rng)

        u_target = self._encode_state(batch['value_midpoint_observations'])
        z_target = self._encode_subgoal(u_target)
        z_target = jax.lax.stop_gradient(z_target)

        observations = self._encode_state(batch['observations'])
        goals = self._encode_state(batch[goal_key])
        cond_observations, cond_goals = self._proposal_context(observations, goals)

        def vector_field_fn(noise, times):
            return self.network.select('z_proposal')(
                cond_observations, goals=cond_goals, actions=noise, times=times,
                params=grad_params,
            )

        def uncond_vector_field_fn(noise, times):
            return self.network.select('z_proposal')(
                jnp.zeros_like(observations),
                goals=jnp.zeros_like(goals),
                actions=noise,
                times=times,
                params=grad_params,
            )

        # Mask to in-trajectory samples only (cf midpoint targets are garbage).
        intraj_mask = batch.get('value_goals_is_intraj', None)

        if self._z_proposal_cfg_enabled():
            loss, flow_info = imf_cfg_loss(
                rng,
                z_target,
                vector_field_fn,
                uncond_vector_field_fn,
                r_equals_t_prob=0.5,
                w_min=self.config.get('z_proposal_cfg_w_min', 1.0),
                w_max=self.config.get('z_proposal_cfg_w_max', 1.0),
                cfg_beta=self.config.get('z_proposal_cfg_beta', None),
                class_dropout_prob=self.config.get('z_proposal_cfg_class_dropout_prob', 0.1),
                mask=intraj_mask,
            )
        else:
            loss, flow_info = imf_loss(
                rng, z_target, vector_field_fn,
                r_equals_t_prob=0.5,
                mask=intraj_mask,
            )

        total_loss = loss
        info = {'loss': loss}
        if intraj_mask is not None:
            info['intraj_frac'] = intraj_mask.mean()
        info['cfg_scale'] = jnp.asarray(self.config.get('z_proposal_cfg_scale', 1.0), dtype=z_target.dtype)
        info['cfg_w_max'] = jnp.asarray(self.config.get('z_proposal_cfg_w_max', 1.0), dtype=z_target.dtype)
        info['cfg_enabled'] = jnp.asarray(float(self._z_proposal_cfg_enabled()), dtype=z_target.dtype)

        for k, v in flow_info.items():
            if k != 'loss':
                info[k] = v

        improve_coef = self.config.get('proposal_improvement_coef', 0.0)
        has_cf = 'value_goals_is_intraj' in batch
        if improve_coef > 0.0 and has_cf:
            random_mask = 1.0 - batch['value_goals_is_intraj']
            num_samples = max(1, int(self.config.get('proposal_improvement_num_samples', 8)))

            z_samples = self._sample_z_proposal(batch, improve_rng, num_samples=num_samples)
            z_samples = jax.lax.stop_gradient(z_samples)

            if num_samples == 1:
                q_logits = self._apply_q_network('target_q', observations, goals, z_samples)
                q_vals = self._aggregate_q_ensembles(jax.nn.sigmoid(q_logits))[None, ...]
                z_best = z_samples
            else:
                obs_bc = jnp.broadcast_to(observations[None], (num_samples, *observations.shape))
                goals_bc = jnp.broadcast_to(goals[None], (num_samples, *goals.shape))
                q_logits = self._apply_q_network(
                    'target_q',
                    obs_bc.reshape(-1, observations.shape[-1]),
                    goals_bc.reshape(-1, goals.shape[-1]),
                    z_samples.reshape(-1, z_samples.shape[-1]),
                )
                q_vals = self._aggregate_q_ensembles(jax.nn.sigmoid(q_logits)).reshape(num_samples, -1)
                best_idx = jnp.argmax(q_vals, axis=0)
                z_best = jnp.take_along_axis(
                    z_samples,
                    best_idx[None, :, None],
                    axis=0,
                )[0]

            best_q = q_vals.max(axis=0)
            v_logits = self.network.select('target_value')(observations, goals=goals)
            v_base = self._aggregate_q_ensembles(jax.nn.sigmoid(v_logits))
            improve_adv = best_q - v_base
            improve_mask = random_mask * (improve_adv > 0).astype(random_mask.dtype)

            improve_loss, improve_info = imf_loss(
                improve_rng,
                jax.lax.stop_gradient(z_best),
                vector_field_fn,
                r_equals_t_prob=0.5,
                mask=improve_mask,
            )
            total_loss = total_loss + improve_coef * improve_loss

            info['improve_loss'] = improve_loss
            info['improve_frac'] = improve_mask.mean()
            info['improve_adv_mean'] = jnp.where(
                improve_mask.sum() > 0,
                (improve_adv * improve_mask).sum() / jnp.maximum(improve_mask.sum(), 1.0),
                0.0,
            )
            info['improve_best_q_mean'] = jnp.where(
                random_mask.sum() > 0,
                (best_q * random_mask).sum() / jnp.maximum(random_mask.sum(), 1.0),
                0.0,
            )
            info['improve_v_mean'] = jnp.where(
                random_mask.sum() > 0,
                (v_base * random_mask).sum() / jnp.maximum(random_mask.sum(), 1.0),
                0.0,
            )
            for k, v in improve_info.items():
                if k != 'loss':
                    info[f'improve_{k}'] = v

        return total_loss, info

    def _sample_z_proposal(self, batch, rng, num_samples=1):
        """Sample z candidates from the z-proposal network via iMF one-shot."""
        goal_key = self._state_goal_key()
        observations = self._encode_state(batch['observations'])
        goals = self._encode_state(batch[goal_key])
        observations, goals = self._proposal_context(observations, goals)
        z_dim = self.config['z_dim']
        cfg_scale = self.config.get('z_proposal_cfg_scale', 1.0)

        if num_samples == 1:
            sample_shape = (observations.shape[0], z_dim)
        else:
            sample_shape = (num_samples, observations.shape[0], z_dim)

        def vector_field_fn(noise, times):
            if num_samples == 1:
                return self.network.select('z_proposal')(
                    observations, goals=goals, actions=noise, times=times,
                )
            else:
                # Broadcast observations/goals to match (num_samples, batch, ...)
                obs_bc = jnp.broadcast_to(observations[None], (num_samples, *observations.shape))
                goals_bc = jnp.broadcast_to(goals[None], (num_samples, *goals.shape))
                obs_flat = obs_bc.reshape(-1, observations.shape[-1])
                goals_flat = goals_bc.reshape(-1, goals.shape[-1])
                noise_flat = noise.reshape(-1, z_dim)
                times_flat = times.reshape(-1, times.shape[-1])
                out = self.network.select('z_proposal')(
                    obs_flat, goals=goals_flat, actions=noise_flat, times=times_flat,
                )
                return out.reshape(num_samples, observations.shape[0], z_dim)

        sample_w = cfg_scale if self._z_proposal_cfg_enabled() else 1.0
        z_samples = imf_one_shot_sample(rng, sample_shape, vector_field_fn, w=sample_w)
        return z_samples

    def q_short_loss(self, batch, grad_params):
        """Distill Q_short(s, g, a) from an n-step bootstrap into V for FRS action selection."""
        q_short_goal_key = 'actor_goals'
        target_goal_key = 'actor_goal_observations'
        q_short_n_step = int(self.config.get('q_short_n_step', 1))
        next_obs_key = 'actor_nstep_observations' if q_short_n_step > 1 and 'actor_nstep_observations' in batch else 'next_observations'
        u_next = self._encode_state(batch[next_obs_key])
        u_g = self._encode_state(batch[target_goal_key])

        v_next_logits = self.network.select('target_value')(
            u_next,
            goals=u_g,
        )
        v_next = jax.nn.sigmoid(v_next_logits)
        v_next_min = jnp.minimum(v_next[0], v_next[1])
        if q_short_n_step > 1 and 'actor_nstep_steps' in batch:
            n_steps = batch['actor_nstep_steps']
        else:
            n_steps = jnp.ones_like(batch['actions'][..., 0], dtype=jnp.int32)
        bootstrap_target = (self.config['discount'] ** n_steps) * v_next_min

        if q_short_n_step > 1 and 'actor_goals_is_intraj' in batch and 'actor_goal_offsets' in batch:
            actor_is_intraj = batch['actor_goals_is_intraj']
            actor_goal_offsets = batch['actor_goal_offsets']
            exact_reached = actor_is_intraj * (actor_goal_offsets <= n_steps)
            exact_target = self.config['discount'] ** actor_goal_offsets
            target = jnp.where(exact_reached > 0, exact_target, bootstrap_target)
        else:
            exact_reached = jnp.zeros_like(bootstrap_target)
            target = bootstrap_target

        q_short_observations, q_short_goals = self._encode_policy_inputs(
            batch['observations'],
            batch[q_short_goal_key],
            grad_params=grad_params,
        )
        q_short_logits = self.network.select('q_short')(
            q_short_observations,
            goals=q_short_goals,
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
            'n_step': jnp.asarray(q_short_n_step, dtype=jnp.float32),
            'exact_reached_frac': exact_reached.mean(),
        }

    def actor_loss(self, batch, grad_params, rng=None):
        """Flat actor loss (same policy extraction interface as TRL)."""
        pe_info = self._get_pe_info()

        if self.config['pe_type'] == 'rpg':
            actor_observations, actor_goals = self._encode_policy_inputs(
                batch['observations'],
                batch['actor_goals'],
                grad_params=grad_params,
            )
            actor_goals = self._actor_proposal_goals(actor_goals)
            dist = self.network.select('actor')(
                actor_observations,
                actor_goals,
                params=grad_params,
            )

            actor_goal_key = 'actor_goal_observations'
            u_next = self._encode_state(batch['next_observations'])
            u_g = self._encode_state(batch[actor_goal_key])
            v_next = self.network.select('value')(
                u_next,
                goals=u_g,
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
            actor_observations, actor_goals = self._encode_policy_inputs(
                batch['observations'],
                batch['actor_goals'],
                grad_params=grad_params,
            )
            actor_goals = self._actor_proposal_goals(actor_goals)
            dist = self.network.select('actor')(
                actor_observations,
                actor_goals,
                params=grad_params,
            )

            actor_goal_key = 'actor_goal_observations'
            u_next = self._encode_state(batch['next_observations'])
            u_g = self._encode_state(batch[actor_goal_key])
            v = self.network.select('value')(
                u_next,
                goals=u_g,
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
            actor_observations, actor_goals = self._encode_policy_inputs(
                batch['observations'],
                batch['actor_goals'],
                grad_params=grad_params,
            )
            actor_goals = self._actor_proposal_goals(actor_goals)

            x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
            x_1 = batch['actions']
            t = jax.random.uniform(t_rng, (batch_size, 1))
            x_t = (1 - t) * x_0 + t * x_1
            y = x_1 - x_0

            pred = self.network.select('actor')(
                actor_observations,
                actor_goals,
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
    def total_loss(self, batch, grad_params, rng=None, step=0):
        info = {}
        rng = rng if rng is not None else self.rng
        rng, vae_rng, value_rng, q_rng, actor_rng, z_proposal_rng, midpoint_decoder_rng = jax.random.split(rng, 7)

        value_batch, using_intraj_warmup, value_goal_warmup_progress = self._select_value_training_batch(
            batch, step, rng=value_rng
        )
        info['value_goal_warmup_active'] = jnp.asarray(using_intraj_warmup, dtype=jnp.float32)
        info['value_goal_warmup_progress'] = jnp.asarray(value_goal_warmup_progress, dtype=jnp.float32)

        vae_loss, vae_info = self._state_vae_loss(value_batch, grad_params, rng=vae_rng)
        for k, v in vae_info.items():
            info[f'vae/{k}'] = v

        cf_weight = 1.0
        info['cf_weight'] = jnp.asarray(cf_weight, dtype=jnp.float32)

        value_loss, value_info = self.value_loss(value_batch, grad_params, rng=value_rng, cf_weight=cf_weight)
        for k, v in value_info.items():
            if k.startswith('latent_value_'):
                continue
            info[f'value/{k}'] = v

        q_loss, q_info = self.q_loss(value_batch, grad_params, rng=q_rng, cf_weight=cf_weight)
        for k, v in q_info.items():
            info[f'q/{k}'] = v

        z_proposal_loss = 0.0
        if self.config['z_proposal_coef'] > 0:
            z_proposal_loss, z_proposal_info = self.z_proposal_loss(value_batch, grad_params, rng=z_proposal_rng)
            for k, v in z_proposal_info.items():
                info[f'z_proposal/{k}'] = v

        midpoint_decoder_loss = 0.0
        if self._grounded_midpoint_targets_enabled() and self.config.get('midpoint_decoder_coef', 0.0) > 0:
            midpoint_decoder_loss, midpoint_decoder_info = self.midpoint_decoder_loss(
                value_batch, grad_params, rng=midpoint_decoder_rng
            )
            for k, v in midpoint_decoder_info.items():
                info[f'midpoint_decoder/{k}'] = v

        midpoint_sigreg_loss = 0.0
        if self.config.get('sigreg_coef', 0.0) > 0:
            midpoint_sigreg_loss, midpoint_sigreg_info = self.midpoint_sigreg_loss(
                value_batch, grad_params, rng=midpoint_decoder_rng
            )
            for k, v in midpoint_sigreg_info.items():
                info[f'sigreg/{k}'] = v

        q_short_loss, q_short_info = self.q_short_loss(batch, grad_params)
        for k, v in q_short_info.items():
            info[f'q_short/{k}'] = v

        actor_loss, actor_info = self.actor_loss(batch, grad_params, rng=actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        latent_value_loss = jnp.asarray(0.0, dtype=jnp.float32)
        if self.config.get('latent_value_coef', 0.0) > 0:
            latent_value_loss = value_info['latent_value_loss']
            for k, v in value_info.items():
                if k.startswith('latent_value_'):
                    info[f'latent_value/{k[len("latent_value_"):]}'] = v

        loss = (
            self._get_reg_coef() * vae_loss
            + value_loss
            + q_loss
            + self.config.get('latent_value_coef', 0.0) * latent_value_loss
            + self.config['z_proposal_coef'] * z_proposal_loss
            + self.config.get('midpoint_decoder_coef', 0.0) * midpoint_decoder_loss
            + self.config.get('sigreg_coef', 0.0) * midpoint_sigreg_loss
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
    def update(self, batch, step=0):
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng, step=step)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, 'value')
        self.target_update(new_network, 'q')
        self.target_update(new_network, 'q_state')
        self.target_update(new_network, 'latent_value')

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
        observations, goals = self._encode_policy_inputs(observations, goals)

        if self.config['pe_type'] == 'frs':
            n_observations = jnp.repeat(jnp.expand_dims(observations, 0), pe_info['num_samples'], axis=0)
            n_goals = jnp.repeat(jnp.expand_dims(goals, 0), pe_info['num_samples'], axis=0)
            n_actor_goals = self._actor_proposal_goals(n_goals)

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
                vels = self.network.select('actor')(n_observations, n_actor_goals, n_actions, t)
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

        actor_goals = self._actor_proposal_goals(goals)
        dist = self.network.select('actor')(observations, actor_goals, temperature=temperature)
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
        ex_value_goals = example_batch.get('value_goal_observations', ex_observations)
        ex_times = ex_actions[..., :1]
        action_dim = ex_actions.shape[-1]
        pe_info = cls._get_pe_info_from_config(config)
        stitch_mode = cls._get_stitch_mode_from_config(config)

        raw_observation_shape = tuple(ex_observations.shape[1:])
        encoder_name = config.get('encoder', None)
        if encoder_name in (None, ''):
            encoder_name = None
        if encoder_name is not None:
            encoder_module = encoder_modules[encoder_name]
            encoder_def = encoder_module()
            rng, encoder_init_rng = jax.random.split(rng)
            encoder_params = encoder_def.init(encoder_init_rng, ex_observations, train=False)['params']
            ex_encoded_observations = encoder_def.apply({'params': encoder_params}, ex_observations, train=False)
            if tuple(ex_goals.shape[1:]) == raw_observation_shape:
                ex_encoded_goals = encoder_def.apply({'params': encoder_params}, ex_goals, train=False)
            else:
                ex_encoded_goals = ex_goals
        else:
            encoder_def = None
            ex_encoded_observations = ex_observations
            ex_encoded_goals = ex_goals
        obs_dim = ex_encoded_observations.shape[-1]

        latent_dtype = ex_encoded_observations.dtype
        ex_u = jnp.zeros((ex_observations.shape[0], config['state_z_dim']), dtype=latent_dtype)
        ex_z = jnp.zeros((ex_observations.shape[0], config['z_dim']), dtype=latent_dtype)
        ex_q_obs = ex_z if stitch_mode == 'all_latent_q' else ex_u
        ex_q_goals = ex_z if stitch_mode == 'all_latent_q' else ex_u

        vae_encoder_hidden_dims = config.get('vae_encoder_hidden_dims', config['value_hidden_dims'])
        vae_decoder_hidden_dims = config.get('vae_decoder_hidden_dims', config['actor_hidden_dims'])
        vae_encoder_def = VAEEncoder(
            hidden_dims=vae_encoder_hidden_dims,
            z_dim=config['state_z_dim'],
            layer_norm=config['layer_norm'],
        )
        vae_decoder_def = VAEDecoder(
            hidden_dims=vae_decoder_hidden_dims,
            obs_dim=obs_dim,
            layer_norm=config['layer_norm'],
        )
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
        latent_value_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=2,
        )
        subgoal_encoder_def = SubgoalEncoder(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            z_dim=config['z_dim'],
        )

        # z-proposal: iMF flow conditioned on (s, g) outputting z candidates.
        ex_z_times = jnp.zeros((ex_observations.shape[0], 5), dtype=latent_dtype)  # iMF packs 5 time dims
        z_proposal_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=config['z_dim'],
            layer_norm=config['layer_norm'],
        )
        midpoint_decoder_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=config['state_z_dim'],
            layer_norm=config['layer_norm'],
        )
        ex_decoder_goals = ex_z

        if config['pe_type'] == 'frs':
            actor_def = ActorVectorField(
                hidden_dims=config['actor_hidden_dims'],
                action_dim=action_dim,
                layer_norm=config['layer_norm'],
            )
            ex_actor_in = (ex_encoded_observations, ex_encoded_goals, ex_actions, ex_times)
        elif config['pe_type'] == 'discrete':
            actor_def = GCDiscreteActor(
                hidden_dims=config['actor_hidden_dims'],
                action_dim=config['pe_discrete']['action_ct'],
                layer_norm=config['layer_norm'],
            )
            ex_actor_in = (ex_encoded_observations, ex_encoded_goals, ex_actions)
        elif config['pe_type'] == 'rpg':
            actor_def = GCActor(
                hidden_dims=config['actor_hidden_dims'],
                action_dim=action_dim,
                layer_norm=config['layer_norm'],
                state_dependent_std=False,
                const_std=pe_info['const_std'],
            )
            ex_actor_in = (ex_encoded_observations, ex_encoded_goals, ex_actions)
        else:
            raise ValueError(f"Unsupported pe_type: {config['pe_type']}")

        network_info = dict(
            vae_encoder=(vae_encoder_def, (ex_encoded_observations,)),
            vae_decoder=(vae_decoder_def, (ex_u,)),
            subgoal_encoder=(subgoal_encoder_def, (ex_u,)),
            value=(value_def, (ex_u, ex_u)),
            target_value=(copy.deepcopy(value_def), (ex_u, ex_u)),
            q=(q_def, (ex_q_obs, ex_q_goals, ex_z)),
            target_q=(copy.deepcopy(q_def), (ex_q_obs, ex_q_goals, ex_z)),
            q_state=(q_state_def, (ex_u, ex_u, ex_u)),
            target_q_state=(copy.deepcopy(q_state_def), (ex_u, ex_u, ex_u)),
            q_short=(q_short_def, (ex_encoded_observations, ex_encoded_goals, ex_actions)),
            latent_value=(latent_value_def, (ex_z, ex_z)),
            target_latent_value=(copy.deepcopy(latent_value_def), (ex_z, ex_z)),
            z_proposal=(z_proposal_def, (ex_u, ex_u, ex_z, ex_z_times)),
            actor=(actor_def, ex_actor_in),
        )
        if encoder_def is not None:
            network_info['encoder'] = (encoder_def, (ex_observations,))
        if config.get('grounded_midpoint_targets', False):
            network_info.update(
                dict(
                    midpoint_decoder=(
                        midpoint_decoder_def,
                        (ex_u, ex_decoder_goals, ex_u, ex_z_times),
                    ),
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
        params['modules_target_q'] = params['modules_q']
        params['modules_target_q_state'] = params['modules_q_state']
        params['modules_target_latent_value'] = params['modules_latent_value']

        config['action_dim'] = action_dim
        config['raw_observation_shape'] = raw_observation_shape
        config['stitch_mode'] = stitch_mode
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
            cf_expectile=0.7,
            q_agg='min',
            z_dim=8,
            state_z_dim=32,
            vae_beta=0.01,
            reg_coef=1.0,
            vae_recon_coef=0.25,
            vae_encoder_hidden_dims=(1024,) * 4,
            vae_decoder_hidden_dims=(256, 256),
            encoder='',
            freeze_encoder=False,
            hybrid_state_midpoint_max_offset=-1,
            grounded_midpoint_targets=False,
            midpoint_decoder_coef=0.0,
            midpoint_decoder_cfg_w_min=1.0,
            midpoint_decoder_cfg_w_max=1.0,
            midpoint_decoder_cfg_scale=1.0,
            midpoint_decoder_cfg_beta=1.0,
            midpoint_decoder_cfg_class_dropout_prob=0.0,
            direct_intraj_value_max_offset=-1,
            z_proposal_coef=0.0,
            z_proposal_cfg_w_min=1.0,
            z_proposal_cfg_w_max=1.0,
            z_proposal_cfg_scale=1.0,
            z_proposal_cfg_beta=1.0,
            z_proposal_cfg_class_dropout_prob=0.0,
            cf_num_z_proposals=1,
            cf_num_z_proposals_intraj=-1,
            cf_num_z_proposals_random=-1,
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
            use_dual_value_goals=False,
            value_goal_warmup_steps=0,
            sigreg_coef=0.0,
            sigreg_num_slices=32,
            latent_value_coef=0.0,
            stitch_mode='',
            latent_value_stitching=False,
            unconditional_z_proposal=False,
            unconditional_midpoint_decoder=False,
            unconditional_actor_proposal=False,
            proposal_improvement_coef=0.0,
            proposal_improvement_num_samples=8,
            latent_round_trip=False,
            direct_midpoint_proposal=False,
            intraj_direct_only_cf=False,
            q_short_n_step=1,
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
