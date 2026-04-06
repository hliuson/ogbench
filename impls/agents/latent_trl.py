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
from utils.latent_regularizers import sliced_epps_pulley_loss
from utils.networks import ActorVectorField, GCActor, GCDiscreteActor, GCValue, MLP
from utils.pretraining import ImpalaDecoder


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
        return self.config['reg_coef']

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
        raw_state_tensors = [
            batch['observations'],
            batch['next_observations'],
            batch['value_midpoint_observations'],
            batch['value_goal_observations'],
        ]
        state_tensors = [self._encode_visual(t, grad_params=grad_params) for t in raw_state_tensors]
        vae_inputs = jnp.concatenate(state_tensors, axis=0)
        if self._has_visual_encoder():
            vae_targets = jnp.concatenate(raw_state_tensors, axis=0).astype(jnp.float32) / 255.0
        else:
            vae_targets = vae_inputs

        mu, log_var = self.network.select('vae_encoder')(vae_inputs, params=grad_params)
        std = jnp.exp(0.5 * log_var)
        eps = jax.random.normal(rng, mu.shape)
        z = mu + std * eps
        recon = self.network.select('vae_decoder')(z, params=grad_params)
        if self._has_visual_encoder():
            recon = jax.nn.sigmoid(recon)

        recon_loss = jnp.mean((recon - vae_targets) ** 2)
        kl_loss = -0.5 * jnp.mean(1.0 + log_var - mu**2 - jnp.exp(log_var))
        total_loss = self.config['vae_recon_coef'] * recon_loss + self.config['vae_beta'] * kl_loss

        return total_loss, {
            'loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'z_mean': mu.mean(),
            'z_std': std.mean(),
        }

    def _encode_subgoal(self, midpoint_observations, grad_params=None):
        if grad_params is None:
            return self.network.select('subgoal_encoder')(midpoint_observations)
        return self.network.select('subgoal_encoder')(midpoint_observations, params=grad_params)

    def _apply_q_network(self, module_name, u_s, u_g, z_mid, grad_params=None):
        if grad_params is None:
            return self.network.select(module_name)(u_s, goals=u_g, actions=z_mid)
        return self.network.select(module_name)(
            u_s,
            goals=u_g,
            actions=z_mid,
            params=grad_params,
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

    def _has_cf_z_stitching(self, batch):
        return 'value_goals_is_intraj' in batch and self.config['z_proposal_coef'] > 0

    def _get_cf_num_z_proposals(self):
        return max(1, int(self.config.get('cf_num_z_proposals', 1)))

    def _select_value_training_batch(self, batch, step, rng=None):
        """Optionally override the value-side goal source during early warmup.

        When dual value goals are available, this ramps from explicit
        in-trajectory goals toward the dataset's configured mixed value-goal
        distribution, while leaving actor-side goals unchanged.
        """
        mix_ramp_steps = int(self.config.get('value_goal_mix_ramp_steps', 0))
        if mix_ramp_steps > 0 and 'value_goal_observations_random' in batch:
            progress = jnp.clip(step / float(mix_ramp_steps), 0.0, 1.0)
            start_traj = float(self.config.get('value_goal_mix_start_trajgoal', self.config['value_p_trajgoal']))
            start_random = float(
                self.config.get('value_goal_mix_start_randomgoal', self.config['value_p_randomgoal'])
            )
            end_traj = float(self.config['value_p_trajgoal'])
            end_random = float(self.config['value_p_randomgoal'])
            traj_mass = start_traj + progress * (end_traj - start_traj)
            random_mass = start_random + progress * (end_random - start_random)
            total_mass = jnp.maximum(traj_mass + random_mass, 1e-8)
            intraj_prob = traj_mass / total_mass

            rng = self.rng if rng is None else rng
            mix_rng = jax.random.fold_in(rng, step)
            choose_intraj = jax.random.bernoulli(
                mix_rng,
                p=intraj_prob,
                shape=batch['observations'].shape[:1],
            )
            choose_goal = choose_intraj[:, None]

            selected_batch = dict(batch)
            selected_batch['value_goals'] = jnp.where(
                choose_goal,
                batch['value_goals_intraj'],
                batch['value_goals_random'],
            )
            selected_batch['value_goal_observations'] = jnp.where(
                choose_goal,
                batch['value_goal_observations_intraj'],
                batch['value_goal_observations_random'],
            )
            selected_batch['value_offsets'] = jnp.where(
                choose_intraj,
                batch['value_offsets_intraj'],
                jnp.zeros_like(batch['value_offsets']),
            )
            selected_batch['value_midpoint_offsets'] = jnp.where(
                choose_intraj,
                batch['value_midpoint_offsets_intraj'],
                jnp.zeros_like(batch['value_midpoint_offsets']),
            )
            selected_batch['value_midpoint_observations'] = jnp.where(
                choose_goal,
                batch['value_midpoint_observations_intraj'],
                batch['observations'],
            )
            selected_batch['value_midpoint_actions'] = jnp.where(
                choose_goal,
                batch['value_midpoint_actions_intraj'],
                batch['actions'],
            )
            selected_batch['value_midpoint_goals'] = jnp.where(
                choose_goal,
                batch['value_midpoint_goals_intraj'],
                batch['value_cur_goals'],
            )
            selected_batch['value_goals_is_intraj'] = choose_intraj.astype(batch['value_goals_is_intraj'].dtype)
            return selected_batch, progress < 1.0, progress

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
            selected_batch['value_midpoint_observations'] = jnp.where(
                goal_mask,
                batch['value_midpoint_observations'],
                batch['value_midpoint_observations_intraj'],
            )
        if 'value_midpoint_goals_intraj' in batch:
            selected_batch['value_midpoint_goals'] = jnp.where(
                goal_mask,
                batch['value_midpoint_goals'],
                batch['value_midpoint_goals_intraj'],
            )
        if 'value_midpoint_actions_intraj' in batch:
            selected_batch['value_midpoint_actions'] = jnp.where(
                goal_mask,
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

    def _select_awr_training_batch(self, batch, step, rng=None):
        """Optionally override the proposal-training intraj/random mix.

        This is intentionally separate from value learning so Q/V can stay on
        their default batch mix while AWR sees a broader CF-heavy proposal mix.
        """
        awr_intraj_prob = float(self.config.get('z_proposal_awr_intraj_prob', -1.0))
        if awr_intraj_prob < 0.0:
            return batch
        if 'value_goal_observations_random' not in batch or 'value_goals_is_intraj' not in batch:
            return batch

        rng = self.rng if rng is None else rng
        mix_rng = jax.random.fold_in(rng, step + 17017)
        choose_intraj = jax.random.bernoulli(
            mix_rng,
            p=awr_intraj_prob,
            shape=batch['observations'].shape[:1],
        )
        choose_goal = choose_intraj[:, None]

        selected_batch = dict(batch)
        selected_batch['value_goals'] = jnp.where(
            choose_goal,
            batch['value_goals_intraj'],
            batch['value_goals_random'],
        )
        selected_batch['value_goal_observations'] = jnp.where(
            choose_goal,
            batch['value_goal_observations_intraj'],
            batch['value_goal_observations_random'],
        )
        if 'value_offsets_intraj' in batch:
            selected_batch['value_offsets'] = jnp.where(
                choose_intraj,
                batch['value_offsets_intraj'],
                jnp.zeros_like(batch['value_offsets']),
            )
        if 'value_midpoint_offsets_intraj' in batch:
            selected_batch['value_midpoint_offsets'] = jnp.where(
                choose_intraj,
                batch['value_midpoint_offsets_intraj'],
                jnp.zeros_like(batch['value_midpoint_offsets']),
            )
        if 'value_midpoint_observations_intraj' in batch:
            selected_batch['value_midpoint_observations'] = jnp.where(
                choose_goal,
                batch['value_midpoint_observations_intraj'],
                batch['observations'],
            )
        if 'value_midpoint_actions_intraj' in batch:
            selected_batch['value_midpoint_actions'] = jnp.where(
                choose_goal,
                batch['value_midpoint_actions_intraj'],
                batch['actions'],
            )
        if 'value_midpoint_goals_intraj' in batch:
            selected_batch['value_midpoint_goals'] = jnp.where(
                choose_goal,
                batch['value_midpoint_goals_intraj'],
                batch['value_cur_goals'],
            )
        selected_batch['value_goals_is_intraj'] = choose_intraj.astype(batch['value_goals_is_intraj'].dtype)
        return selected_batch

    def _distance_weight_from_value(self, v_base):
        """Compute TRL-style distance weights from a base value estimate V(s, g)."""
        lam = float(self.config.get('lam', 0.0))
        if lam == 0.0:
            return jnp.ones_like(v_base), jnp.zeros_like(v_base)

        safe_v = jnp.clip(v_base, 1e-8, 1.0)
        implied_dist = jax.lax.stop_gradient(jnp.log(safe_v) / jnp.log(self.config['discount']))
        weight = (1.0 / (1.0 + implied_dist)) ** lam
        return weight, implied_dist

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
        value_obs = jnp.concatenate([observations, midpoint_observations], axis=0)
        value_goals = jnp.concatenate([midpoint_goals, goals], axis=0)
        value_logits = self.network.select('target_value')(value_obs, goals=value_goals)
        first_v_logits, second_v_logits = jnp.split(value_logits, 2, axis=1)

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

    def _prepare_value_ctx(self, batch, grad_params):
        """Cache value-side encodings shared across multiple losses."""
        u_s = self._encode_state(batch['observations'], grad_params=grad_params)
        u_g = self._encode_state(batch['value_goal_observations'], grad_params=grad_params)
        u_mid = self._encode_state(batch['value_midpoint_observations'], grad_params=grad_params)
        z_mid = self._encode_subgoal(u_mid, grad_params=grad_params)
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
        return {
            'u_s': u_s,
            'u_g': u_g,
            'u_mid': u_mid,
            'z_mid': z_mid,
            'v_logits': v_logits,
            'vs': vs,
            'direct_target_ens': direct_target_ens,
        }

    def _prepare_policy_ctx(self, batch, grad_params):
        """Cache shared actor/q-short policy encodings."""
        actor_observations, actor_goals = self._encode_policy_inputs(
            batch['observations'],
            batch['actor_goals'],
            grad_params=grad_params,
        )
        actor_goal_u = self._encode_state(batch['actor_goal_observations'])
        return {
            'actor_observations': actor_observations,
            'actor_goals': actor_goals,
            'actor_goal_u': actor_goal_u,
        }

    def _prepare_proposal_ctx(self, batch):
        """Cache proposal-conditioning encodings using the non-grad path."""
        u_s = self._encode_state(batch['observations'])
        u_g = self._encode_state(batch['value_goal_observations'])
        return {
            'u_s': u_s,
            'u_g': u_g,
        }

    def _prepare_sigreg_ctx(self, batch, grad_params):
        """Cache intraj midpoint latents for SIGReg."""
        midpoint_observations = batch.get('value_midpoint_observations_intraj', batch['value_midpoint_observations'])
        u_mid = self._encode_state(midpoint_observations, grad_params=grad_params)
        z_mid = self._encode_subgoal(u_mid, grad_params=grad_params)
        return {
            'z_mid': z_mid,
        }

    def _evaluate_proposed_q_values(self, u_s, u_g, z_proposed):
        """Evaluate target Q on one or more proposed midpoint latents."""
        if z_proposed.ndim == 2:
            q_proposed_logits = self._apply_q_network('target_q', u_s, u_g, z_proposed)
            q_proposed = jax.nn.sigmoid(q_proposed_logits)
            return self._aggregate_q_ensembles(q_proposed)

        num_cf_samples = z_proposed.shape[0]
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
        return q_proposed_values.max(axis=0)

    def _sample_midpoint_decoder(self, z, rng, num_samples=1, dtype=jnp.float32):
        """Sample midpoint state latents from the unconditional decoder p(u_mid | z)."""
        state_latent_dim = self.config['state_z_dim']
        batch_size = z.shape[0] if num_samples == 1 else z.shape[1]
        observations = jnp.zeros((batch_size, state_latent_dim), dtype=dtype)
        decoder_goals = z if num_samples == 1 else None

        if num_samples == 1:
            sample_shape = (batch_size, state_latent_dim)
        else:
            sample_shape = (num_samples, batch_size, state_latent_dim)

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
        """Condition proposal on the current (s, g) pair."""
        return observations, goals

    def _actor_proposal_goals(self, goals):
        """Keep actor conditioning aligned with the sampled goal."""
        return goals

    def midpoint_decoder_loss(self, batch, grad_params, rng, value_ctx=None):
        """Train the unconditional midpoint decoder: z_mid -> u_mid."""
        if value_ctx is None:
            u_target = self._encode_state(batch['value_midpoint_observations'])
            z_target = self._encode_subgoal(u_target)
        else:
            u_target = value_ctx['u_mid']
            z_target = value_ctx['z_mid']
        u_s = jnp.zeros_like(u_target)
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
        for k, v in flow_info.items():
            if k != 'loss':
                info[k] = v
        return loss, info

    def midpoint_sigreg_loss(self, batch, grad_params, rng, sigreg_ctx=None):
        """Regularize midpoint latents toward a simple Gaussian geometry."""
        if sigreg_ctx is None:
            sigreg_ctx = self._prepare_sigreg_ctx(batch, grad_params)
        z_mid = sigreg_ctx['z_mid']
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

    def _q_loss_grounded(self, batch, grad_params, rng=None, value_ctx=None, proposal_ctx=None):
        """Train latent Q from grounded midpoint state factors."""
        has_cf = self._has_cf_z_stitching(batch)

        q_rng, decoder_rng = jax.random.split(rng)
        if value_ctx is None:
            value_ctx = self._prepare_value_ctx(batch, grad_params)
        u_s = value_ctx['u_s']
        u_g = value_ctx['u_g']
        z_mid = value_ctx['z_mid']
        q_logits = self._apply_q_network('q', u_s, u_g, z_mid, grad_params=grad_params)
        intraj_target = value_ctx['direct_target_ens']

        if has_cf:
            is_intraj = batch['value_goals_is_intraj']
            is_intraj_2d = is_intraj[None, ...]

            z_cf = self._sample_z_proposal(
                batch,
                q_rng,
                num_samples=1,
                observations=None if proposal_ctx is None else proposal_ctx['u_s'],
                goals=None if proposal_ctx is None else proposal_ctx['u_g'],
            )
            z_cf = jax.lax.stop_gradient(z_cf)
            m_cf = self._sample_midpoint_decoder(z_cf, decoder_rng, num_samples=1, dtype=u_s.dtype)
            m_cf = jax.lax.stop_gradient(m_cf)

            q_cf_logits = self._apply_q_network('q', u_s, u_g, z_cf, grad_params=grad_params)
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
        base_v = self._aggregate_q_ensembles(value_ctx['vs'])
        dist_weight, implied_dist = self._distance_weight_from_value(base_v)
        q_loss = dist_weight * self.bce_loss(q_logits, jax.lax.stop_gradient(target))
        total_loss = q_loss.mean()

        info = {
            'total_loss': total_loss,
            'q_loss': q_loss.mean(),
            'q_mean': qs.mean(),
            'q_max': qs.max(),
            'q_min': qs.min(),
            'base_v_mean': base_v.mean(),
            'implied_dist_mean': implied_dist.mean(),
            'dist_weight_mean': dist_weight.mean(),
            'grounded_target_mean': target.mean(),
            'intraj_target_mean': intraj_target.mean(),
        }
        if has_cf:
            info['cf_frac'] = (1.0 - is_intraj).mean()
            info['cf_target_mean'] = cf_target.mean()
        return total_loss, info

    def _value_loss_grounded(self, batch, grad_params, rng=None, value_ctx=None, proposal_ctx=None):
        """Use direct TRL targets locally and latent-Q stitching for longer horizons."""
        has_cf = self._has_cf_z_stitching(batch)
        if value_ctx is None:
            value_ctx = self._prepare_value_ctx(batch, grad_params)
        u_s = value_ctx['u_s']
        u_g = value_ctx['u_g']
        v_logits = value_ctx['v_logits']
        vs = value_ctx['vs']
        direct_target_ens = value_ctx['direct_target_ens']
        direct_target = self._aggregate_q_ensembles(direct_target_ens)
        z_mid = jax.lax.stop_gradient(value_ctx['z_mid'])
        q_traj_logits = self._apply_q_network('target_q', u_s, u_g, z_mid)
        q_traj = jax.nn.sigmoid(q_traj_logits)
        target_traj_q = self._aggregate_q_ensembles(q_traj)

        direct_cutoff = self.config.get('direct_intraj_value_max_offset', -1)
        if direct_cutoff is not None and direct_cutoff >= 0:
            direct_mask = batch['value_offsets'] <= direct_cutoff
            target_traj = jnp.where(direct_mask, direct_target, target_traj_q)
        else:
            direct_mask = None
            target_traj = target_traj_q

        use_proposal = self.config['z_proposal_coef'] > 0

        if use_proposal and has_cf:
            is_intraj = batch['value_goals_is_intraj']
            num_cf_samples = self._get_cf_num_z_proposals()
            z_proposed = self._sample_z_proposal(
                batch,
                rng,
                num_samples=num_cf_samples,
                observations=None if proposal_ctx is None else proposal_ctx['u_s'],
                goals=None if proposal_ctx is None else proposal_ctx['u_g'],
            )
            z_proposed = jax.lax.stop_gradient(z_proposed)
            target_proposed = self._evaluate_proposed_q_values(u_s, u_g, z_proposed)

            if self.config.get('intraj_cf_max_target', True):
                intraj_target = jnp.maximum(target_traj, target_proposed)
            else:
                intraj_target = target_traj
            cf_target_val = target_proposed
            target = jnp.where(is_intraj, intraj_target, cf_target_val)
        elif use_proposal:
            num_cf_samples = self._get_cf_num_z_proposals()
            z_proposed = self._sample_z_proposal(
                batch,
                rng,
                num_samples=num_cf_samples,
                observations=None if proposal_ctx is None else proposal_ctx['u_s'],
                goals=None if proposal_ctx is None else proposal_ctx['u_g'],
            )
            z_proposed = jax.lax.stop_gradient(z_proposed)
            target_proposed = self._evaluate_proposed_q_values(u_s, u_g, z_proposed)
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
        v_base = self._aggregate_q_ensembles(vs)
        dist_weight, implied_dist = self._distance_weight_from_value(v_base)
        v_loss = expectile_weight * dist_weight * self.bce_loss(v_logits, jax.lax.stop_gradient(target))
        total_loss = v_loss.mean()

        predicted_steps = jnp.log(vs + 1e-8) / jnp.log(self.config['discount'])
        actual_steps = batch['value_offsets']
        calib_mask = (actual_steps > 0).astype(jnp.float32)
        if has_cf:
            calib_mask = calib_mask * batch['value_goals_is_intraj']
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
            'base_v_mean': v_base.mean(),
            'implied_dist_mean': implied_dist.mean(),
            'dist_weight_mean': dist_weight.mean(),
            'calibration_rel_gap_mean': calib_mean,
            'calibration_rel_gap_max': calib_max,
            'backup_traj_mean': target_traj.mean(),
            'direct_target_mean': direct_target.mean(),
        }
        info['q_traj_mean'] = target_traj_q.mean()
        if direct_mask is not None:
            if has_cf:
                info['direct_frac'] = (direct_mask * batch['value_goals_is_intraj']).sum() / jnp.maximum(
                    batch['value_goals_is_intraj'].sum(), 1.0
                )
            else:
                info['direct_frac'] = direct_mask.mean()
        if use_proposal:
            info['q_proposed_mean'] = target_proposed.mean()
            if has_cf:
                intraj_mask = batch['value_goals_is_intraj']
                cf_mask = 1.0 - intraj_mask

                def _masked_mean(x, mask):
                    denom = jnp.maximum(mask.sum(), 1.0)
                    return (x * mask).sum() / denom

                info['q_proposed_mean_intraj'] = _masked_mean(target_proposed, intraj_mask)
                info['q_proposed_mean_cf'] = _masked_mean(target_proposed, cf_mask)
                info['q_proposed_mean_random'] = info['q_proposed_mean_cf']
                info['q_traj_mean_intraj'] = _masked_mean(target_traj_q, intraj_mask)
                info['q_traj_mean_cf'] = _masked_mean(target_traj_q, cf_mask)
        return total_loss, info

    def z_proposal_loss(self, batch, grad_params, rng, step=0):
        """Train the z-proposal network with support-AWR over candidate midpoints."""
        rng, awr_rng = jax.random.split(rng)
        awr_beta = self.config.get('z_proposal_awr_beta', 0.0)
        awr_num_random_support = max(1, int(self.config.get('z_proposal_awr_num_random_support', 1)))
        awr_value_eps = self.config.get('z_proposal_awr_value_eps', 0.0)
        intraj_mask = batch.get('value_goals_is_intraj', None)
        proposal_mask = intraj_mask
        awr_base_v = None
        if awr_beta > 0.0:
            awr_batch = self._select_awr_training_batch(batch, step, awr_rng)
            flow_observations = self._encode_state(awr_batch['observations'])
            flow_goals = self._encode_state(awr_batch['value_goal_observations'])
            intraj_mask = awr_batch.get('value_goals_is_intraj', intraj_mask)
            batch_size = awr_batch['observations'].shape[0]
            total_support = awr_num_random_support + 1 if intraj_mask is not None else awr_num_random_support
            w_rng, g_rng = jax.random.split(awr_rng)
            w_rngs = jax.random.split(w_rng, total_support)
            w_perms = jnp.stack([jax.random.permutation(r, batch_size) for r in w_rngs], axis=0)
            w_obs = awr_batch['observations'][w_perms]
            u_target = self._encode_state(w_obs.reshape(-1, *w_obs.shape[2:]))
            z_candidates = self._encode_subgoal(u_target)
            z_candidates = jax.lax.stop_gradient(z_candidates.reshape(total_support, batch_size, -1))
            if intraj_mask is not None:
                intraj_z = self._encode_subgoal(
                    self._encode_state(awr_batch['value_midpoint_observations'])
                )
                intraj_z = jax.lax.stop_gradient(intraj_z)
                intraj_support_mask = intraj_mask.astype(bool)[:, None]
                z_candidates = z_candidates.at[0].set(
                    jnp.where(intraj_support_mask, intraj_z, z_candidates[0])
                )

            g_perm = jax.random.permutation(g_rng, batch_size)
            g_obs = awr_batch['observations'][g_perm]
            flow_goals = self._encode_state(g_obs)

            obs_dim = flow_observations.shape[-1]
            goal_dim = flow_goals.shape[-1]
            obs_bc = jnp.broadcast_to(flow_observations[None], (total_support, *flow_observations.shape))
            goals_bc = jnp.broadcast_to(flow_goals[None], (total_support, *flow_goals.shape))

            q_logits_mid = self._apply_q_network(
                'target_q',
                obs_bc.reshape(-1, flow_observations.shape[-1]),
                goals_bc.reshape(-1, flow_goals.shape[-1]),
                z_candidates.reshape(-1, z_candidates.shape[-1]),
            )
            q_mid = self._aggregate_q_ensembles(jax.nn.sigmoid(q_logits_mid)).reshape(total_support, batch_size)
            v_logits = self.network.select('target_value')(flow_observations, goals=flow_goals)
            v_base = self._aggregate_q_ensembles(jax.nn.sigmoid(v_logits))
            awr_adv_candidates = jax.lax.stop_gradient(q_mid - v_base[None, ...])
            norm_base = v_base[None, ...] + awr_value_eps
            awr_score_matrix = awr_adv_candidates / norm_base
            awr_weight_matrix = jnp.exp(awr_beta * awr_score_matrix).astype(z_candidates.dtype)

            awr_max_weight = self.config.get('z_proposal_awr_max_weight', 20.0)
            if awr_max_weight is not None and awr_max_weight > 0:
                awr_weight_matrix = jnp.minimum(awr_weight_matrix, awr_max_weight)
            if self.config.get('z_proposal_awr_subtract_one', False):
                awr_weight_matrix = jnp.maximum(awr_weight_matrix - 1.0, 0.0)

            z_target = z_candidates.reshape(-1, z_candidates.shape[-1])
            flow_observations = obs_bc.reshape(-1, obs_dim)
            flow_goals = goals_bc.reshape(-1, goal_dim)
            awr_adv = awr_score_matrix.reshape(-1)
            awr_weight = awr_weight_matrix.reshape(-1)
            awr_base_v = jnp.broadcast_to(v_base[None, ...], awr_weight_matrix.shape).reshape(-1)
            has_positive = (awr_adv_candidates > 0).reshape(-1)
            proposal_mask = awr_weight
        else:
            u_target = self._encode_state(batch['value_midpoint_observations'])
            z_target = self._encode_subgoal(u_target)
            z_target = jax.lax.stop_gradient(z_target)
            flow_observations = self._encode_state(batch['observations'])
            flow_goals = self._encode_state(batch['value_goal_observations'])
            awr_adv = None
            awr_weight = None
            awr_base_v = None
            has_positive = None

        cond_observations, cond_goals = self._proposal_context(flow_observations, flow_goals)

        def vector_field_fn(noise, times):
            return self.network.select('z_proposal')(
                cond_observations, goals=cond_goals, actions=noise, times=times,
                params=grad_params,
            )

        loss, flow_info = imf_loss(
            rng, z_target, vector_field_fn,
            r_equals_t_prob=0.5,
            mask=proposal_mask,
        )

        total_loss = loss
        info = {'loss': loss}
        if intraj_mask is not None:
            info['intraj_frac'] = intraj_mask.mean()
        info['awr_enabled'] = jnp.asarray(float(awr_beta > 0.0), dtype=z_target.dtype)
        info['awr_intraj_prob_cfg'] = jnp.asarray(
            float(self.config.get('z_proposal_awr_intraj_prob', -1.0)), dtype=z_target.dtype
        )
        if awr_adv is not None:
            awr_weight_flat = awr_weight.reshape(-1)
            awr_weight_sorted = jnp.sort(awr_weight_flat)
            n_weights = awr_weight_sorted.shape[0]

            def _quantile_at(fraction):
                idx = jnp.minimum(
                    n_weights - 1,
                    jnp.asarray(jnp.floor((n_weights - 1) * fraction), dtype=jnp.int32),
                )
                return awr_weight_sorted[idx]

            info['awr_adv_mean'] = awr_adv.mean()
            info['awr_positive_frac'] = has_positive.astype(z_target.dtype).mean()
            info['awr_weight_mean'] = awr_weight.mean()
            info['awr_weight_median'] = _quantile_at(0.5)
            info['awr_weight_p90'] = _quantile_at(0.9)
            info['awr_weight_p99'] = _quantile_at(0.99)
            info['awr_weight_max'] = awr_weight.max()
            info['awr_weight_nonzero_frac'] = (awr_weight > 0.0).mean()
            info['awr_weight_gt1_frac'] = (awr_weight > 1.0).mean()
            if awr_base_v is not None:
                info['awr_base_v_mean'] = awr_base_v.mean()
            awr_max_weight = self.config.get('z_proposal_awr_max_weight', 20.0)
            if awr_max_weight is not None and awr_max_weight > 0:
                info['awr_weight_cap_frac'] = (awr_weight >= (awr_max_weight - 1e-6)).mean()
            info['awr_weight_subtract_one'] = jnp.asarray(
                float(self.config.get('z_proposal_awr_subtract_one', False)),
                dtype=z_target.dtype,
            )
            info['awr_random_support'] = jnp.asarray(1.0, dtype=z_target.dtype)
            info['awr_num_random_support'] = jnp.asarray(float(awr_num_random_support), dtype=z_target.dtype)
            info['awr_intraj_support'] = jnp.asarray(float(intraj_mask is not None), dtype=z_target.dtype)

        for k, v in flow_info.items():
            if k != 'loss':
                info[k] = v

        return total_loss, info

    def _sample_z_proposal(self, batch, rng, num_samples=1, observations=None, goals=None):
        """Sample z candidates from the z-proposal network via iMF one-shot."""
        if observations is None:
            observations = self._encode_state(batch['observations'])
        if goals is None:
            goals = self._encode_state(batch['value_goal_observations'])
        observations, goals = self._proposal_context(observations, goals)
        z_dim = self.config['z_dim']

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

        z_samples = imf_one_shot_sample(rng, sample_shape, vector_field_fn)
        return z_samples

    def q_short_loss(self, batch, grad_params, policy_ctx=None):
        """Distill Q_short(s, g, a) from an n-step bootstrap into V for FRS action selection."""
        q_short_goal_key = 'actor_goals'
        target_goal_key = 'actor_goal_observations'
        q_short_n_step = int(self.config.get('q_short_n_step', 1))
        next_obs_key = 'actor_nstep_observations' if q_short_n_step > 1 and 'actor_nstep_observations' in batch else 'next_observations'
        u_next = self._encode_state(batch[next_obs_key])
        u_g = policy_ctx['actor_goal_u'] if policy_ctx is not None else self._encode_state(batch[target_goal_key])

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

        if policy_ctx is None:
            q_short_observations, q_short_goals = self._encode_policy_inputs(
                batch['observations'],
                batch[q_short_goal_key],
                grad_params=grad_params,
            )
        else:
            q_short_observations = policy_ctx['actor_observations']
            q_short_goals = policy_ctx['actor_goals']
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

    def actor_loss(self, batch, grad_params, rng=None, policy_ctx=None):
        """Flat actor loss (same policy extraction interface as TRL)."""
        pe_info = self._get_pe_info()

        if self.config['pe_type'] == 'rpg':
            if policy_ctx is None:
                actor_observations, actor_goals = self._encode_policy_inputs(
                    batch['observations'],
                    batch['actor_goals'],
                    grad_params=grad_params,
                )
                actor_goal_u = self._encode_state(batch['actor_goal_observations'])
            else:
                actor_observations = policy_ctx['actor_observations']
                actor_goals = policy_ctx['actor_goals']
                actor_goal_u = policy_ctx['actor_goal_u']
            actor_goals = self._actor_proposal_goals(actor_goals)
            dist = self.network.select('actor')(
                actor_observations,
                actor_goals,
                params=grad_params,
            )

            u_next = self._encode_state(batch['next_observations'])
            v_next = self.network.select('value')(
                u_next,
                goals=actor_goal_u,
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
            if policy_ctx is None:
                actor_observations, actor_goals = self._encode_policy_inputs(
                    batch['observations'],
                    batch['actor_goals'],
                    grad_params=grad_params,
                )
                actor_goal_u = self._encode_state(batch['actor_goal_observations'])
            else:
                actor_observations = policy_ctx['actor_observations']
                actor_goals = policy_ctx['actor_goals']
                actor_goal_u = policy_ctx['actor_goal_u']
            actor_goals = self._actor_proposal_goals(actor_goals)
            dist = self.network.select('actor')(
                actor_observations,
                actor_goals,
                params=grad_params,
            )

            u_next = self._encode_state(batch['next_observations'])
            v = self.network.select('value')(
                u_next,
                goals=actor_goal_u,
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
            if policy_ctx is None:
                actor_observations, actor_goals = self._encode_policy_inputs(
                    batch['observations'],
                    batch['actor_goals'],
                    grad_params=grad_params,
                )
            else:
                actor_observations = policy_ctx['actor_observations']
                actor_goals = policy_ctx['actor_goals']
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
        value_ctx = self._prepare_value_ctx(value_batch, grad_params)
        policy_ctx = self._prepare_policy_ctx(batch, grad_params)
        proposal_ctx = None
        if self.config['z_proposal_coef'] > 0:
            proposal_ctx = self._prepare_proposal_ctx(value_batch)
        sigreg_ctx = None
        if self.config.get('sigreg_coef', 0.0) > 0:
            sigreg_ctx = self._prepare_sigreg_ctx(value_batch, grad_params)

        vae_loss, vae_info = self._state_vae_loss(value_batch, grad_params, rng=vae_rng)
        for k, v in vae_info.items():
            info[f'vae/{k}'] = v

        value_loss, value_info = self._value_loss_grounded(
            value_batch, grad_params, rng=value_rng, value_ctx=value_ctx, proposal_ctx=proposal_ctx
        )
        for k, v in value_info.items():
            info[f'value/{k}'] = v

        q_loss, q_info = self._q_loss_grounded(
            value_batch, grad_params, rng=q_rng, value_ctx=value_ctx, proposal_ctx=proposal_ctx
        )
        for k, v in q_info.items():
            info[f'q/{k}'] = v

        z_proposal_loss = 0.0
        if self.config['z_proposal_coef'] > 0:
            z_proposal_loss, z_proposal_info = self.z_proposal_loss(
                value_batch, grad_params, rng=z_proposal_rng, step=step
            )
            for k, v in z_proposal_info.items():
                info[f'z_proposal/{k}'] = v

        midpoint_decoder_loss = 0.0
        if self.config.get('midpoint_decoder_coef', 0.0) > 0:
            midpoint_decoder_loss, midpoint_decoder_info = self.midpoint_decoder_loss(
                value_batch, grad_params, rng=midpoint_decoder_rng, value_ctx=value_ctx
            )
            for k, v in midpoint_decoder_info.items():
                info[f'midpoint_decoder/{k}'] = v

        midpoint_sigreg_loss = 0.0
        if self.config.get('sigreg_coef', 0.0) > 0:
            midpoint_sigreg_loss, midpoint_sigreg_info = self.midpoint_sigreg_loss(
                value_batch, grad_params, rng=midpoint_decoder_rng, sigreg_ctx=sigreg_ctx
            )
            for k, v in midpoint_sigreg_info.items():
                info[f'sigreg/{k}'] = v

        q_short_loss, q_short_info = self.q_short_loss(batch, grad_params, policy_ctx=policy_ctx)
        for k, v in q_short_info.items():
            info[f'q_short/{k}'] = v

        actor_loss, actor_info = self.actor_loss(batch, grad_params, rng=actor_rng, policy_ctx=policy_ctx)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = (
            self._get_reg_coef() * vae_loss
            + value_loss
            + q_loss
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
        ex_times = ex_actions[..., :1]
        action_dim = ex_actions.shape[-1]
        pe_info = cls._get_pe_info_from_config(config)
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

        vae_encoder_hidden_dims = config.get('vae_encoder_hidden_dims', config['value_hidden_dims'])
        vae_decoder_hidden_dims = config.get('vae_decoder_hidden_dims', config['actor_hidden_dims'])
        vae_encoder_def = VAEEncoder(
            hidden_dims=vae_encoder_hidden_dims,
            z_dim=config['state_z_dim'],
            layer_norm=config['layer_norm'],
        )
        if encoder_def is not None and len(raw_observation_shape) == 3:
            vae_decoder_def = ImpalaDecoder(
                output_channels=raw_observation_shape[-1],
                mlp_hidden_dims=vae_decoder_hidden_dims,
            )
        else:
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
            q=(q_def, (ex_u, ex_u, ex_z)),
            target_q=(copy.deepcopy(q_def), (ex_u, ex_u, ex_z)),
            q_short=(q_short_def, (ex_encoded_observations, ex_encoded_goals, ex_actions)),
            z_proposal=(z_proposal_def, (ex_u, ex_u, ex_z, ex_z_times)),
            actor=(actor_def, ex_actor_in),
            midpoint_decoder=(
                midpoint_decoder_def,
                (ex_u, ex_decoder_goals, ex_u, ex_z_times),
            ),
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
            midpoint_decoder_coef=0.0,
            direct_intraj_value_max_offset=-1,
            z_proposal_coef=0.0,
            z_proposal_awr_beta=0.0,
            z_proposal_awr_max_weight=20.0,
            z_proposal_awr_num_random_support=4,
            z_proposal_awr_value_eps=0.01,
            z_proposal_awr_intraj_prob=-1.0,
            z_proposal_awr_subtract_one=False,
            cf_num_z_proposals=1,
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
            value_goal_mix_ramp_steps=0,
            value_goal_mix_start_trajgoal=1.0,
            value_goal_mix_start_randomgoal=0.0,
            sigreg_coef=0.0,
            sigreg_num_slices=32,
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
