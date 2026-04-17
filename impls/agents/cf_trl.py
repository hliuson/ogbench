import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax
from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.flows import imf_loss, imf_one_shot_sample
from utils.networks import ActorVectorField, GCActor, GCDiscreteActor, GCValue


class CFTRLAgent(flax.struct.PyTreeNode):
    """CF-TRL with intraj or flow-Q proposer objectives."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    @staticmethod
    def _get_pe_info_from_config(config):
        if config['pe_type'] == 'discrete':
            return config['pe_discrete']
        return config[config['pe_type']]

    @staticmethod
    def _validate_config(config):
        """Fail fast on invalid or inconsistent configs."""
        def _require(condition, message):
            if not condition:
                raise ValueError(f'Invalid cf_trl config: {message}')

        def _check_prob_mix(name, probs):
            probs = [float(p) for p in probs]
            _require(all(p >= 0.0 for p in probs), f'{name} probabilities must be non-negative, got {probs}')
            total = sum(probs)
            _require(abs(total - 1.0) <= 1e-6, f'{name} probabilities must sum to 1, got {probs} (sum={total})')

        _require(config['batch_size'] > 0, f'batch_size must be positive, got {config["batch_size"]}')
        _require(config['cf_num_z_proposals'] >= 1, f'cf_num_z_proposals must be >= 1, got {config["cf_num_z_proposals"]}')
        _require(config['q_action_n_step'] >= 1, f'q_action_n_step must be >= 1, got {config["q_action_n_step"]}')
        _require(config['lr'] > 0.0, f'lr must be positive, got {config["lr"]}')
        _require(0.0 < config['discount'] <= 1.0, f'discount must be in (0, 1], got {config["discount"]}')
        _require(
            np.isclose(config['z_proposal_coef'], 1.0),
            f'cf_trl hard-codes z_proposal_coef=1.0, got {config["z_proposal_coef"]}',
        )
        _require(
            int(config.get('z_proposal_random_goal_num_support', 0)) >= 0,
            'z_proposal_random_goal_num_support must be non-negative',
        )
        _require(
            config.get('z_proposal_learning_method', 'intraj') in {'intraj', 'flow_q'},
            "z_proposal_learning_method must be one of 'intraj' or 'flow_q'",
        )
        _require(config.get('use_dual_value_goals', True), 'cf_trl requires use_dual_value_goals=True')
        _require(
            config['pe_type'] in {'frs', 'rpg', 'discrete'},
            f"pe_type must be one of 'frs', 'rpg', 'discrete', got {config['pe_type']}",
        )
        _check_prob_mix(
            'value goal mix',
            [config['value_p_curgoal'], config['value_p_trajgoal'], config['value_p_randomgoal']],
        )
        _check_prob_mix(
            'actor goal mix',
            [config['actor_p_curgoal'], config['actor_p_trajgoal'], config['actor_p_randomgoal']],
        )

    @staticmethod
    def _raise_on_nonfinite_losses(step, total_loss, value_loss, z_proposal_loss, q_action_loss, actor_loss):
        scalars = {
            'total_loss': float(total_loss),
            'value_loss': float(value_loss),
            'z_proposal_loss': float(z_proposal_loss),
            'q_action_loss': float(q_action_loss),
            'actor_loss': float(actor_loss),
        }
        bad = {k: v for k, v in scalars.items() if not np.isfinite(v)}
        if bad:
            bad_str = ', '.join(f'{k}={v}' for k, v in bad.items())
            raise FloatingPointError(f'Non-finite cf_trl losses at step {int(step)}: {bad_str}')

    def _get_pe_info(self):
        return self._get_pe_info_from_config(self.config)

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

    def _midpoint_dim(self):
        return int(self.config['state_dim'])

    @staticmethod
    def bce_loss(pred_logit, target):
        log_pred = jax.nn.log_sigmoid(pred_logit)
        log_not_pred = jax.nn.log_sigmoid(-pred_logit)
        return -(log_pred * target + log_not_pred * (1 - target))

    def _aggregate_q_ensembles(self, values):
        return jnp.minimum(values[0], values[1])

    @staticmethod
    def _masked_mean(values, mask):
        mask = mask.astype(values.dtype)
        denom = jnp.maximum(mask.sum(), 1.0)
        return (values * mask).sum() / denom

    def _get_cf_num_z_proposals(self):
        return max(1, int(self.config.get('cf_num_z_proposals', 1)))

    def _get_z_proposal_learning_method(self):
        return self.config.get('z_proposal_learning_method', 'intraj')

    def _use_mixed_goals_for_proposal(self):
        return bool(self.config.get('z_proposal_use_mixed_goals', False))

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

    def _select_intraj_proposal_training_batch(self, batch):
        """Select the in-trajectory proposer-training triples."""
        if 'value_goals_is_intraj' not in batch:
            raise ValueError('cf_trl requires value_goals_is_intraj in the proposer batch.')
        if 'value_goal_observations_intraj' not in batch:
            raise ValueError('cf_trl requires dual value-goal intraj fields for proposer training.')

        selected_batch = dict(batch)
        selected_batch['value_goals'] = batch['value_goals_intraj']
        selected_batch['value_goal_observations'] = batch['value_goal_observations_intraj']
        if 'value_offsets_intraj' in batch:
            selected_batch['value_offsets'] = batch['value_offsets_intraj']
        if 'value_midpoint_offsets_intraj' in batch:
            selected_batch['value_midpoint_offsets'] = batch['value_midpoint_offsets_intraj']
        if 'value_midpoint_observations_intraj' in batch:
            selected_batch['value_midpoint_observations'] = batch['value_midpoint_observations_intraj']
        if 'value_midpoint_actions_intraj' in batch:
            selected_batch['value_midpoint_actions'] = batch['value_midpoint_actions_intraj']
        if 'value_midpoint_goals_intraj' in batch:
            selected_batch['value_midpoint_goals'] = batch['value_midpoint_goals_intraj']
        selected_batch['value_goals_is_intraj'] = jnp.ones_like(batch['value_goals_is_intraj'])
        return selected_batch

    def _select_proposal_training_batch(self, batch, step, rng=None):
        """Select proposer-training triples for the configured learning rule."""
        del step, rng
        method = self._get_z_proposal_learning_method()
        if method not in {'intraj', 'flow_q'}:
            raise ValueError(f'Unsupported z_proposal_learning_method: {method}')

        if 'value_goals_is_intraj' not in batch:
            raise ValueError('cf_trl requires value_goals_is_intraj in the proposer batch.')

        if self._use_mixed_goals_for_proposal():
            return dict(batch)
        if method == 'intraj' and int(self.config.get('z_proposal_random_goal_num_support', 0)) > 0:
            return dict(batch)
        return self._select_intraj_proposal_training_batch(batch)

    def _score_support_candidates(self, observations, goals, candidates):
        """Score candidate supports with factorized target values V(s,w) * V(w,g)."""
        batch_size, num_candidates, candidate_dim = candidates.shape
        obs_dim = observations.shape[-1]
        goal_dim = goals.shape[-1]

        obs_flat = jnp.broadcast_to(observations[:, None, :], (batch_size, num_candidates, obs_dim))
        goals_flat = jnp.broadcast_to(goals[:, None, :], (batch_size, num_candidates, goal_dim))
        candidates_flat = candidates.reshape(batch_size * num_candidates, candidate_dim)

        first_v_logits = self.network.select('target_value')(
            obs_flat.reshape(batch_size * num_candidates, obs_dim),
            goals=candidates_flat,
        )
        second_v_logits = self.network.select('target_value')(
            candidates_flat,
            goals=goals_flat.reshape(batch_size * num_candidates, goal_dim),
        )
        first_v = self._aggregate_q_ensembles(jax.nn.sigmoid(first_v_logits)).reshape(batch_size, num_candidates)
        second_v = self._aggregate_q_ensembles(jax.nn.sigmoid(second_v_logits)).reshape(batch_size, num_candidates)
        return first_v * second_v, first_v, second_v

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
        u_s = self._encode_visual(batch['observations'], grad_params=grad_params)
        u_g = self._encode_visual(batch['value_goal_observations'], grad_params=grad_params)
        u_mid = self._encode_visual(batch['value_midpoint_observations'], grad_params=grad_params)
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
        actor_goal_u = self._encode_visual(batch['actor_goal_observations'])
        return {
            'actor_observations': actor_observations,
            'actor_goals': actor_goals,
            'actor_goal_u': actor_goal_u,
        }

    def _prepare_proposal_ctx(self, batch):
        """Cache proposal-conditioning encodings using the non-grad path."""
        u_s = self._encode_visual(batch['observations'])
        u_g = self._encode_visual(batch['value_goal_observations'])
        return {
            'u_s': u_s,
            'u_g': u_g,
        }

    def _evaluate_proposed_q_values(self, u_s, u_g, z_proposed):
        """Evaluate proposed midpoint candidates with factorized V(s, w) * V(w, g)."""
        if z_proposed.ndim == 2:
            first_v_logits = self.network.select('target_value')(u_s, goals=z_proposed)
            second_v_logits = self.network.select('target_value')(z_proposed, goals=u_g)
            first_v = self._aggregate_q_ensembles(jax.nn.sigmoid(first_v_logits))
            second_v = self._aggregate_q_ensembles(jax.nn.sigmoid(second_v_logits))
            return first_v * second_v

        num_cf_samples = z_proposed.shape[0]
        obs_bc = jnp.broadcast_to(u_s[None], (num_cf_samples, *u_s.shape))
        goals_bc = jnp.broadcast_to(u_g[None], (num_cf_samples, *u_g.shape))
        z_flat = z_proposed.reshape(-1, z_proposed.shape[-1])
        first_v_logits = self.network.select('target_value')(
            obs_bc.reshape(-1, u_s.shape[-1]),
            goals=z_flat,
        )
        second_v_logits = self.network.select('target_value')(
            z_flat,
            goals=goals_bc.reshape(-1, u_g.shape[-1]),
        )
        first_v = self._aggregate_q_ensembles(jax.nn.sigmoid(first_v_logits)).reshape(num_cf_samples, -1)
        second_v = self._aggregate_q_ensembles(jax.nn.sigmoid(second_v_logits)).reshape(num_cf_samples, -1)
        return (first_v * second_v).max(axis=0)

    def _value_loss_grounded(self, batch, grad_params, rng=None, value_ctx=None, proposal_ctx=None):
        """Use grounded factorized-V targets plus proposal-side CF stitching."""
        if 'value_goals_is_intraj' not in batch:
            raise ValueError('cf_trl requires value_goals_is_intraj in the value batch.')
        if value_ctx is None:
            value_ctx = self._prepare_value_ctx(batch, grad_params)
        u_s = value_ctx['u_s']
        u_g = value_ctx['u_g']
        v_logits = value_ctx['v_logits']
        vs = value_ctx['vs']
        direct_target_ens = value_ctx['direct_target_ens']
        direct_target = self._aggregate_q_ensembles(direct_target_ens)
        target_traj = direct_target
        is_intraj = batch['value_goals_is_intraj']

        proposed_beats_traj = None
        z_proposed = self._sample_z_proposal(
            batch,
            rng,
            num_samples=self._get_cf_num_z_proposals(),
            observations=None if proposal_ctx is None else proposal_ctx['u_s'],
            goals=None if proposal_ctx is None else proposal_ctx['u_g'],
        )
        z_proposed = jax.lax.stop_gradient(z_proposed)
        target_proposed = self._evaluate_proposed_q_values(u_s, u_g, z_proposed)
        proposed_beats_traj = target_proposed > target_traj

        intraj_target = jnp.maximum(target_traj, target_proposed)
        target = jnp.where(is_intraj, intraj_target, target_proposed)

        tau = jnp.where(
            is_intraj,
            self.config['expectile'],
            self.config.get('cf_expectile', self.config['expectile']),
        )
        expectile_weight = jnp.where(target >= vs, tau, (1 - tau))
        v_base = self._aggregate_q_ensembles(vs)
        dist_weight, implied_dist = self._distance_weight_from_value(v_base)
        v_loss = expectile_weight * dist_weight * self.bce_loss(v_logits, jax.lax.stop_gradient(target))
        total_loss = v_loss.mean()

        predicted_steps = jnp.log(vs + 1e-8) / jnp.log(self.config['discount'])
        actual_steps = batch['value_offsets']
        calib_mask = (actual_steps > 0).astype(jnp.float32)
        calib_mask = calib_mask * is_intraj
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
            'v_loss': total_loss,
            'v_mean': vs.mean(),
            'v_max': vs.max(),
            'v_min': vs.min(),
            'base_v_mean': v_base.mean(),
            'implied_dist_mean': implied_dist.mean(),
            'dist_weight_mean': dist_weight.mean(),
            'calibration_rel_gap_mean': calib_mean,
            'calibration_rel_gap_max': calib_max,
            'target_traj_mean': target_traj.mean(),
        }
        info['target_proposed_mean'] = target_proposed.mean()
        intraj_denom = jnp.maximum(is_intraj.sum(), 1.0)
        info['cf_override_frac_intraj'] = (
            proposed_beats_traj.astype(jnp.float32) * is_intraj
        ).sum() / intraj_denom
        info['cf_override_frac_all'] = proposed_beats_traj.astype(jnp.float32).mean()
        cf_mask = 1.0 - is_intraj

        def _masked_mean(x, mask):
            denom = jnp.maximum(mask.sum(), 1.0)
            return (x * mask).sum() / denom

        info['target_proposed_mean_intraj'] = _masked_mean(target_proposed, is_intraj)
        info['target_proposed_mean_cf'] = _masked_mean(target_proposed, cf_mask)
        info['target_traj_mean_intraj'] = _masked_mean(target_traj, is_intraj)
        info['target_traj_mean_cf'] = _masked_mean(target_traj, cf_mask)
        return total_loss, info

    def _z_proposal_intraj_loss(self, batch, grad_params, rng, step=0):
        """Train the z-proposal network with support-AWR over candidate midpoints."""
        rng, awr_rng = jax.random.split(rng)
        awr_beta = self.config.get('z_proposal_awr_beta', 0.0)
        random_support_k = int(self.config.get('z_proposal_random_goal_num_support', 0))
        awr_batch = self._select_proposal_training_batch(batch, step, awr_rng)
        intraj_mask = awr_batch.get('value_goals_is_intraj', None)
        flow_observations = self._encode_visual(awr_batch['observations'])
        flow_goals = self._encode_visual(awr_batch['value_goal_observations'])
        if intraj_mask is None:
            raise ValueError('cf_trl requires value_goals_is_intraj in the proposer batch.')
        z_target_intraj = self._encode_visual(awr_batch['value_midpoint_observations'])
        z_target_intraj = jax.lax.stop_gradient(z_target_intraj)
        random_goal_mask = (intraj_mask <= 0.5).astype(z_target_intraj.dtype)
        z_target = z_target_intraj
        support_scores = None
        best_support_scores = None
        proposal_mask = None

        if random_support_k > 0:
            if 'value_random_support_observations' not in awr_batch:
                raise ValueError('cf_trl requires value_random_support_observations when random support is enabled.')
            support_observations = awr_batch['value_random_support_observations']
            batch_size, num_support = support_observations.shape[:2]
            support_flat = support_observations.reshape(batch_size * num_support, *support_observations.shape[2:])
            support_targets = self._encode_visual(support_flat)
            support_targets = jax.lax.stop_gradient(support_targets)
            support_targets = support_targets.reshape(batch_size, num_support, support_targets.shape[-1])
            support_scores, _, _ = self._score_support_candidates(flow_observations, flow_goals, support_targets)
            best_support_idxs = jnp.argmax(support_scores, axis=1)
            best_support_targets = support_targets[jnp.arange(batch_size), best_support_idxs]
            best_support_scores = support_scores[jnp.arange(batch_size), best_support_idxs]
            z_target = jnp.where(random_goal_mask[:, None] > 0, best_support_targets, z_target_intraj)

        if awr_beta > 0.0:
            intraj_scores, _, _ = self._score_support_candidates(
                flow_observations,
                flow_goals,
                z_target_intraj[:, None, :],
            )
            midpoint_scores = intraj_scores[:, 0]
            if best_support_scores is not None:
                midpoint_scores = jnp.where(random_goal_mask > 0, best_support_scores, midpoint_scores)

            v_logits = self.network.select('target_value')(flow_observations, goals=flow_goals)
            v_base = self._aggregate_q_ensembles(jax.nn.sigmoid(v_logits))
            awr_adv = jax.lax.stop_gradient(midpoint_scores - v_base)
            awr_score = awr_adv / (v_base + 0.001)
            awr_weight = jnp.exp(awr_beta * awr_score).astype(z_target.dtype)

            awr_max_weight = self.config.get('z_proposal_awr_max_weight', 20.0)
            if awr_max_weight is not None and awr_max_weight > 0:
                awr_weight = jnp.minimum(awr_weight, awr_max_weight)
            has_positive = awr_adv > 0
            proposal_mask = awr_weight
        else:
            awr_adv = None
            awr_weight = None
            has_positive = None

        def vector_field_fn(noise, times):
            return self.network.select('z_proposal')(
                flow_observations, goals=flow_goals, actions=noise, times=times,
                params=grad_params,
            )

        loss, flow_info = imf_loss(
            rng, z_target, vector_field_fn,
            r_equals_t_prob=0.5,
            mask=proposal_mask,
        )

        info = {'loss': loss}
        if intraj_mask is not None:
            info['intraj_frac'] = intraj_mask.mean()
            info['random_goal_frac'] = 1.0 - intraj_mask.mean()
        if support_scores is not None:
            random_rows = random_goal_mask[:, None]
            info['random_support_score_mean'] = self._masked_mean(support_scores, random_rows)
            info['random_support_top1_score_mean'] = self._masked_mean(best_support_scores, random_goal_mask)
        if awr_adv is not None:
            info['awr_adv_mean'] = awr_adv.mean()
            info['awr_positive_frac'] = has_positive.astype(z_target.dtype).mean()
            info['awr_weight_mean'] = awr_weight.mean()
            info['awr_weight_max'] = awr_weight.max()
            info['awr_base_v_mean'] = v_base.mean()
            awr_max_weight = self.config.get('z_proposal_awr_max_weight', 20.0)
            if awr_max_weight is not None and awr_max_weight > 0:
                info['awr_weight_cap_frac'] = (awr_weight >= (awr_max_weight - 1e-6)).mean()

        for k, v in flow_info.items():
            if k != 'loss':
                info[k] = v

        return loss, info

    def _sample_z_proposal_with_module(
        self,
        module_name,
        rng,
        num_samples=1,
        observations=None,
        goals=None,
        grad_params=None,
    ):
        """Sample z candidates from a proposal module via iMF one-shot."""
        if observations is None or goals is None:
            raise ValueError('Proposal sampling requires explicit encoded observations and goals.')
        midpoint_dim = self._midpoint_dim()

        if num_samples == 1:
            sample_shape = (observations.shape[0], midpoint_dim)
        else:
            sample_shape = (num_samples, observations.shape[0], midpoint_dim)

        def _apply_module(obs, goal_inputs, noise, times):
            kwargs = dict(goals=goal_inputs, actions=noise, times=times)
            if grad_params is None:
                return self.network.select(module_name)(obs, **kwargs)
            return self.network.select(module_name)(obs, params=grad_params, **kwargs)

        def vector_field_fn(noise, times):
            if num_samples == 1:
                return _apply_module(observations, goals, noise, times)
            obs_bc = jnp.broadcast_to(observations[None], (num_samples, *observations.shape))
            goals_bc = jnp.broadcast_to(goals[None], (num_samples, *goals.shape))
            obs_flat = obs_bc.reshape(-1, observations.shape[-1])
            goals_flat = goals_bc.reshape(-1, goals.shape[-1])
            noise_flat = noise.reshape(-1, midpoint_dim)
            times_flat = times.reshape(-1, times.shape[-1])
            out = _apply_module(obs_flat, goals_flat, noise_flat, times_flat)
            return out.reshape(num_samples, observations.shape[0], midpoint_dim)

        return imf_one_shot_sample(rng, sample_shape, vector_field_fn)

    def _z_proposal_flow_q_loss(self, batch, grad_params, rng, step=0):
        """Train the z-proposal network by maximizing factorized midpoint value."""
        flow_q_batch = self._select_proposal_training_batch(batch, step, rng=None)
        flow_observations = self._encode_visual(flow_q_batch['observations'])
        flow_goals = self._encode_visual(flow_q_batch['value_goal_observations'])
        z_proposed = self._sample_z_proposal(
            flow_q_batch,
            rng,
            num_samples=1,
            observations=flow_observations,
            goals=flow_goals,
            grad_params=grad_params,
        )

        first_v_logits = self.network.select('target_value')(flow_observations, goals=z_proposed)
        second_v_logits = self.network.select('target_value')(z_proposed, goals=flow_goals)
        first_v = self._aggregate_q_ensembles(jax.nn.sigmoid(first_v_logits))
        second_v = self._aggregate_q_ensembles(jax.nn.sigmoid(second_v_logits))
        flow_q = first_v * second_v
        loss = -flow_q.mean()

        info = {
            'loss': loss,
            'flow_q_mean': flow_q.mean(),
            'flow_q_max': flow_q.max(),
            'flow_q_min': flow_q.min(),
            'flow_q_first_v_mean': first_v.mean(),
            'flow_q_second_v_mean': second_v.mean(),
            'sample_norm_mean': jnp.linalg.norm(z_proposed, axis=-1).mean(),
        }

        if 'value_goals_is_intraj' in flow_q_batch:
            intraj_mask = flow_q_batch['value_goals_is_intraj']
            cf_mask = 1.0 - intraj_mask
            info['intraj_frac'] = intraj_mask.mean()
            info['random_goal_frac'] = cf_mask.mean()
            info['flow_q_mean_intraj'] = self._masked_mean(flow_q, intraj_mask)
            info['flow_q_mean_cf'] = self._masked_mean(flow_q, cf_mask)

        return loss, info

    def z_proposal_loss(self, batch, grad_params, rng, step=0):
        method = self._get_z_proposal_learning_method()
        if method == 'intraj':
            return self._z_proposal_intraj_loss(batch, grad_params, rng, step=step)
        if method == 'flow_q':
            return self._z_proposal_flow_q_loss(batch, grad_params, rng, step=step)
        raise ValueError(f'Unsupported z_proposal_learning_method: {method}')

    def _sample_z_proposal(self, batch, rng, num_samples=1, observations=None, goals=None, grad_params=None):
        """Sample z candidates from the z-proposal network via iMF one-shot."""
        if observations is None:
            observations = self._encode_visual(batch['observations'])
        if goals is None:
            goals = self._encode_visual(batch['value_goal_observations'])
        return self._sample_z_proposal_with_module(
            'z_proposal',
            rng,
            num_samples=num_samples,
            observations=observations,
            goals=goals,
            grad_params=grad_params,
        )

    def q_action_loss(self, batch, grad_params, policy_ctx=None):
        """Distill Q_short(s, g, a) from an n-step bootstrap into V for FRS action selection."""
        q_action_n_step = int(self.config.get('q_action_n_step', 1))
        next_obs_key = 'actor_nstep_observations' if q_action_n_step > 1 and 'actor_nstep_observations' in batch else 'next_observations'
        u_next = self._encode_visual(batch[next_obs_key])
        u_g = policy_ctx['actor_goal_u'] if policy_ctx is not None else self._encode_visual(batch['actor_goal_observations'])

        v_next_logits = self.network.select('target_value')(
            u_next,
            goals=u_g,
        )
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

        if policy_ctx is None:
            q_action_observations, q_action_goals = self._encode_policy_inputs(
                batch['observations'],
                batch['actor_goals'],
                grad_params=grad_params,
            )
        else:
            q_action_observations = policy_ctx['actor_observations']
            q_action_goals = policy_ctx['actor_goals']
        q_action_logits = self.network.select('q_action')(
            q_action_observations,
            goals=q_action_goals,
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

    def actor_loss(self, batch, grad_params, rng=None, policy_ctx=None):
        """Flat actor loss (same policy extraction interface as TRL)."""
        pe_info = self._get_pe_info()

        if policy_ctx is None:
            actor_observations, actor_goals = self._encode_policy_inputs(
                batch['observations'],
                batch['actor_goals'],
                grad_params=grad_params,
            )
            actor_goal_u = self._encode_visual(batch['actor_goal_observations'])
        else:
            actor_observations = policy_ctx['actor_observations']
            actor_goals = policy_ctx['actor_goals']
            actor_goal_u = policy_ctx['actor_goal_u']

        if self.config['pe_type'] == 'rpg':
            dist = self.network.select('actor')(actor_observations, actor_goals, params=grad_params)

            u_next = self._encode_visual(batch['next_observations'])
            v_next = self.network.select('value')(u_next, goals=actor_goal_u)
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
            dist = self.network.select('actor')(actor_observations, actor_goals, params=grad_params)

            u_next = self._encode_visual(batch['next_observations'])
            v = self.network.select('value')(u_next, goals=actor_goal_u).mean(axis=0)
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
                actor_observations, actor_goals, x_t, t, params=grad_params,
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
        rng, value_rng, actor_rng, z_proposal_rng = jax.random.split(rng, 4)

        value_batch, using_intraj_warmup, value_goal_warmup_progress = self._select_value_training_batch(
            batch, step, rng=value_rng
        )
        info['value_goal_warmup_active'] = jnp.asarray(using_intraj_warmup, dtype=jnp.float32)
        info['value_goal_warmup_progress'] = jnp.asarray(value_goal_warmup_progress, dtype=jnp.float32)
        value_ctx = self._prepare_value_ctx(value_batch, grad_params)
        policy_ctx = self._prepare_policy_ctx(batch, grad_params)
        proposal_ctx = self._prepare_proposal_ctx(value_batch)

        value_loss, value_info = self._value_loss_grounded(
            value_batch, grad_params, rng=value_rng, value_ctx=value_ctx, proposal_ctx=proposal_ctx
        )
        for k, v in value_info.items():
            info[f'value/{k}'] = v

        z_proposal_loss, z_proposal_info = self.z_proposal_loss(
            value_batch, grad_params, rng=z_proposal_rng, step=step
        )
        for k, v in z_proposal_info.items():
            info[f'z_proposal/{k}'] = v

        q_action_loss, q_action_info = self.q_action_loss(batch, grad_params, policy_ctx=policy_ctx)
        for k, v in q_action_info.items():
            info[f'q_action/{k}'] = v

        actor_loss, actor_info = self.actor_loss(batch, grad_params, rng=actor_rng, policy_ctx=policy_ctx)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = (
            value_loss
            + z_proposal_loss
            + q_action_loss
            + actor_loss
        )
        finite_losses = jnp.stack(
            [
                jnp.asarray(loss, dtype=jnp.float32),
                jnp.asarray(value_loss, dtype=jnp.float32),
                jnp.asarray(z_proposal_loss, dtype=jnp.float32),
                jnp.asarray(q_action_loss, dtype=jnp.float32),
                jnp.asarray(actor_loss, dtype=jnp.float32),
            ]
        )

        def _raise(_):
            jax.debug.callback(
                self._raise_on_nonfinite_losses,
                jnp.asarray(step, dtype=jnp.int32),
                jnp.asarray(loss, dtype=jnp.float32),
                jnp.asarray(value_loss, dtype=jnp.float32),
                jnp.asarray(z_proposal_loss, dtype=jnp.float32),
                jnp.asarray(q_action_loss, dtype=jnp.float32),
                jnp.asarray(actor_loss, dtype=jnp.float32),
                ordered=True,
            )
            return None

        jax.lax.cond(jnp.all(jnp.isfinite(finite_losses)), lambda _: None, _raise, operand=None)
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
                actions = n_actions[jnp.argmax(q_action, axis=0), jnp.arange(observations.shape[0])]
            else:
                actions = n_actions[jnp.argmax(q_action)]

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
        cls._validate_config(config)
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
        state_dim = ex_encoded_observations.shape[-1]
        latent_dtype = ex_encoded_observations.dtype
        ex_u = ex_encoded_observations
        midpoint_dim = state_dim
        value_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=2,
        )
        q_action_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=1,
        )
        # z-proposal: iMF flow conditioned on (s, g) outputting z candidates.
        ex_z = jnp.zeros((ex_observations.shape[0], midpoint_dim), dtype=latent_dtype)
        ex_z_times = jnp.zeros((ex_observations.shape[0], 5), dtype=latent_dtype)  # iMF packs 5 time dims
        z_proposal_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=midpoint_dim,
            layer_norm=config['layer_norm'],
        )

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
            value=(value_def, (ex_u, ex_u)),
            target_value=(copy.deepcopy(value_def), (ex_u, ex_u)),
            q_action=(q_action_def, (ex_encoded_observations, ex_encoded_goals, ex_actions)),
            z_proposal=(z_proposal_def, (ex_u, ex_u, ex_z, ex_z_times)),
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

        config['action_dim'] = action_dim
        config['raw_observation_shape'] = raw_observation_shape
        config['state_dim'] = state_dim
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='cf_trl',
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
            encoder='',
            freeze_encoder=False,
            z_proposal_coef=1.0,
            z_proposal_learning_method='intraj',
            z_proposal_use_mixed_goals=True,
            z_proposal_awr_beta=1.0,
            z_proposal_awr_max_weight=20.0,
            cf_num_z_proposals=1,
            pe_type='frs',  # frs, rpg, discrete
            frs=ml_collections.ConfigDict(dict(flow_steps=10, num_samples=32)),
            rpg=ml_collections.ConfigDict(dict(alpha=0.03, const_std=True)),
            pe_discrete=ml_collections.ConfigDict(dict(alpha=0.03, action_ct=0)),
            discrete=False,
            dataset_class='GCDataset',
            value_p_curgoal=0.0,
            value_p_trajgoal=0.8,
            value_p_randomgoal=0.2,
            value_geom_sample=True,
            use_dual_value_goals=True,
            value_goal_warmup_steps=0,
            value_goal_mix_ramp_steps=0,
            value_goal_mix_start_trajgoal=0.8,
            value_goal_mix_start_randomgoal=0.2,
            z_proposal_random_goal_num_support=0,
            q_action_n_step=25,
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
