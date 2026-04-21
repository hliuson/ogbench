import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax

from agents.qaction_pe import QActionPolicyExtractionMixin
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.flows import imf_loss, imf_one_shot_sample
from utils.networks import ActorVectorField, GCActor, GCValue


class DynaNStepGCIQLAgent(flax.struct.PyTreeNode, QActionPolicyExtractionMixin):
    """GCIQL with real n-step targets plus a mean-flow lookahead generator."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    @staticmethod
    def expectile_loss(adv, diff, expectile):
        weight = jnp.where(adv >= 0, expectile, (1 - expectile))
        return weight * (diff**2)

    @staticmethod
    def _validate_config(config):
        allowed_modes = {'real', 'model', 'augment'}
        mode = config.get('dyna_target_mode', 'augment')
        if mode not in allowed_modes:
            raise ValueError(f'Unsupported dyna_target_mode: {mode!r}. Expected one of {sorted(allowed_modes)}')
        if int(config.get('critic_n_step', 1)) < 1:
            raise ValueError(f"critic_n_step must be >= 1, got {config.get('critic_n_step', 1)}")
        if int(config.get('q_action_n_step', 1)) < 1:
            raise ValueError(f"q_action_n_step must be >= 1, got {config.get('q_action_n_step', 1)}")
        if int(config.get('dyna_num_samples', 1)) < 1:
            raise ValueError(f"dyna_num_samples must be >= 1, got {config.get('dyna_num_samples', 1)}")
        if float(config.get('dyna_loss_coef', 1.0)) < 0.0:
            raise ValueError(f"dyna_loss_coef must be non-negative, got {config.get('dyna_loss_coef', 1.0)}")
        if float(config.get('dyna_critic_coef', 1.0)) < 0.0:
            raise ValueError(f"dyna_critic_coef must be non-negative, got {config.get('dyna_critic_coef', 1.0)}")
        if float(config.get('dyna_lr', 0.0)) < 0.0:
            raise ValueError(f"dyna_lr must be non-negative, got {config.get('dyna_lr', 0.0)}")
        if config.get('pe_type', 'frs') not in {'frs', 'rpg'}:
            raise ValueError(f"Unsupported pe_type for DynaNStepGCIQLAgent: {config.get('pe_type')!r}")
        if config.get('discrete', False):
            raise ValueError('gciql_dyna_nstep only supports continuous actions')

    def _critic_next_obs_key(self, batch):
        critic_n_step = int(self.config.get('critic_n_step', 1))
        if critic_n_step > 1 and 'value_nstep_observations' in batch:
            return 'value_nstep_observations'
        return 'next_observations'

    def _critic_n_steps(self, batch):
        critic_n_step = int(self.config.get('critic_n_step', 1))
        if critic_n_step > 1 and 'value_nstep_steps' in batch:
            return batch['value_nstep_steps']
        return jnp.ones_like(batch['actions'][..., 0], dtype=jnp.int32)

    def _dyna_target_mode(self):
        return self.config.get('dyna_target_mode', 'augment')

    def _dyna_loss_coef(self):
        return float(self.config.get('dyna_loss_coef', 1.0))

    def _dyna_critic_coef(self):
        return float(self.config.get('dyna_critic_coef', 1.0))

    def _dyna_num_samples(self):
        return max(1, int(self.config.get('dyna_num_samples', 1)))

    def _uses_dyna_generator(self):
        return self._dyna_target_mode() != 'real' or self._dyna_loss_coef() > 0.0

    def _evaluate_target_value(self, next_observations, goals):
        if next_observations.ndim == goals.ndim:
            next_v = self.network.select('target_value')(next_observations, goals)
            next_v = jax.nn.sigmoid(next_v)
            if next_v.ndim > 1:
                next_v = jnp.minimum(next_v[0], next_v[1])
            return next_v

        num_samples, batch_size = next_observations.shape[:2]
        obs_flat = next_observations.reshape((num_samples * batch_size, *next_observations.shape[2:]))
        goals_bc = jnp.broadcast_to(goals[None], (num_samples, *goals.shape))
        goal_flat = goals_bc.reshape((num_samples * batch_size, *goals.shape[1:]))
        next_v = self.network.select('target_value')(obs_flat, goal_flat)
        next_v = jax.nn.sigmoid(next_v)
        if next_v.ndim > 1:
            next_v = jnp.minimum(next_v[0], next_v[1])
        return next_v.reshape((num_samples, batch_size))

    def _critic_targets_from_next_observations(self, batch, next_observations):
        next_v = self._evaluate_target_value(next_observations, batch['value_goals'])
        n_steps = self._critic_n_steps(batch)
        discount = self.config['discount'] ** n_steps
        if next_v.ndim > 1:
            bootstrap_target = discount[None, :] * next_v
        else:
            bootstrap_target = discount * next_v

        if 'value_goals_is_intraj' in batch and 'value_offsets' in batch:
            is_intraj = batch['value_goals_is_intraj']
            offsets = batch['value_offsets']
            exact_reached = is_intraj * (offsets <= n_steps)
            exact_target = self.config['discount'] ** offsets
            if bootstrap_target.ndim > 1:
                q_target = jnp.where(exact_reached[None, :] > 0, exact_target[None, :], bootstrap_target)
            else:
                q_target = jnp.where(exact_reached > 0, exact_target, bootstrap_target)
        else:
            exact_reached = jnp.zeros_like(discount, dtype=jnp.float32)
            q_target = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v

        return q_target, exact_reached

    @staticmethod
    def _masked_mean(values, mask):
        mask = mask.astype(values.dtype)
        denom = jnp.maximum(mask.sum(), 1.0)
        return (values * mask).sum() / denom

    def _sample_dyna_candidates(self, observations, goals, rng, num_samples=1, grad_params=None):
        state_dim = int(self.config['state_dim'])
        batch_size = observations.shape[0]
        if num_samples == 1:
            sample_shape = (batch_size, state_dim)
        else:
            sample_shape = (num_samples, batch_size, state_dim)

        def _apply_module(obs, goal_inputs, noise, times):
            kwargs = dict(goals=goal_inputs, actions=noise, times=times)
            if grad_params is None:
                return self.network.select('dyna_generator')(obs, **kwargs)
            return self.network.select('dyna_generator')(obs, params=grad_params, **kwargs)

        def vector_field_fn(noise, times):
            if num_samples == 1:
                return _apply_module(observations, goals, noise, times)

            obs_bc = jnp.broadcast_to(observations[None], (num_samples, *observations.shape))
            goals_bc = jnp.broadcast_to(goals[None], (num_samples, *goals.shape))
            obs_flat = obs_bc.reshape((num_samples * batch_size, *observations.shape[1:]))
            goals_flat = goals_bc.reshape((num_samples * batch_size, *goals.shape[1:]))
            noise_flat = noise.reshape((num_samples * batch_size, noise.shape[-1]))
            times_flat = times.reshape((num_samples * batch_size, times.shape[-1]))
            out = _apply_module(obs_flat, goals_flat, noise_flat, times_flat)
            return out.reshape((num_samples, batch_size, state_dim))

        samples = imf_one_shot_sample(rng, sample_shape, vector_field_fn, dtype=observations.dtype)
        if num_samples == 1:
            samples = samples[None, ...]
        return samples

    def _critic_bce_loss_against_targets(self, q1_logits, q2_logits, targets):
        if targets.ndim == 1:
            return (
                self.bce_loss(q1_logits, jax.lax.stop_gradient(targets))
                + self.bce_loss(q2_logits, jax.lax.stop_gradient(targets))
            ).mean()

        q1_bc = jnp.broadcast_to(q1_logits[None, :], targets.shape)
        q2_bc = jnp.broadcast_to(q2_logits[None, :], targets.shape)
        return (
            self.bce_loss(q1_bc, jax.lax.stop_gradient(targets))
            + self.bce_loss(q2_bc, jax.lax.stop_gradient(targets))
        ).mean()

    def _dyna_training_targets(self, batch):
        next_observations = batch[self._critic_next_obs_key(batch)]
        train_mask = jnp.ones_like(batch['actions'][..., 0], dtype=jnp.float32)
        exact_reached = jnp.zeros_like(train_mask)

        if 'value_goals_is_intraj' not in batch or 'value_offsets' not in batch:
            return next_observations, train_mask, exact_reached

        is_intraj = batch['value_goals_is_intraj']
        n_steps = self._critic_n_steps(batch)
        exact_reached = is_intraj * (batch['value_offsets'] <= n_steps)
        goal_mask = exact_reached.astype(bool)
        goal_mask = goal_mask.reshape((goal_mask.shape[0],) + (1,) * (next_observations.ndim - 1))
        target_observations = jnp.where(goal_mask, batch['value_goals'], next_observations)
        if self.config.get('dyna_train_on_intraj_only', True):
            train_mask = is_intraj.astype(jnp.float32)
        return target_observations, train_mask, exact_reached

    def value_loss(self, batch, grad_params):
        q1_logits, q2_logits = self.network.select('target_critic')(
            batch['observations'],
            batch['value_goals'],
            batch['actions'],
        )
        q = jax.nn.sigmoid(jnp.minimum(q1_logits, q2_logits))
        v_logits = self.network.select('value')(
            batch['observations'],
            batch['value_goals'],
            params=grad_params,
        )
        v = jax.nn.sigmoid(v_logits)

        if 'value_goals_is_intraj' in batch:
            tau = jnp.where(
                batch['value_goals_is_intraj'] > 0,
                self.config['expectile'],
                self.config.get('cf_expectile', self.config['expectile']),
            )
        else:
            tau = self.config['expectile']
        expectile_weight = jnp.where(q >= v, tau, (1 - tau))
        value_loss = (expectile_weight * self.bce_loss(v_logits, jax.lax.stop_gradient(q))).mean()

        return value_loss, {
            'value_loss': value_loss,
            'v_mean': v.mean(),
            'v_max': v.max(),
            'v_min': v.min(),
            'q_mean': q.mean(),
        }

    def dyna_generator_loss(self, batch, grad_params, rng):
        if not self._uses_dyna_generator():
            zero = jnp.asarray(0.0, dtype=jnp.float32)
            return zero, {
                'loss': zero,
                'disabled': jnp.asarray(1.0, dtype=jnp.float32),
            }

        target_observations, train_mask, exact_reached = self._dyna_training_targets(batch)

        def vector_field_fn(noise, times):
            return self.network.select('dyna_generator')(
                batch['observations'],
                batch['value_goals'],
                noise,
                times,
                params=grad_params,
            )

        loss, flow_info = imf_loss(
            rng,
            target_observations,
            vector_field_fn,
            r_equals_t_prob=float(self.config.get('dyna_r_equals_t_prob', 0.5)),
            mask=train_mask,
        )
        info = {
            'loss': loss,
            'train_mask_frac': train_mask.mean(),
            'exact_goal_frac': exact_reached.mean(),
        }
        for k, v in flow_info.items():
            if k != 'loss':
                info[k] = v
        return loss, info

    def critic_loss(self, batch, grad_params, rng=None):
        real_next_observations = batch[self._critic_next_obs_key(batch)]
        real_target, exact_reached = self._critic_targets_from_next_observations(batch, real_next_observations)
        model_target_samples = None
        if self._dyna_target_mode() != 'real':
            if rng is None:
                raise ValueError('critic_loss requires rng when dyna_target_mode != "real"')
            model_next_observations = self._sample_dyna_candidates(
                batch['observations'],
                batch['value_goals'],
                rng,
                num_samples=self._dyna_num_samples(),
            )
            model_next_observations = jax.lax.stop_gradient(model_next_observations)
            model_target_samples, _ = self._critic_targets_from_next_observations(batch, model_next_observations)

        q1_logits, q2_logits = self.network.select('critic')(
            batch['observations'],
            batch['value_goals'],
            batch['actions'],
            params=grad_params,
        )

        critic_loss_real = self._critic_bce_loss_against_targets(q1_logits, q2_logits, real_target)
        if model_target_samples is not None:
            critic_loss_model = self._critic_bce_loss_against_targets(q1_logits, q2_logits, model_target_samples)
        else:
            critic_loss_model = jnp.asarray(0.0, dtype=jnp.float32)

        mode = self._dyna_target_mode()
        if mode == 'real':
            critic_loss = critic_loss_real
        elif mode == 'model':
            critic_loss = critic_loss_model
        elif mode == 'augment':
            critic_loss = critic_loss_real + self._dyna_critic_coef() * critic_loss_model
        else:
            raise ValueError(f'Unsupported dyna_target_mode: {mode!r}')

        info = {
            'critic_loss': critic_loss,
            'critic_loss_real': critic_loss_real,
            'critic_loss_model': critic_loss_model,
            'real_target_mean': real_target.mean(),
            'real_target_max': real_target.max(),
            'real_target_min': real_target.min(),
            'exact_reached_frac': exact_reached.mean(),
            'n_step': jnp.asarray(int(self.config.get('critic_n_step', 1)), dtype=jnp.float32),
            'dyna_enabled': jnp.asarray(model_target_samples is not None, dtype=jnp.float32),
            'dyna_num_samples': jnp.asarray(self._dyna_num_samples(), dtype=jnp.float32),
        }
        if mode == 'real':
            info['q_mean'] = real_target.mean()
            info['q_max'] = real_target.max()
            info['q_min'] = real_target.min()
        elif model_target_samples is not None:
            info['model_target_mean'] = model_target_samples.mean()
            info['model_target_max'] = model_target_samples.max()
            info['model_target_min'] = model_target_samples.min()
            info['q_mean'] = model_target_samples.mean() if mode == 'model' else real_target.mean()
            info['q_max'] = model_target_samples.max() if mode == 'model' else real_target.max()
            info['q_min'] = model_target_samples.min() if mode == 'model' else real_target.min()

            real_target_bc = real_target[None, :]
            model_beats_real = (model_target_samples > real_target_bc).astype(jnp.float32)
            info['model_minus_real_mean'] = (model_target_samples - real_target_bc).mean()
            info['model_beats_real_frac'] = model_beats_real.mean()
            if 'value_goals_is_intraj' in batch:
                intraj_mask = batch['value_goals_is_intraj']
                info['model_target_mean_intraj'] = self._masked_mean(model_target_samples.mean(axis=0), intraj_mask)
                info['model_target_mean_cf'] = self._masked_mean(model_target_samples.mean(axis=0), 1.0 - intraj_mask)
                info['model_beats_real_frac_intraj'] = self._masked_mean(model_beats_real.mean(axis=0), intraj_mask)
                info['model_beats_real_frac_cf'] = self._masked_mean(model_beats_real.mean(axis=0), 1.0 - intraj_mask)
        else:
            info['q_mean'] = real_target.mean()
            info['q_max'] = real_target.max()
            info['q_min'] = real_target.min()

        return critic_loss, info

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        info = {}
        rng = rng if rng is not None else self.rng

        rng, critic_rng, actor_rng, dyna_rng = jax.random.split(rng, 4)

        value_loss, value_info = self.value_loss(batch, grad_params)
        for k, v in value_info.items():
            info[f'value/{k}'] = v

        critic_loss, critic_info = self.critic_loss(batch, grad_params, rng=critic_rng)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        dyna_loss, dyna_info = self.dyna_generator_loss(batch, grad_params, rng=dyna_rng)
        for k, v in dyna_info.items():
            info[f'dyna/{k}'] = v

        q_action_loss, q_action_info = self.q_action_loss(batch, grad_params)
        for k, v in q_action_info.items():
            info[f'q_action/{k}'] = v

        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = value_loss + critic_loss + self._dyna_loss_coef() * dyna_loss + q_action_loss + actor_loss
        return loss, info

    @jax.jit
    def update(self, batch):
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, 'value')
        self.target_update(new_network, 'critic')
        return self.replace(network=new_network, rng=new_rng), info

    @classmethod
    def create(cls, seed, example_batch, config):
        cls._validate_config(config)
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_observations = example_batch['observations']
        ex_actions = example_batch['actions']
        ex_actor_goals = example_batch['actor_goals']
        ex_value_goals = example_batch['value_goals']
        ex_times = ex_actions[..., :1]
        action_dim = ex_actions.shape[-1]
        state_dim = ex_observations.shape[-1]
        pe_info = cls._get_pe_info_from_config(config)

        value_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            ensemble=False,
        )
        critic_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=2,
        )
        q_action_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            ensemble=False,
        )
        dyna_generator_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=state_dim,
            layer_norm=config['layer_norm'],
        )

        if config['pe_type'] == 'frs':
            actor_def = ActorVectorField(
                hidden_dims=config['actor_hidden_dims'],
                action_dim=action_dim,
                layer_norm=config['layer_norm'],
            )
            ex_actor_in = (ex_observations, ex_actor_goals, ex_actions, ex_times)
        elif config['pe_type'] == 'rpg':
            actor_def = GCActor(
                hidden_dims=config['actor_hidden_dims'],
                action_dim=action_dim,
                layer_norm=config['layer_norm'],
                state_dependent_std=False,
                const_std=pe_info['const_std'],
            )
            ex_actor_in = (ex_observations, ex_actor_goals, ex_actions)
        else:
            raise ValueError(f"Unsupported pe_type for DynaNStepGCIQLAgent: {config['pe_type']}")

        ex_dyna_target = jnp.zeros((ex_observations.shape[0], state_dim), dtype=ex_observations.dtype)
        ex_dyna_times = jnp.zeros((ex_observations.shape[0], 5), dtype=ex_observations.dtype)
        network_info = dict(
            value=(value_def, (ex_observations, ex_value_goals)),
            target_value=(copy.deepcopy(value_def), (ex_observations, ex_value_goals)),
            critic=(critic_def, (ex_observations, ex_value_goals, ex_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, ex_value_goals, ex_actions)),
            q_action=(q_action_def, (ex_observations, ex_actor_goals, ex_actions)),
            actor=(actor_def, ex_actor_in),
            dyna_generator=(dyna_generator_def, (ex_observations, ex_value_goals, ex_dyna_target, ex_dyna_times)),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_params = network_def.init(init_rng, **network_args)['params']

        dyna_lr = float(config.get('dyna_lr', 0.0))
        if dyna_lr > 0.0 and not np.isclose(dyna_lr, config['lr']):

            def _label_from_path(path, _):
                top_key = getattr(path[0], 'key', None) if path else None
                if top_key == 'modules_dyna_generator':
                    return 'dyna'
                return 'default'

            param_labels = jax.tree_util.tree_map_with_path(_label_from_path, network_params)
            network_tx = optax.multi_transform(
                {
                    'default': optax.adam(learning_rate=config['lr']),
                    'dyna': optax.adam(learning_rate=dyna_lr),
                },
                param_labels,
            )
        else:
            network_tx = optax.adam(learning_rate=config['lr'])

        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params['modules_target_value'] = params['modules_value']
        params['modules_target_critic'] = params['modules_critic']

        config['action_dim'] = action_dim
        config['state_dim'] = state_dim
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    return ml_collections.ConfigDict(
        dict(
            agent_name='gciql_dyna_nstep',
            lr=3e-4,
            batch_size=1024,
            actor_hidden_dims=(1024,) * 4,
            value_hidden_dims=(1024,) * 4,
            layer_norm=True,
            discount=0.999,
            tau=0.005,
            expectile=0.9,
            cf_expectile=0.9,
            critic_n_step=25,
            dyna_target_mode='augment',
            dyna_num_samples=4,
            dyna_loss_coef=1.0,
            dyna_critic_coef=1.0,
            dyna_train_on_intraj_only=True,
            dyna_r_equals_t_prob=0.5,
            dyna_lr=3e-3,
            pe_type='frs',
            frs=ml_collections.ConfigDict(dict(flow_steps=10, num_samples=32)),
            rpg=ml_collections.ConfigDict(dict(alpha=0.03, const_std=True)),
            discrete=False,
            dataset_class='GCDataset',
            need_value_intraj_mask=True,
            need_actor_nstep=True,
            q_action_target_is_bounded=True,
            q_action_n_step=25,
            value_p_curgoal=0.0,
            value_p_trajgoal=0.8,
            value_p_randomgoal=0.2,
            value_geom_sample=True,
            actor_p_curgoal=0.0,
            actor_p_trajgoal=0.5,
            actor_p_randomgoal=0.5,
            actor_geom_sample=True,
            gc_negative=False,
            p_aug=0.0,
            frame_stack=ml_collections.config_dict.placeholder(int),
        )
    )
