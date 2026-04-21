import copy

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from agents.cf_trl import CFTRLAgent, get_config as get_cftrl_config
from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState
from utils.flows import _ensure_like, _pack_times
from utils.networks import ActorVectorField, GCValue


class CFTRLFlowQAgent(CFTRLAgent):
    """CF-TRL ablation with an intraj iMF prior plus a one-step flow-Q proposer."""

    @staticmethod
    def _validate_config(config):
        method = config.get('z_proposal_learning_method', 'flow_q')
        if method != 'flow_q':
            raise ValueError(
                "Invalid cf_trl_flowq config: cf_trl_flowq always uses the flow_q proposer objective"
            )
        if float(config.get('z_proposal_flow_alpha', 1.0)) < 0.0:
            raise ValueError(
                'Invalid cf_trl_flowq config: '
                f'z_proposal_flow_alpha must be non-negative, got {config.get("z_proposal_flow_alpha", 1.0)}'
            )
        if config.get('z_proposal_flow_regularizer_source', 'midpoint') != 'midpoint':
            raise ValueError(
                'Invalid cf_trl_flowq config: '
                'cf_trl_flowq always uses the in-trajectory iMF prior regularizer'
            )

        compat_config = copy.deepcopy(config)
        compat_config['z_proposal_learning_method'] = 'awr'
        CFTRLAgent._validate_config(compat_config)

    def _sample_z_one_step_with_module(
        self,
        module_name,
        rng,
        num_samples=1,
        observations=None,
        goals=None,
        grad_params=None,
    ):
        """Sample z candidates from a one-step proposal module."""
        if observations is None or goals is None:
            raise ValueError('Proposal sampling requires explicit encoded observations and goals.')
        midpoint_dim = self._midpoint_dim()

        if num_samples == 1:
            noises = jax.random.normal(rng, (observations.shape[0], midpoint_dim), dtype=observations.dtype)
            if grad_params is None:
                return self.network.select(module_name)(observations, goals=goals, actions=noises)
            return self.network.select(module_name)(
                observations,
                goals=goals,
                actions=noises,
                params=grad_params,
            )

        noises = jax.random.normal(
            rng,
            (num_samples, observations.shape[0], midpoint_dim),
            dtype=observations.dtype,
        )
        obs_bc = jnp.broadcast_to(observations[None], (num_samples, *observations.shape))
        goals_bc = jnp.broadcast_to(goals[None], (num_samples, *goals.shape))
        obs_flat = obs_bc.reshape(-1, observations.shape[-1])
        goals_flat = goals_bc.reshape(-1, goals.shape[-1])
        noises_flat = noises.reshape(-1, midpoint_dim)
        if grad_params is None:
            out = self.network.select(module_name)(obs_flat, goals=goals_flat, actions=noises_flat)
        else:
            out = self.network.select(module_name)(
                obs_flat,
                goals=goals_flat,
                actions=noises_flat,
                params=grad_params,
            )
        return out.reshape(num_samples, observations.shape[0], midpoint_dim)

    def _sample_z_proposal_prior_from_noises(
        self,
        observations,
        goals,
        noises,
        grad_params=None,
    ):
        """Evaluate the learned iMF prior at fixed Gaussian noises."""
        if noises.ndim == observations.ndim:
            obs_inputs = observations
            goal_inputs = goals
            noise_inputs = noises
        elif noises.ndim == observations.ndim + 1:
            num_samples = noises.shape[0]
            obs_inputs = jnp.broadcast_to(observations[None], (num_samples, *observations.shape)).reshape(
                -1, observations.shape[-1]
            )
            goal_inputs = jnp.broadcast_to(goals[None], (num_samples, *goals.shape)).reshape(-1, goals.shape[-1])
            noise_inputs = noises.reshape(-1, noises.shape[-1])
        else:
            raise ValueError(
                'Unexpected noise rank for prior sampling: '
                f'noises.ndim={noises.ndim}, observations.ndim={observations.ndim}'
            )

        r = jnp.zeros((*noise_inputs.shape[:-1], 1), dtype=noise_inputs.dtype)
        t = jnp.ones((*noise_inputs.shape[:-1], 1), dtype=noise_inputs.dtype)
        w = _ensure_like(None, t)
        times = _pack_times(t, r, w, jnp.zeros_like(t), jnp.ones_like(t))
        if grad_params is None:
            out = self.network.select('z_proposal')(
                obs_inputs,
                goals=goal_inputs,
                actions=noise_inputs,
                times=times,
            )
        else:
            out = self.network.select('z_proposal')(
                obs_inputs,
                goals=goal_inputs,
                actions=noise_inputs,
                times=times,
                params=grad_params,
            )
        if isinstance(out, tuple):
            out = out[0]
        z = noise_inputs - out
        if noises.ndim == observations.ndim + 1:
            return z.reshape(noises.shape[0], observations.shape[0], noises.shape[-1])
        return z

    def _z_proposal_flow_q_loss(self, batch, grad_params, rng, step=0):
        """Train an intraj iMF prior plus a one-step Q-improved proposer."""
        rng, prior_rng, noise_rng = jax.random.split(rng, 3)
        flow_q_batch = self._select_proposal_training_batch(batch, step, rng=None)
        flow_observations = self._encode_visual(flow_q_batch['observations'])
        flow_goals = self._encode_visual(flow_q_batch['value_goal_observations'])

        prior_loss, prior_info = self._compute_z_proposal_awr_loss(flow_q_batch, grad_params, prior_rng)

        midpoint_dim = self._midpoint_dim()
        noises = jax.random.normal(
            noise_rng,
            (flow_observations.shape[0], midpoint_dim),
            dtype=flow_observations.dtype,
        )
        z_prior = self._sample_z_proposal_prior_from_noises(flow_observations, flow_goals, noises)
        z_prior = jax.lax.stop_gradient(z_prior)
        z_proposed = self.network.select('z_onestep')(
            flow_observations,
            goals=flow_goals,
            actions=noises,
            params=grad_params,
        )

        distill_loss = jnp.mean((z_proposed - z_prior) ** 2)

        first_v_logits = self.network.select('target_value')(flow_observations, goals=z_proposed)
        second_v_logits = self.network.select('target_value')(z_proposed, goals=flow_goals)
        first_v = self._aggregate_q_ensembles(jax.nn.sigmoid(first_v_logits))
        second_v = self._aggregate_q_ensembles(jax.nn.sigmoid(second_v_logits))
        flow_q = first_v * second_v
        q_loss = -flow_q.mean()
        flow_alpha = float(self.config.get('z_proposal_flow_alpha', 1.0))
        loss = prior_loss + q_loss + flow_alpha * distill_loss

        info = {
            'loss': loss,
            'q_loss': q_loss,
            'flow_loss': distill_loss,
            'prior_loss': prior_loss,
            'distill_loss': distill_loss,
            'flow_alpha': jnp.asarray(flow_alpha, dtype=flow_q.dtype),
            'flow_q_mean': flow_q.mean(),
            'flow_q_max': flow_q.max(),
            'flow_q_min': flow_q.min(),
            'flow_q_first_v_mean': first_v.mean(),
            'flow_q_second_v_mean': second_v.mean(),
            'sample_norm_mean': jnp.linalg.norm(z_proposed, axis=-1).mean(),
            'prior_target_norm_mean': jnp.linalg.norm(z_prior, axis=-1).mean(),
        }

        if 'value_goals_is_intraj' in flow_q_batch:
            intraj_mask = flow_q_batch['value_goals_is_intraj']
            cf_mask = 1.0 - intraj_mask
            info['intraj_frac'] = intraj_mask.mean()
            info['random_goal_frac'] = cf_mask.mean()
            info['flow_q_mean_intraj'] = self._masked_mean(flow_q, intraj_mask)
            info['flow_q_mean_cf'] = self._masked_mean(flow_q, cf_mask)

        for k, v in prior_info.items():
            if k != 'loss':
                info[f'prior_{k}'] = v

        return loss, info

    def z_proposal_loss(self, batch, grad_params, rng, step=0):
        return self._z_proposal_flow_q_loss(batch, grad_params, rng, step=step)

    def _sample_z_proposal(self, batch, rng, num_samples=1, observations=None, goals=None, grad_params=None):
        """Sample z candidates from the one-step proposal network."""
        if observations is None:
            observations = self._encode_visual(batch['observations'])
        if goals is None:
            goals = self._encode_visual(batch['value_goal_observations'])
        return self._sample_z_one_step_with_module(
            'z_onestep',
            rng,
            num_samples=num_samples,
            observations=observations,
            goals=goals,
            grad_params=grad_params,
        )

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
        ex_z = jnp.zeros((ex_observations.shape[0], midpoint_dim), dtype=latent_dtype)
        ex_z_times = jnp.zeros((ex_observations.shape[0], 5), dtype=latent_dtype)
        z_proposal_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=midpoint_dim,
            layer_norm=config['layer_norm'],
        )
        z_onestep_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=midpoint_dim,
            layer_norm=config['layer_norm'],
        )
        actor_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            layer_norm=config['layer_norm'],
        )
        ex_actor_in = (ex_encoded_observations, ex_encoded_goals, ex_actions, ex_times)

        network_info = dict(
            value=(value_def, (ex_u, ex_u)),
            target_value=(copy.deepcopy(value_def), (ex_u, ex_u)),
            q_action=(q_action_def, (ex_encoded_observations, ex_encoded_goals, ex_actions)),
            z_proposal=(z_proposal_def, (ex_u, ex_u, ex_z, ex_z_times)),
            z_onestep=(z_onestep_def, (ex_u, ex_u, ex_z)),
            actor=(actor_def, ex_actor_in),
        )
        if encoder_def is not None:
            network_info['encoder'] = (encoder_def, (ex_observations,))

        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_params = network_def.init(init_rng, **network_args)['params']

        z_proposal_lr = float(config.get('z_proposal_lr', 0.0))
        if z_proposal_lr > 0.0 and not np.isclose(z_proposal_lr, config['lr']):
            def _label_from_path(path, _):
                top_key = getattr(path[0], 'key', None) if path else None
                if top_key in {'modules_z_proposal', 'modules_z_onestep'}:
                    return 'proposal'
                return 'default'

            param_labels = jax.tree_util.tree_map_with_path(_label_from_path, network_params)
            network_tx = optax.multi_transform(
                {
                    'default': optax.adam(learning_rate=config['lr']),
                    'proposal': optax.adam(learning_rate=z_proposal_lr),
                },
                param_labels,
            )
        else:
            network_tx = optax.adam(learning_rate=config['lr'])

        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network_params
        params['modules_target_value'] = params['modules_value']

        config['action_dim'] = action_dim
        config['raw_observation_shape'] = raw_observation_shape
        config['state_dim'] = state_dim
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = get_cftrl_config()
    config.agent_name = 'cf_trl_flowq'
    config.z_proposal_flow_alpha = 1.0
    return config
