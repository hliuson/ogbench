import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from agents.latent_trl import LatentTRLAgent, SubgoalEncoder, get_config as latent_trl_get_config
from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState
from utils.networks import ActorVectorField, GCActor, GCDiscreteActor, GCValue


class LatentTRLStateDecoderAgent(LatentTRLAgent):
    """Latent TRL variant that drops the state VAE and decodes raw states from z via iMF."""

    def _get_reg_coef(self):
        return 0.0

    def _encode_state(self, observations, grad_params=None, rng=None, sample=False):
        del rng, sample
        return self._encode_visual(observations, grad_params=grad_params)

    def _state_vae_loss(self, batch, grad_params, rng):
        del batch, grad_params, rng
        zero = jnp.asarray(0.0, dtype=jnp.float32)
        return zero, {
            'loss': zero,
            'recon_loss': zero,
            'kl_loss': zero,
        }

    def _sample_midpoint_decoder(self, z, rng, num_samples=1, dtype=jnp.float32):
        """Sample raw midpoint states from the unconditional decoder p(w_mid | z)."""
        state_dim = int(self.config['state_dim'])
        batch_size = z.shape[0] if num_samples == 1 else z.shape[1]
        observations = jnp.zeros((batch_size, state_dim), dtype=dtype)
        decoder_goals = z if num_samples == 1 else None

        if num_samples == 1:
            sample_shape = (batch_size, state_dim)
        else:
            sample_shape = (num_samples, batch_size, state_dim)

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
            noise_flat = noise.reshape(-1, state_dim)
            times_flat = times.reshape(-1, times.shape[-1])
            out = self.network.select('midpoint_decoder')(
                obs_flat,
                goals=goals_flat,
                actions=noise_flat,
                times=times_flat,
            )
            return out.reshape(num_samples, observations.shape[0], state_dim)

        from utils.flows import imf_one_shot_sample

        return imf_one_shot_sample(rng, sample_shape, vector_field_fn)

    @classmethod
    def create(cls, seed, example_batch, config):
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
        ex_state = ex_encoded_observations
        ex_zero_state = jnp.zeros_like(ex_state)
        ex_z = jnp.zeros((ex_observations.shape[0], config['z_dim']), dtype=latent_dtype)

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
        q_action_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=1,
        )
        subgoal_encoder_def = SubgoalEncoder(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            z_dim=config['z_dim'],
        )

        ex_z_times = jnp.zeros((ex_observations.shape[0], 5), dtype=latent_dtype)
        z_proposal_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=config['z_dim'],
            layer_norm=config['layer_norm'],
        )
        midpoint_decoder_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=obs_dim,
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
            subgoal_encoder=(subgoal_encoder_def, (ex_state,)),
            value=(value_def, (ex_state, ex_state)),
            target_value=(copy.deepcopy(value_def), (ex_state, ex_state)),
            q=(q_def, (ex_state, ex_state, ex_z)),
            target_q=(copy.deepcopy(q_def), (ex_state, ex_state, ex_z)),
            q_action=(q_action_def, (ex_encoded_observations, ex_encoded_goals, ex_actions)),
            z_proposal=(z_proposal_def, (ex_state, ex_state, ex_z, ex_z_times)),
            actor=(actor_def, ex_actor_in),
            midpoint_decoder=(midpoint_decoder_def, (ex_zero_state, ex_z, ex_state, ex_z_times)),
        )
        if encoder_def is not None:
            network_info['encoder'] = (encoder_def, (ex_observations,))

        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params['modules_target_value'] = params['modules_value']
        params['modules_target_q'] = params['modules_q']

        config['action_dim'] = action_dim
        config['raw_observation_shape'] = raw_observation_shape
        config['state_dim'] = obs_dim
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = latent_trl_get_config()
    config.agent_name = 'latent_trl_state_decoder'
    config.reg_coef = 0.0
    config.vae_recon_coef = 0.0
    return config
