"""Decoupled Q-Chunking (DQC) agent.

This implementation is based on the DQC paper which proposes using different chunk sizes
for policy and critic networks: policies with short chunk sizes are easier to learn,
while critics with long chunk sizes can speed up value learning.
"""

from functools import partial
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.encoders import GCEncoder, encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import ActorVectorField, GCValue, MLP


def ensemblize(cls, num_qs, out_axes=0, **kwargs):
    """Ensemblize a module."""
    import flax.linen as nn
    return nn.vmap(
        cls,
        variable_axes={'params': 0},
        split_rngs={'params': True},
        in_axes=None,
        out_axes=out_axes,
        axis_size=num_qs,
        **kwargs,
    )


def apply_bfn(sample_fn, score_fn, n):
    """Apply best-of-N sampling."""
    def fn(rng):
        y = jax.vmap(sample_fn)(jax.random.split(rng, n))
        scores = jax.vmap(score_fn)(y)
        indices = jnp.argmax(scores, axis=0)
        y_reshaped = y.reshape((n, -1, y.shape[-1]))
        batch_size = y_reshaped.shape[1]
        indices_reshaped = indices.reshape(-1)
        y_out = y_reshaped[indices_reshaped, jnp.arange(batch_size)].reshape((y.shape[1:]))
        return y_out
    return fn


class GCValueEnsemble(GCValue):
    """Goal-conditioned value with configurable ensemble size.

    Extends GCValue to support num_ensembles parameter instead of boolean ensemble.
    """

    num_ensembles: int = 2

    def setup(self):
        mlp_module = MLP
        if self.num_ensembles > 1:
            mlp_module = ensemblize(mlp_module, self.num_ensembles)
        value_net = mlp_module((*self.hidden_dims, 1), activate_final=False, layer_norm=self.layer_norm)
        self.value_net = value_net


class DQCAgent(flax.struct.PyTreeNode):
    """Decoupled Q-Chunking (DQC) agent.

    DQC uses different chunk sizes for policy and critic networks, enabling
    faster value learning with long chunk critics while keeping policies
    easy to learn with short chunks.
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    @staticmethod
    def bce_loss(pred_logit, target):
        """Compute the BCE loss."""
        log_pred = jax.nn.log_sigmoid(pred_logit)
        log_not_pred = jax.nn.log_sigmoid(-pred_logit)
        loss = -(log_pred * target + log_not_pred * (1 - target))
        return loss

    def chunk_critic_loss(self, batch, grad_params, rng):
        """Compute the chunk critic loss."""
        next_v = self.network.select('value')(
            batch['high_value_next_observations'],
            goals=batch['high_value_goals']
        )
        next_v = jax.nn.sigmoid(next_v)

        target_v = batch['high_value_rewards'] + \
            (self.config['discount'] ** batch['high_value_backup_horizon']) * batch['high_value_masks'] * next_v
        target_v = jnp.clip(target_v, 0, 1)

        q_logit = self.network.select('chunk_critic')(
            batch['observations'],
            goals=batch['high_value_goals'],
            actions=batch['high_value_action_chunks'],
            params=grad_params
        )
        q = jax.nn.sigmoid(q_logit)
        critic_loss = self.bce_loss(q_logit, target_v).mean()

        return critic_loss, {
            'critic_loss': critic_loss,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
        }

    def action_critic_loss(self, batch, grad_params, rng):
        """Compute the action critic loss with distillation from chunk critic."""
        if self.config['use_chunk_critic']:
            target_v = self.network.select('chunk_critic')(
                batch['observations'],
                goals=batch['high_value_goals'],
                actions=batch['high_value_action_chunks']
            )
            target_v = jax.nn.sigmoid(target_v)
        else:
            next_v = self.network.select('value')(
                batch['high_value_next_observations'],
                goals=batch['high_value_goals']
            )
            next_v = jax.nn.sigmoid(next_v)
            target_v = batch['high_value_rewards'] + \
                (self.config['discount'] ** batch['high_value_backup_horizon']) * batch['high_value_masks'] * next_v
            target_v = jnp.clip(target_v, 0, 1)

        q_logit = self.network.select('action_critic')(
            batch['observations'],
            goals=batch['high_value_goals'],
            actions=batch['high_value_action_chunks'][..., :self.config['ac_action_dim']],
            params=grad_params
        )

        q = jax.nn.sigmoid(q_logit)
        clipped_target_v = jnp.clip(target_v, 1e-5, 1. - 1e-5)
        target_v_logit = jnp.log(clipped_target_v) - jnp.log(1. - clipped_target_v)

        weight = jnp.where(target_v >= q, self.config['kappa_d'], (1 - self.config['kappa_d']))

        if self.config['distill_method'] == 'expectile':
            critic_loss = (weight * self.bce_loss(q_logit, target_v) * batch['valids'][..., self.config['policy_chunk_size'] - 1]).mean()
        elif self.config['distill_method'] == 'quantile':
            critic_loss = (weight * jnp.abs(q_logit - target_v_logit) * batch['valids'][..., self.config['policy_chunk_size'] - 1]).mean()
        else:
            raise NotImplementedError(f"Unknown distill_method: {self.config['distill_method']}")

        total_loss = critic_loss
        info = {'critic_loss': critic_loss, 'q_mean': q.mean(), 'q_max': q.max(), 'q_min': q.min()}

        # Value function update with implicit backup.
        ex_actions = batch['high_value_action_chunks'][..., :self.config['ac_action_dim']]
        ex_qs = self.network.select('target_action_critic')(
            batch['observations'],
            goals=batch['high_value_goals'],
            actions=ex_actions
        )
        ex_qs = jax.nn.sigmoid(ex_qs)

        if self.config['q_agg'] == 'mean':
            ex_q = ex_qs.mean(axis=0)
        else:
            ex_q = ex_qs.min(axis=0)

        ex_q_logit = jnp.log(ex_q + 1e-8) - jnp.log(1. - ex_q + 1e-8)

        v_logit = self.network.select('value')(
            batch['observations'],
            goals=batch['high_value_goals'],
            params=grad_params
        )
        v = jax.nn.sigmoid(v_logit)

        if self.config['implicit_backup_type'] == 'expectile':
            weight = jnp.where(ex_q >= v, self.config['kappa_b'], (1 - self.config['kappa_b']))
            value_loss = (weight * self.bce_loss(v_logit, ex_q) * batch['valids'][..., self.config['policy_chunk_size'] - 1]).mean()
        elif self.config['implicit_backup_type'] == 'quantile':
            weight = jnp.where(ex_q >= v, self.config['kappa_b'], (1 - self.config['kappa_b']))
            value_loss = (weight * jnp.abs(v_logit - ex_q_logit) * batch['valids'][..., self.config['policy_chunk_size'] - 1]).mean()
        else:
            raise NotImplementedError(f"Unknown implicit_backup_type: {self.config['implicit_backup_type']}")

        total_loss += value_loss
        info.update({
            'value_loss': value_loss,
            'adv': (ex_q - v).mean(),
            'v_mean': v.mean(),
            'v_max': v.max(),
            'v_min': v.min()
        })

        return total_loss, info

    def actor_loss(self, batch, grad_params, rng):
        """Compute the flow-matching actor loss."""
        batch_size = batch['actions'].shape[0]
        rng, x_rng, t_rng = jax.random.split(rng, 3)

        # BC flow loss.
        x_0 = jax.random.normal(x_rng, (batch_size, self.config['ac_action_dim']))
        x_1 = batch['high_value_action_chunks'][..., :self.config['ac_action_dim']]
        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0

        pred = self.network.select('actor_bc')(
            batch['observations'],
            actions=x_t,
            times=t,
            params=grad_params
        )
        bc_flow_loss = jnp.mean(
            jnp.mean(jnp.square(pred - vel), axis=-1) * batch['valids'][..., self.config['policy_chunk_size'] - 1]
        )

        return bc_flow_loss, {'bc_flow_loss': bc_flow_loss}

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        rng, actor_rng, action_critic_rng, chunk_critic_rng = jax.random.split(rng, 4)

        if self.config['use_chunk_critic']:
            chunk_critic_loss, chunk_critic_info = self.chunk_critic_loss(batch, grad_params, chunk_critic_rng)
            for k, v in chunk_critic_info.items():
                info[f'chunk_critic/{k}'] = v

        action_critic_loss, action_critic_info = self.action_critic_loss(batch, grad_params, action_critic_rng)
        for k, v in action_critic_info.items():
            info[f'action_critic/{k}'] = v

        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = (chunk_critic_loss if self.config['use_chunk_critic'] else 0) + action_critic_loss + actor_loss
        return loss, info

    def target_update(self, network, module_name):
        """Update the target network."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @jax.jit
    def update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, 'action_critic')

        return self.replace(network=new_network, rng=new_rng), info

    @partial(jax.jit, static_argnames='best_of_n_override')
    def sample_actions(
        self,
        observations,
        goals=None,
        seed=None,
        temperature=1.0,
        best_of_n_override=None,
    ):
        """Sample actions from the flow-based actor with best-of-N selection."""
        def sample_fn(key):
            noises = jax.random.normal(
                key,
                (*observations.shape[:-1], self.config['ac_action_dim'])
            )
            actions = self.compute_flow_actions(observations, noises)
            return actions

        def score_fn(actions):
            if self.config['q_agg'] == 'mean':
                q = self.network.select('action_critic')(
                    observations, goals=goals, actions=actions
                ).mean(axis=0)
            elif self.config['q_agg'] == 'min':
                q = self.network.select('action_critic')(
                    observations, goals=goals, actions=actions
                ).min(axis=0)
            return q

        best_of_n = self.config['best_of_n'] if best_of_n_override is None else best_of_n_override
        bfn_sample_fn = apply_bfn(sample_fn, score_fn, best_of_n)
        actions = bfn_sample_fn(seed)

        # Only return the first action (for policy_chunk_size=1) or all actions
        if self.config['policy_chunk_size'] == 1:
            return actions[..., :self.config['action_dim']]
        else:
            return actions

    @jax.jit
    def compute_flow_actions(self, observations, noises, goals=None):
        """Compute actions using flow integration."""
        actions = noises
        for i in range(self.config['flow_steps']):
            t = jnp.full((*observations.shape[:-1], 1), i / self.config['flow_steps'])
            vels = self.network.select('actor_bc')(
                observations, actions=actions, goals=goals, times=t, is_encoded=True
            )
            actions = actions + vels / self.config['flow_steps']
        actions = jnp.clip(actions, -1, 1)
        return actions

    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,
        config,
    ):
        """Create a new DQC agent.

        Args:
            seed: Random seed.
            ex_observations: Example batch of observations.
            ex_actions: Example batch of actions.
            config: Configuration dictionary.
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_goals = ex_observations
        action_dim = ex_actions.shape[-1]
        ac_action_dim = config['policy_chunk_size'] * action_dim
        ex_action_chunks = jnp.zeros((*ex_actions.shape[:-1], config['backup_horizon'] * action_dim))
        ex_action_low_chunks = ex_action_chunks[..., :ac_action_dim]
        ex_times = ex_actions[..., :1]

        # Define encoders.
        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['value'] = GCEncoder(concat_encoder=encoder_module())
            encoders['chunk_critic'] = GCEncoder(concat_encoder=encoder_module())
            encoders['action_critic'] = GCEncoder(concat_encoder=encoder_module())
            encoders['actor'] = GCEncoder(concat_encoder=encoder_module())

        # Define networks.
        chunk_critic_def = GCValueEnsemble(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=config['num_qs'],
            gc_encoder=encoders.get('chunk_critic'),
        )

        value_def = GCValueEnsemble(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=1,
            gc_encoder=encoders.get('value'),
        )

        action_critic_def = GCValueEnsemble(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=config['num_qs'],
            gc_encoder=encoders.get('action_critic'),
        )

        target_action_critic_def = GCValueEnsemble(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=config['num_qs'],
            gc_encoder=encoders.get('action_critic'),
        )

        actor_bc_flow_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=ac_action_dim,
            layer_norm=config['actor_layer_norm'],
            gc_encoder=encoders.get('actor'),
        )

        network_info = dict(
            action_critic=(action_critic_def, (ex_observations, ex_goals, ex_action_low_chunks)),
            target_action_critic=(target_action_critic_def, (ex_observations, ex_goals, ex_action_low_chunks)),
            actor_bc=(actor_bc_flow_def, (ex_observations, None, ex_action_low_chunks, ex_times)),
            value=(value_def, (ex_observations, ex_goals)),
        )
        if config['use_chunk_critic']:
            network_info['chunk_critic'] = (chunk_critic_def, (ex_observations, ex_goals, ex_action_chunks))

        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params['modules_target_action_critic'] = params['modules_action_critic']

        # Store computed dimensions in config.
        config = dict(config)
        config['action_dim'] = action_dim
        config['ac_action_dim'] = ac_action_dim

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            # Agent hyperparameters.
            agent_name='dqc',  # Agent name.
            lr=3e-4,  # Learning rate.
            batch_size=4096,  # Batch size.
            actor_hidden_dims=(1024, 1024, 1024, 1024),  # Policy network hidden dimensions.
            value_hidden_dims=(1024, 1024, 1024, 1024),  # Value network hidden dimensions.
            layer_norm=True,  # Whether to use layer normalization for the critic(s).
            actor_layer_norm=True,  # Whether to use layer normalization for the policy.
            discount=0.999,  # Discount factor.
            tau=0.005,  # Target network update rate.
            num_qs=2,  # Number of Q ensembles.
            q_agg='mean',  # Aggregation function for Q values ('mean' or 'min').
            flow_steps=10,  # Number of flow steps for the policy.
            # DQC horizon parameters.
            use_chunk_critic=True,  # Whether to use a separate chunked critic.
            backup_horizon=25,  # Backing up value from a couple of steps in the future.
            policy_chunk_size=1,  # Policy chunk size.
            # DQC backup and distillation parameters.
            distill_method='expectile',  # Implicit maximization loss for training the distilled critic.
            kappa_d=0.5,  # Implicit coefficient for distillation.
            implicit_backup_type='quantile',  # Implicit maximization loss for implicit value backup.
            kappa_b=0.9,  # Implicit value backup coefficient.
            best_of_n=32,  # Best-of-N policy extraction.
            discrete=False,  # Whether the action space is discrete.
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name.
            # Dataset hyperparameters.
            dataset_class='CGCDataset',  # Dataset class name.
            value_p_curgoal=0.2,  # Probability of using the current state as the value goal.
            value_p_trajgoal=0.5,  # Probability of using a future state in the same trajectory as the value goal.
            value_p_randomgoal=0.3,  # Probability of using a random state as the value goal.
            value_geom_sample=False,  # Whether to use geometric sampling for future value goals.
            gc_negative=True,  # Whether to use '0 if s == g else -1' (True) or '1 if s == g else 0' (False) as reward.
            p_aug=0.0,  # Probability of applying image augmentation.
            frame_stack=ml_collections.config_dict.placeholder(int),  # Number of frames to stack.
        )
    )
    return config
