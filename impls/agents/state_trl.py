import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from agents.qaction_pe import QActionPolicyExtractionMixin
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import ActorVectorField, GCActor, GCValue


class StateTRLAgent(flax.struct.PyTreeNode, QActionPolicyExtractionMixin):
    """State-space TRL baseline with midpoint-supervised value learning."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def _distance_weight_from_value(self, base_v):
        if self.config.get('lam', 0.0) <= 0:
            ones = jnp.ones_like(base_v)
            return ones, jnp.zeros_like(base_v)
        safe_v = jnp.clip(base_v, 1e-8, 1.0)
        implied_dist = jnp.log(safe_v) / jnp.log(self.config['discount'])
        dist_weight = (1.0 / (1.0 + implied_dist)) ** self.config['lam']
        return dist_weight, implied_dist

    def value_loss(self, batch, grad_params):
        v_logits = self.network.select('value')(
            batch['observations'],
            goals=batch['value_goals'],
            params=grad_params,
        )
        vs = jax.nn.sigmoid(v_logits)

        first_v_logits = self.network.select('target_value')(
            batch['observations'],
            goals=batch['value_midpoint_goals'],
        )
        first_v = jnp.where(
            (batch['value_midpoint_offsets'] <= 1)[None, ...],
            self.config['discount'] ** batch['value_midpoint_offsets'][None, ...],
            jax.nn.sigmoid(first_v_logits),
        )

        second_v_logits = self.network.select('target_value')(
            batch['value_midpoint_observations'],
            goals=batch['value_goals'],
        )
        second_offset = batch['value_offsets'][None, ...] - batch['value_midpoint_offsets']
        second_v = jnp.where(
            (second_offset <= 1)[None, ...],
            self.config['discount'] ** second_offset,
            jax.nn.sigmoid(second_v_logits),
        )
        intraj_target = first_v * second_v

        if 'value_goals_is_intraj' in batch:
            is_intraj = batch['value_goals_is_intraj'][None, ...]
            cf_target = jnp.zeros_like(intraj_target)
            target = jnp.where(is_intraj > 0, intraj_target, cf_target)
        else:
            cf_target = None
            target = intraj_target
        tau = self.config['expectile']

        expectile_weight = jnp.where(target >= vs, tau, (1 - tau))
        base_v = jnp.minimum(vs[0], vs[1])
        dist_weight, implied_dist = self._distance_weight_from_value(base_v)
        value_loss = (expectile_weight * dist_weight * self.bce_loss(v_logits, jax.lax.stop_gradient(target))).mean()

        info = {
            'value_loss': value_loss,
            'v_mean': vs.mean(),
            'v_max': vs.max(),
            'v_min': vs.min(),
            'target_mean': target.mean(),
            'intraj_target_mean': intraj_target.mean(),
            'base_v_mean': base_v.mean(),
            'implied_dist_mean': implied_dist.mean(),
            'dist_weight_mean': dist_weight.mean(),
        }
        if cf_target is not None:
            info['cf_target_mean'] = cf_target.mean()
            info['cf_frac'] = (1.0 - batch['value_goals_is_intraj']).mean()
        return value_loss, info

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        info = {}
        rng = rng if rng is not None else self.rng

        value_loss, value_info = self.value_loss(batch, grad_params)
        for k, v in value_info.items():
            info[f'value/{k}'] = v

        q_action_loss, q_action_info = self.q_action_loss(batch, grad_params)
        for k, v in q_action_info.items():
            info[f'q_action/{k}'] = v

        rng, actor_rng = jax.random.split(rng)
        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = value_loss + q_action_loss + actor_loss
        return loss, info

    @jax.jit
    def update(self, batch):
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, 'value')
        return self.replace(network=new_network, rng=new_rng), info

    @classmethod
    def create(cls, seed, example_batch, config):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_observations = example_batch['observations']
        ex_actions = example_batch['actions']
        ex_actor_goals = example_batch['actor_goals']
        ex_value_goals = example_batch['value_goals']
        ex_times = ex_actions[..., :1]
        action_dim = ex_actions.shape[-1]
        pe_info = cls._get_pe_info_from_config(config)

        value_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=2,
        )
        q_action_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            ensemble=False,
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
            raise ValueError(f"Unsupported pe_type for StateTRLAgent: {config['pe_type']}")

        network_info = dict(
            value=(value_def, (ex_observations, ex_value_goals)),
            target_value=(copy.deepcopy(value_def), (ex_observations, ex_value_goals)),
            q_action=(q_action_def, (ex_observations, ex_actor_goals, ex_actions)),
            actor=(actor_def, ex_actor_in),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params['modules_target_value'] = params['modules_value']

        config['action_dim'] = action_dim
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    return ml_collections.ConfigDict(
        dict(
            agent_name='state_trl',
            lr=3e-4,
            batch_size=1024,
            actor_hidden_dims=(1024,) * 4,
            value_hidden_dims=(1024,) * 4,
            layer_norm=True,
            discount=0.999,
            tau=0.005,
            lam=0.0,
            expectile=0.7,
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
