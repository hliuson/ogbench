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
)

class SubgoalEncoder(nn.Module):
    """Subgoal encoder: s_j -> z_j."""

    hidden_dims: Sequence[int]
    layer_norm: bool
    z_dim: int

    @nn.compact
    def __call__(self, x):
        return MLP((*self.hidden_dims, self.z_dim), activate_final=False, layer_norm=self.layer_norm)(x)


class LatentTRLAgent(flax.struct.PyTreeNode):
    """Latent TRL agent with action-free V(s, g) and latent subgoal Q(s, z, g)."""

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

    def _encode_subgoal(self, midpoint_observations, grad_params=None):
        if grad_params is None:
            return self.network.select('subgoal_encoder')(midpoint_observations)
        return self.network.select('subgoal_encoder')(midpoint_observations, params=grad_params)

    @staticmethod
    def bce_loss(pred_logit, target):
        log_pred = jax.nn.log_sigmoid(pred_logit)
        log_not_pred = jax.nn.log_sigmoid(-pred_logit)
        loss = -(log_pred * target + log_not_pred * (1 - target))
        return loss

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

    def q_loss(self, batch, grad_params):
        """Compute Q(s, z_j, g) loss as BCE to the triangle inequality target."""
        goal_key = 'value_goal_observations' if self.config['oracle_distill'] else 'value_goals'
        midpoint_goal_key = 'value_midpoint_observations' if self.config['oracle_distill'] else 'value_midpoint_goals'

        z_mid = self._encode_subgoal(batch['value_midpoint_observations'], grad_params=grad_params)
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
            'q_loss': q_loss,
            'q_mean': qs.mean(),
            'q_max': qs.max(),
            'q_min': qs.min(),
        }

    def value_loss(self, batch, grad_params):
        """Compute the value loss using max/expectile over Q(s, z_j, g)."""
        goal_key = 'value_goal_observations' if self.config['oracle_distill'] else 'value_goals'
        v_logits = self.network.select('value')(
            batch['observations'],
            goals=batch[goal_key],
            params=grad_params,
        )
        vs = jax.nn.sigmoid(v_logits)

        z_mid = self._encode_subgoal(batch['value_midpoint_observations'])
        z_mid = jax.lax.stop_gradient(z_mid)
        q_logits = self.network.select('target_q')(
            batch['observations'],
            goals=batch[goal_key],
            actions=z_mid,
        )
        q = jax.nn.sigmoid(q_logits)
        if self.config['q_agg'] == 'min':
            target = jnp.minimum(q[0], q[1])
        elif self.config['q_agg'] == 'mean':
            target = q.mean(axis=0)

        expectile_weight = jnp.where(
            target >= vs,
            self.config['expectile'],
            (1 - self.config['expectile']),
        )
        dist = jax.lax.stop_gradient(jnp.log(target) / jnp.log(self.config['discount']))
        dist_weight = (1 / (1 + dist)) ** self.config['lam']
        v_loss = expectile_weight * dist_weight * self.bce_loss(v_logits, target)

        total_loss = v_loss.mean()

        return total_loss, {
            'total_loss': total_loss,
            'v_loss': v_loss,
            'v_mean': vs.mean(),
            'v_max': vs.max(),
            'v_min': vs.min(),
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
                'v_loss': v_loss,
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
                'v_loss': v_loss,
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

        value_loss, value_info = self.value_loss(batch, grad_params)
        for k, v in value_info.items():
            info[f'value/{k}'] = v

        q_loss, q_info = self.q_loss(batch, grad_params)
        for k, v in q_info.items():
            info[f'q/{k}'] = v

        q_short_loss, q_short_info = self.q_short_loss(batch, grad_params)
        for k, v in q_short_info.items():
            info[f'q_short/{k}'] = v

        rng, actor_rng = jax.random.split(rng)
        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = value_loss + q_loss + q_short_loss + actor_loss
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
        ex_times = ex_actions[..., :1]
        action_dim = ex_actions.shape[-1]
        pe_info = cls._get_pe_info_from_config(config)
        ex_z = jnp.zeros((ex_observations.shape[0], config['z_dim']), dtype=ex_observations.dtype)

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
        subgoal_encoder_def = SubgoalEncoder(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            z_dim=config['z_dim'],
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

        ex_value_goals = ex_observations if config['oracle_distill'] else ex_goals
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

        config['action_dim'] = action_dim
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
            oracle_distill=False,
            q_agg='min',
            z_dim=8,
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
