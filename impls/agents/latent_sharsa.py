import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax

import flax.linen as nn

from utils.encoders import encoder_modules, ResMLPEncoder
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import ActorVectorField, GCValue, MLP, ResMLPActorVectorField, ResMLPValue


class SubgoalEncoder(nn.Module):
    """Option-style subgoal encoder: (s, s', g) -> z using ResMLP + projection."""

    hidden_dim: int = 256
    num_blocks: int = 2
    z_dim: int = 8

    @nn.compact
    def __call__(self, x):
        x = ResMLPEncoder(hidden_dim=self.hidden_dim, num_blocks=self.num_blocks)(x)
        x = nn.Dense(self.z_dim)(x)
        return x


class GoalEncoder(nn.Module):
    """Encodes raw goal into a compact representation using ResMLP."""

    hidden_dim: int = 256
    num_blocks: int = 2
    output_dim: int = 64

    @nn.compact
    def __call__(self, x):
        x = ResMLPEncoder(hidden_dim=self.hidden_dim, num_blocks=self.num_blocks)(x)
        x = nn.Dense(self.output_dim)(x)
        return x


class LatentSHARSAAgent(flax.struct.PyTreeNode):
    """SHARSA agent with latent subgoals."""

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

    def _encode_observations(self, observations, grad_params=None):
        """Encode observations (identity if no encoder, otherwise use encoder)."""
        if self.config['encoder'] == 'none':
            return observations
        if self.config['freeze_encoder']:
            return self.network.select('encoder')(observations)
        else:
            return self.network.select('encoder')(observations, params=grad_params)

    def _encode_subgoal(self, observations, next_observations, goals, grad_params=None):
        """Encode (current state, next state, goal) into latent subgoal z (option-style)."""
        enc = self._encode_observations(observations, grad_params)
        enc_next = self._encode_observations(next_observations, grad_params)
        enc_input = jnp.concatenate([enc, enc_next, goals], axis=-1)
        return self.network.select('subgoal_encoder')(enc_input, params=grad_params)

    def high_value_loss(self, batch, grad_params):
        """Compute the high-level SARSA value loss."""
        # Option-style subgoal encoding: (s, s', g) -> z
        z_target = self._encode_subgoal(
            batch['observations'],
            batch['high_value_next_observations'],
            batch['high_value_goals'],
            grad_params,
        )
        z_target = jax.lax.stop_gradient(z_target)
        q1, q2 = self.network.select('target_high_critic')(
            batch['observations'], goals=batch['high_value_goals'], actions=z_target
        )
        if self.config['value_loss_type'] == 'bce':
            q1, q2 = jax.nn.sigmoid(q1), jax.nn.sigmoid(q2)

        if self.config['q_agg'] == 'min':
            q = jnp.minimum(q1, q2)
        elif self.config['q_agg'] == 'mean':
            q = (q1 + q2) / 2

        v = self.network.select('high_value')(batch['observations'], batch['high_value_goals'], params=grad_params)
        if self.config['value_loss_type'] == 'bce':
            v_logit = v
            v = jax.nn.sigmoid(v_logit)

        if self.config['value_loss_type'] == 'squared':
            value_loss = ((v - q) ** 2).mean()
        elif self.config['value_loss_type'] == 'bce':
            value_loss = (self.bce_loss(v_logit, q)).mean()

        return value_loss, {
            'value_loss': value_loss,
            'v_mean': v.mean(),
            'v_max': v.max(),
            'v_min': v.min(),
        }

    def high_critic_loss(self, batch, grad_params):
        """Compute the high-level SARSA critic loss."""
        next_v = self.network.select('high_value')(batch['high_value_next_observations'], batch['high_value_goals'])
        if self.config['value_loss_type'] == 'bce':
            next_v = jax.nn.sigmoid(next_v)
        q = (
            batch['high_value_rewards']
            + (self.config['discount'] ** batch['high_value_subgoal_steps']) * batch['high_value_masks'] * next_v
        )

        # Option-style subgoal encoding: (s, s', g) -> z
        z_actions = self._encode_subgoal(
            batch['observations'],
            batch['high_value_next_observations'],
            batch['high_value_goals'],
            grad_params,
        )
        q1, q2 = self.network.select('high_critic')(
            batch['observations'], batch['high_value_goals'], z_actions, params=grad_params
        )

        if self.config['value_loss_type'] == 'squared':
            critic_loss = ((q1 - q) ** 2 + (q2 - q) ** 2).mean()
        elif self.config['value_loss_type'] == 'bce':
            q1_logit, q2_logit = q1, q2
            critic_loss = self.bce_loss(q1_logit, q).mean() + self.bce_loss(q2_logit, q).mean()

        return critic_loss, {
            'critic_loss': critic_loss,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
        }

    def high_actor_loss(self, batch, grad_params, rng=None):
        """Compute the high-level flow BC actor loss."""
        # Option-style subgoal encoding: (s, s', g) -> z
        # Don't pass grad_params - subgoal encoder shouldn't get gradients from actor loss
        z_target = self._encode_subgoal(
            batch['observations'],
            batch['high_actor_next_observations'],
            batch['high_actor_goals'],
        )
        batch_size, action_dim = z_target.shape
        x_rng, t_rng = jax.random.split(rng, 2)

        x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
        x_1 = z_target
        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * x_1
        y = x_1 - x_0

        pred = self.network.select('high_actor_flow')(
            batch['observations'], batch['high_actor_goals'], x_t, t, params=grad_params
        )

        actor_loss = jnp.mean((pred - y) ** 2)

        actor_info = {
            'actor_loss': actor_loss,
        }

        return actor_loss, actor_info

    def low_actor_loss(self, batch, grad_params, rng):
        """Compute the low-level flow BC actor loss."""
        batch_size, action_dim = batch['actions'].shape
        x_rng, t_rng = jax.random.split(rng, 2)

        x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
        x_1 = batch['actions']
        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * x_1
        y = x_1 - x_0

        # Option-style subgoal encoding: (s, s', g) -> z
        # Use high_actor_goals (final goal) to match inference-time z distribution
        # NOTE: There's still a mismatch in s' (second arg):
        #   - high_actor uses s_{t+k} where k = min(25, dist_to_goal, dist_to_traj_end)
        #   - low_actor uses s_{t+25} (clamped only to traj end)
        # So high actor z's cover variable-length transitions, low actor only sees 25-step z's.
        z_goals = self._encode_subgoal(
            batch['observations'],
            batch['low_actor_goal_observations'],
            batch['high_actor_goals'],  # Must match high actor's goal conditioning
            grad_params,
        )
        # Optionally condition low actor on the final goal as well
        if self.config['low_actor_goal_conditioned']:
            if self.config['encode_goal']:
                encoded_goal = self.network.select('goal_encoder')(
                    batch['high_actor_goals'], params=grad_params
                )
                z_goals = jnp.concatenate([z_goals, encoded_goal], axis=-1)
            else:
                z_goals = jnp.concatenate([z_goals, batch['high_actor_goals']], axis=-1)
        pred = self.network.select('low_actor_flow')(batch['observations'], z_goals, x_t, t, params=grad_params)

        actor_loss = jnp.mean((pred - y) ** 2)

        actor_info = {
            'actor_loss': actor_loss,
        }

        return actor_loss, actor_info

    def total_loss(self, batch, grad_params, rng=None, step=0):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        rng, high_value_rng, high_critic_rng, high_actor_rng, low_actor_rng = jax.random.split(rng, 5)

        high_value_loss, high_value_info = self.high_value_loss(batch, grad_params)
        for k, v in high_value_info.items():
            info[f'high_value/{k}'] = v

        high_critic_loss, high_critic_info = self.high_critic_loss(batch, grad_params)
        for k, v in high_critic_info.items():
            info[f'high_critic/{k}'] = v

        low_actor_loss, low_actor_info = self.low_actor_loss(batch, grad_params, low_actor_rng)
        for k, v in low_actor_info.items():
            info[f'low_actor/{k}'] = v

        loss = high_value_loss + high_critic_loss + low_actor_loss

        # Only train high actor after warmup (let subgoal encoder stabilize first)
        def add_high_actor_loss():
            ha_loss, ha_info = self.high_actor_loss(batch, grad_params, high_actor_rng)
            return loss + ha_loss, ha_info

        def skip_high_actor_loss():
            return loss, {'actor_loss': 0.0}

        loss, high_actor_info = jax.lax.cond(
            step >= self.config['high_actor_warmup_steps'],
            add_high_actor_loss,
            skip_high_actor_loss,
        )
        for k, v in high_actor_info.items():
            info[f'high_actor/{k}'] = v

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
    def update(self, batch, step=0):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng, step=step)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, 'high_critic')

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(
        self,
        observations,
        goals=None,
        seed=None,
        temperature=1.0,
    ):
        """Sample actions from the actor."""
        high_seed, low_seed = jax.random.split(seed)

        # High-level: rejection sampling.
        orig_observations = observations
        n_subgoals = jax.random.normal(
            high_seed,
            (
                self.config['num_samples'],
                *observations.shape[:-1],
                self.config['z_dim'],
            ),
        )
        n_observations = jnp.repeat(jnp.expand_dims(observations, 0), self.config['num_samples'], axis=0)
        n_goals = jnp.repeat(jnp.expand_dims(goals, 0), self.config['num_samples'], axis=0)
        n_orig_observations = jnp.repeat(jnp.expand_dims(orig_observations, 0), self.config['num_samples'], axis=0)

        for i in range(self.config['flow_steps']):
            t = jnp.full((self.config['num_samples'], *observations.shape[:-1], 1), i / self.config['flow_steps'])
            vels = self.network.select('high_actor_flow')(n_observations, n_goals, n_subgoals, t)
            n_subgoals = n_subgoals + vels / self.config['flow_steps']

        q = self.network.select('high_critic')(n_orig_observations, goals=n_goals, actions=n_subgoals).min(axis=0)
        subgoals = n_subgoals[jnp.argmax(q)]

        # Low-level: behavioral cloning.
        # Optionally condition on the final goal as well
        if self.config['low_actor_goal_conditioned']:
            if self.config['encode_goal']:
                encoded_goal = self.network.select('goal_encoder')(goals)
                low_actor_goals = jnp.concatenate([subgoals, encoded_goal], axis=-1)
            else:
                low_actor_goals = jnp.concatenate([subgoals, goals], axis=-1)
        else:
            low_actor_goals = subgoals
        actions = jax.random.normal(low_seed, (*observations.shape[:-1], self.config['action_dim']))
        for i in range(self.config['flow_steps']):
            t = jnp.full((*observations.shape[:-1], 1), i / self.config['flow_steps'])
            vels = self.network.select('low_actor_flow')(observations, low_actor_goals, actions, t)
            actions = actions + vels / self.config['flow_steps']
        actions = jnp.clip(actions, -1, 1)

        return actions

    @classmethod
    def create(
        cls,
        seed,
        example_batch,
        config,
    ):
        """Create a new agent.

        Args:
            seed: Random seed.
            example_batch: Example batch from the dataset.
            config: Configuration dictionary.
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)

        ex_observations = example_batch['observations']
        ex_actions = example_batch['actions']
        ex_goals = example_batch['high_actor_goals']
        ex_times = ex_actions[..., :1]
        ex_z = np.zeros((ex_observations.shape[0], config['z_dim']), dtype=np.float32)
        action_dim = ex_actions.shape[-1]
        goal_dim = ex_goals.shape[-1]
        obs_dim = ex_observations.shape[-1]

        # Define networks.
        if config['use_resmlp']:
            # Use ResMLP architecture for all networks
            high_value_def = ResMLPValue(
                hidden_dim=config['resmlp_hidden_dim'],
                num_blocks=config['resmlp_num_blocks'],
                num_ensembles=1,
            )
            high_critic_def = ResMLPValue(
                hidden_dim=config['resmlp_hidden_dim'],
                num_blocks=config['resmlp_num_blocks'],
                num_ensembles=2,
            )
            high_actor_flow_def = ResMLPActorVectorField(
                hidden_dim=config['resmlp_hidden_dim'],
                num_blocks=config['resmlp_num_blocks'],
                action_dim=config['z_dim'],
            )
            low_actor_flow_def = ResMLPActorVectorField(
                hidden_dim=config['resmlp_hidden_dim'],
                num_blocks=config['resmlp_num_blocks'],
                action_dim=action_dim,
            )
        else:
            # Use standard MLP architecture
            high_value_def = GCValue(
                hidden_dims=config['value_hidden_dims'],
                layer_norm=config['layer_norm'],
                num_ensembles=1,
            )
            high_critic_def = GCValue(
                hidden_dims=config['value_hidden_dims'],
                layer_norm=config['layer_norm'],
                num_ensembles=2,
            )
            high_actor_flow_def = ActorVectorField(
                hidden_dims=config['actor_hidden_dims'],
                action_dim=config['z_dim'],
                layer_norm=config['layer_norm'],
            )
            low_actor_flow_def = ActorVectorField(
                hidden_dims=config['actor_hidden_dims'],
                action_dim=action_dim,
                layer_norm=config['layer_norm'],
            )

        # Optionally use a state encoder (e.g., for pixel observations or deeper state encoding)
        if config['encoder'] != 'none':
            rng, enc_shape_rng = jax.random.split(rng)
            encoder_module = encoder_modules[config['encoder']]
            encoder_def = encoder_module()
            enc_shape_params = encoder_def.init(enc_shape_rng, ex_observations)['params']
            ex_enc = encoder_def.apply({'params': enc_shape_params}, ex_observations)
            enc_dim = ex_enc.shape[-1]
        else:
            encoder_def = None
            ex_enc = ex_observations
            enc_dim = obs_dim

        # Option-style subgoal encoder: (enc(s), enc(s'), g) -> z
        # Uses ResMLP architecture with projection to z_dim
        subgoal_encoder_def = SubgoalEncoder(
            hidden_dim=config['subgoal_encoder_hidden_dim'],
            num_blocks=config['subgoal_encoder_num_blocks'],
            z_dim=config['z_dim'],
        )
        ex_subgoal_input = np.concatenate([ex_enc, ex_enc, ex_goals], axis=-1)

        # Optionally create goal encoder for low actor goal conditioning
        goal_encoder_def = None
        if config['low_actor_goal_conditioned'] and config['encode_goal']:
            goal_encoder_def = GoalEncoder(
                hidden_dim=config['goal_encoder_hidden_dim'],
                num_blocks=config['goal_encoder_num_blocks'],
                output_dim=config['goal_encoder_output_dim'],
            )
            ex_encoded_goal = np.zeros((ex_observations.shape[0], config['goal_encoder_output_dim']), dtype=np.float32)
            ex_low_actor_goal = np.concatenate([ex_z, ex_encoded_goal], axis=-1)
        elif config['low_actor_goal_conditioned']:
            ex_low_actor_goal = np.concatenate([ex_z, ex_goals], axis=-1)
        else:
            ex_low_actor_goal = ex_z

        network_info = dict(
            subgoal_encoder=(subgoal_encoder_def, (ex_subgoal_input,)),
            high_value=(high_value_def, (ex_observations, ex_goals)),
            high_critic=(high_critic_def, (ex_observations, ex_goals, ex_z)),
            target_high_critic=(copy.deepcopy(high_critic_def), (ex_observations, ex_goals, ex_z)),
            high_actor_flow=(high_actor_flow_def, (ex_observations, ex_goals, ex_z, ex_times)),
            low_actor_flow=(low_actor_flow_def, (ex_observations, ex_low_actor_goal, ex_actions, ex_times)),
        )
        if encoder_def is not None:
            network_info['encoder'] = (encoder_def, (ex_observations,))
        if goal_encoder_def is not None:
            network_info['goal_encoder'] = (goal_encoder_def, (ex_goals,))
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params['modules_target_high_critic'] = params['modules_high_critic']

        config['action_dim'] = action_dim
        config['goal_dim'] = goal_dim
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            # Agent hyperparameters.
            agent_name='latent_sharsa',  # Agent name.
            lr=3e-4,  # Learning rate.
            batch_size=1024,  # Batch size.
            actor_hidden_dims=(1024, 1024, 1024, 1024),  # Actor network hidden dimensions.
            value_hidden_dims=(1024, 1024, 1024, 1024),  # Value network hidden dimensions.
            layer_norm=True,  # Whether to use layer normalization.
            discount=0.999,  # Discount factor.
            tau=0.005,  # Target network update rate.
            q_agg='min',  # Aggregation function for Q values.
            action_dim=ml_collections.config_dict.placeholder(int),  # Action dimension (set automatically).
            goal_dim=ml_collections.config_dict.placeholder(int),  # Goal dimension (set automatically).
            z_dim=8,  # Latent subgoal dimension.
            subgoal_encoder_hidden_dim=256,  # Subgoal encoder hidden dimension (ResMLP).
            subgoal_encoder_num_blocks=2,  # Number of ResMLP blocks in subgoal encoder.
            value_loss_type='bce',  # Value loss type ('squared' or 'bce').
            flow_steps=10,  # Number of flow steps.
            num_samples=32,  # Number of samples for the actor.
            discrete=False,  # Whether the action space is discrete.
            encoder='none',  # Encoder name ('none' for no encoder, or e.g., 'resmlp_small').
            freeze_encoder=True,  # Whether to freeze encoder weights (if encoder is used).
            high_actor_warmup_steps=0,  # Steps to freeze high actor (0 = no warmup).
            low_actor_goal_conditioned=False,  # Whether low actor sees the final goal in addition to z.
            encode_goal=False,  # Whether to encode the goal through a ResMLP before passing to low actor.
            goal_encoder_hidden_dim=256,  # Goal encoder hidden dimension (ResMLP).
            goal_encoder_num_blocks=2,  # Number of ResMLP blocks in goal encoder.
            goal_encoder_output_dim=64,  # Output dimension of goal encoder.
            use_resmlp=False,  # Whether to use ResMLP architecture for actors/critics.
            resmlp_hidden_dim=256,  # Hidden dimension for ResMLP networks (favor depth over width).
            resmlp_num_blocks=4,  # Number of residual blocks (4 layers each = 17 total layers).
            # Dataset hyperparameters.
            dataset_class='HGCDataset',  # Dataset class name.
            subgoal_steps=25,  # Subgoal steps.
            value_p_curgoal=0.2,  # Probability of using the current state as the value goal.
            value_p_trajgoal=0.5,  # Probability of using a future state in the same trajectory as the value goal.
            value_p_randomgoal=0.3,  # Probability of using a random state as the value goal.
            value_geom_sample=False,  # Whether to use geometric sampling for future value goals.
            actor_p_curgoal=0.0,  # Probability of using the current state as the actor goal.
            actor_p_trajgoal=0.5,  # Probability of using a future state in the same trajectory as the actor goal.
            actor_p_randomgoal=0.5,  # Probability of using a random state as the actor goal.
            actor_geom_sample=True,  # Whether to use geometric sampling for future actor goals.
            gc_negative=False,  # Whether to use '0 if s == g else -1' (True) or '1 if s == g else 0' (False) as reward.
            p_aug=0.0,  # Probability of applying image augmentation.
            frame_stack=ml_collections.config_dict.placeholder(int),  # Number of frames to stack.
        )
    )
    return config
