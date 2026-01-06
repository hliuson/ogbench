"""World model module for representation learning.

This module provides a deterministic world model that learns latent representations
by predicting next-state latents from current state and action. It supports both
state-based and pixel-based observations via the existing encoder infrastructure.

Usage:
    # Create world model
    world_model = WorldModel.create(seed, ex_observations, ex_actions, config)

    # Training step
    world_model, info = world_model.update(batch)

    # Get latent representations
    latents = world_model.network.select('encoder')(observations)

    # Get dynamics predictions (in latent space)
    next_latents = world_model.network.select('dynamics')(
        jnp.concatenate([latents, actions], axis=-1)
    )
"""

from typing import Any, Sequence

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import MLP, default_init


class StateEncoder(nn.Module):
    """Encoder for state-based observations.

    Maps observations to a latent space using an MLP.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        latent_dim: Output latent dimension.
        layer_norm: Whether to apply layer normalization.
    """

    hidden_dims: Sequence[int]
    latent_dim: int
    layer_norm: bool = True

    @nn.compact
    def __call__(self, observations):
        x = MLP(
            hidden_dims=(*self.hidden_dims, self.latent_dim),
            activate_final=False,
            layer_norm=self.layer_norm,
        )(observations)
        return x


class LatentDynamics(nn.Module):
    """Dynamics model that predicts next latent from current latent and action.

    Takes concatenated [z_s, a] as input and outputs predicted z_s'.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        latent_dim: Output latent dimension (should match encoder output).
        layer_norm: Whether to apply layer normalization.
    """

    hidden_dims: Sequence[int]
    latent_dim: int
    layer_norm: bool = True

    @nn.compact
    def __call__(self, latent_action):
        """Forward pass.

        Args:
            latent_action: Concatenated [z_s, a] tensor.

        Returns:
            Predicted next latent z_s'.
        """
        x = MLP(
            hidden_dims=(*self.hidden_dims, self.latent_dim),
            activate_final=False,
            layer_norm=self.layer_norm,
        )(latent_action)
        return x


class RewardPredictor(nn.Module):
    """Reward predictor that predicts reward given latent state and goal.

    Takes concatenated [z_s, z_g] as input and predicts the reward.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        layer_norm: Whether to apply layer normalization.
    """

    hidden_dims: Sequence[int] = (256, 256)
    layer_norm: bool = True

    @nn.compact
    def __call__(self, latent_goal):
        """Forward pass.

        Args:
            latent_goal: Concatenated [z_s, z_g] tensor.

        Returns:
            Predicted reward scalar.
        """
        x = MLP(
            hidden_dims=self.hidden_dims,
            activate_final=True,
            layer_norm=self.layer_norm,
        )(latent_goal)
        return nn.Dense(1, kernel_init=default_init())(x).squeeze(-1)


class WorldModel(flax.struct.PyTreeNode):
    """World model for representation learning.

    This model learns latent representations by predicting next-state latents
    from current state and action. The loss is a simple distance metric (MSE)
    between predicted and actual next latent, with stop-gradient on the target.

    The model also includes a reward predictor that predicts rewards given
    latent state and goal, trained jointly with the dynamics model.

    Attributes:
        rng: Random number generator state.
        network: TrainState containing all network modules.
        config: Configuration dictionary.
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def dynamics_loss(self, batch, grad_params):
        """Compute the dynamics prediction loss.

        Predicts next latent from current latent and action, compares to
        actual next latent with stop-gradient.

        Args:
            batch: Batch dictionary with 'observations', 'actions', 'next_observations'.
            grad_params: Parameters to compute gradients for.

        Returns:
            Tuple of (loss, info_dict).
        """
        # Encode current and next observations
        z_s = self.network.select('encoder')(batch['observations'], params=grad_params)
        z_s_next_target = self.network.select('encoder')(batch['next_observations'])  # stop-grad (no params)

        # Predict next latent
        z_s_next_pred = self.network.select('dynamics')(
            jnp.concatenate([z_s, batch['actions']], axis=-1),
            params=grad_params,
        )

        # MSE loss with stop-gradient on target
        dynamics_loss = jnp.mean((z_s_next_pred - jax.lax.stop_gradient(z_s_next_target)) ** 2)

        return dynamics_loss, {
            'dynamics_loss': dynamics_loss,
            'latent_norm': jnp.mean(jnp.linalg.norm(z_s, axis=-1)),
            'latent_std': jnp.std(z_s),
        }

    def reward_loss(self, batch, grad_params):
        """Compute the reward prediction loss.

        Predicts rewards from latent state and goal, trained jointly
        with the encoder.

        Args:
            batch: Batch dictionary with 'observations', 'value_goals', and 'rewards'.
            grad_params: Parameters to compute gradients for.

        Returns:
            Tuple of (loss, info_dict).
        """
        # Encode observations and goals
        z_s = self.network.select('encoder')(batch['observations'], params=grad_params)
        z_g = self.network.select('encoder')(batch['value_goals'], params=grad_params)

        # Predict rewards from [z_s, z_g]
        pred_rewards = self.network.select('reward_predictor')(
            jnp.concatenate([z_s, z_g], axis=-1),
            params=grad_params,
        )

        # MSE loss for reward prediction
        reward_loss = jnp.mean((pred_rewards - batch['rewards']) ** 2)

        return reward_loss, {
            'reward_loss': reward_loss,
            'reward_pred_mean': jnp.mean(pred_rewards),
            'reward_pred_std': jnp.std(pred_rewards),
            'reward_target_mean': jnp.mean(batch['rewards']),
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss.

        Args:
            batch: Batch dictionary.
            grad_params: Parameters to compute gradients for.
            rng: Optional random number generator.

        Returns:
            Tuple of (total_loss, info_dict).
        """
        info = {}

        # Dynamics loss (main representation learning objective)
        dynamics_loss, dynamics_info = self.dynamics_loss(batch, grad_params)
        for k, v in dynamics_info.items():
            info[f'dynamics/{k}'] = v

        # Reward prediction loss
        if self.config['use_reward_loss']:
            reward_loss, reward_info = self.reward_loss(batch, grad_params)
            for k, v in reward_info.items():
                info[f'reward/{k}'] = v
        else:
            reward_loss = 0.0

        loss = dynamics_loss + self.config['reward_loss_coef'] * reward_loss
        return loss, info

    @jax.jit
    def update(self, batch):
        """Update the world model and return a new model with information dictionary.

        Args:
            batch: Batch dictionary with training data.

        Returns:
            Tuple of (new_world_model, info_dict).
        """
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def encode(self, observations):
        """Encode observations to latent representations.

        Convenience method for getting latent representations.

        Args:
            observations: Observation tensor.

        Returns:
            Latent representation tensor.
        """
        return self.network.select('encoder')(observations)

    @jax.jit
    def predict_next_latent(self, observations, actions):
        """Predict next latent from observations and actions.

        Args:
            observations: Current observation tensor.
            actions: Action tensor.

        Returns:
            Predicted next latent tensor.
        """
        z_s = self.network.select('encoder')(observations)
        z_s_next = self.network.select('dynamics')(
            jnp.concatenate([z_s, actions], axis=-1)
        )
        return z_s_next

    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,
        config,
    ):
        """Create a new world model.

        Args:
            seed: Random seed.
            ex_observations: Example batch of observations for initialization.
            ex_actions: Example batch of actions for initialization.
            config: Configuration dictionary.

        Returns:
            Initialized WorldModel instance.
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        action_dim = ex_actions.shape[-1]
        latent_dim = config['latent_dim']

        # Define encoder based on observation type
        if config['encoder'] is not None:
            # Pixel observations: use visual encoder
            encoder_module = encoder_modules[config['encoder']]
            # The ImpalaEncoder outputs to mlp_hidden_dims[-1], we add a projection
            encoder_def = nn.Sequential([
                encoder_module(
                    mlp_hidden_dims=config['encoder_hidden_dims'],
                    layer_norm=config['layer_norm'],
                ),
                nn.Dense(latent_dim, kernel_init=default_init()),
            ])
        else:
            # State observations: use MLP encoder
            encoder_def = StateEncoder(
                hidden_dims=config['encoder_hidden_dims'],
                latent_dim=latent_dim,
                layer_norm=config['layer_norm'],
            )

        # Define dynamics model
        dynamics_def = LatentDynamics(
            hidden_dims=config['dynamics_hidden_dims'],
            latent_dim=latent_dim,
            layer_norm=config['layer_norm'],
        )

        # Define reward predictor
        reward_predictor_def = RewardPredictor(
            hidden_dims=config['reward_hidden_dims'],
            layer_norm=config['layer_norm'],
        )

        # Example inputs for initialization
        ex_latent_action = jnp.zeros((ex_observations.shape[0], latent_dim + action_dim))
        ex_latent_goal = jnp.zeros((ex_observations.shape[0], latent_dim * 2))  # [z_s, z_g]

        # Build network dict
        network_info = dict(
            encoder=(encoder_def, (ex_observations,)),
            dynamics=(dynamics_def, (ex_latent_action,)),
            reward_predictor=(reward_predictor_def, (ex_latent_goal,)),
        )

        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    """Get default configuration for the world model.

    Returns:
        ml_collections.ConfigDict with default hyperparameters.
    """
    config = ml_collections.ConfigDict(
        dict(
            # Model hyperparameters.
            lr=3e-4,  # Learning rate.
            batch_size=1024,  # Batch size.
            latent_dim=256,  # Latent representation dimension.
            encoder_hidden_dims=(256, 256),  # Encoder MLP hidden dimensions (for state obs).
            dynamics_hidden_dims=(256, 256),  # Dynamics MLP hidden dimensions.
            reward_hidden_dims=(256, 256),  # Reward predictor hidden dimensions.
            layer_norm=True,  # Whether to use layer normalization.
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None for state, 'impala_small' for pixels).
            # Reward prediction hyperparameters.
            use_reward_loss=True,  # Whether to train the reward predictor.
            reward_loss_coef=1.0,  # Coefficient for reward prediction loss.
            # Dataset hyperparameters (for compatibility with main.py).
            dataset_class='GCDataset',  # Dataset class name.
            value_p_curgoal=0.0,  # Probability of using the current state as the value goal.
            value_p_trajgoal=1.0,  # Probability of using a future state in the same trajectory as the value goal.
            value_p_randomgoal=0.0,  # Probability of using a random state as the value goal.
            value_geom_sample=False,  # Whether to use geometric sampling for future value goals.
            actor_p_curgoal=0.0,  # Probability of using the current state as the actor goal.
            actor_p_trajgoal=1.0,  # Probability of using a future state in the same trajectory as the actor goal.
            actor_p_randomgoal=0.0,  # Probability of using a random state as the actor goal.
            actor_geom_sample=False,  # Whether to use geometric sampling for future actor goals.
            gc_negative=True,  # Whether to use negative rewards for non-goal states.
            discount=0.99,  # Discount factor (for geometric goal sampling).
            p_aug=0.0,  # Probability of applying image augmentation.
            frame_stack=ml_collections.config_dict.placeholder(int),  # Number of frames to stack.
        )
    )
    return config
