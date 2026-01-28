import copy
from typing import Any, Sequence

import distrax
import flax
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax

import flax.linen as nn

from utils.encoders import encoder_modules, ResMLPEncoder
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import (
    ActorVectorField,
    GCValue,
    MLP,
    ResMLPActorVectorField,
    ResMLPValue,
    default_init,
)


def gumbel_softmax(logits, rng, temperature=1.0, hard=False):
    """Sample from the Gumbel-Softmax distribution with optional straight-through."""
    temperature = jnp.maximum(temperature, 1e-6)
    uniform = jax.random.uniform(rng, logits.shape, minval=1e-6, maxval=1.0, dtype=logits.dtype)
    gumbels = -jnp.log(-jnp.log(uniform))
    y_soft = jax.nn.softmax((logits + gumbels) / temperature, axis=-1)
    if not hard:
        return y_soft
    y_hard = jax.nn.one_hot(jnp.argmax(y_soft, axis=-1), y_soft.shape[-1], dtype=y_soft.dtype)
    return y_soft + jax.lax.stop_gradient(y_hard - y_soft)


def factorized_gumbel_softmax(logits, rng, n_factors, n_codes, temperature=1.0, hard=False):
    """Sample from factorized Gumbel-Softmax: n_factors independent categoricals with n_codes each.

    Args:
        logits: shape (..., n_factors * n_codes)
        rng: random key
        n_factors: number of independent categorical factors
        n_codes: number of codes per factor
        temperature: Gumbel-Softmax temperature
        hard: whether to use straight-through estimator

    Returns:
        shape (..., n_factors * n_codes) - concatenated one-hot vectors for each factor
    """
    # Reshape to (..., n_factors, n_codes)
    batch_shape = logits.shape[:-1]
    logits = logits.reshape(*batch_shape, n_factors, n_codes)

    temperature = jnp.maximum(temperature, 1e-6)
    uniform = jax.random.uniform(rng, logits.shape, minval=1e-6, maxval=1.0, dtype=logits.dtype)
    gumbels = -jnp.log(-jnp.log(uniform))
    y_soft = jax.nn.softmax((logits + gumbels) / temperature, axis=-1)

    if hard:
        y_hard = jax.nn.one_hot(jnp.argmax(y_soft, axis=-1), n_codes, dtype=y_soft.dtype)
        y = y_soft + jax.lax.stop_gradient(y_hard - y_soft)
    else:
        y = y_soft

    # Flatten back to (..., n_factors * n_codes)
    return y.reshape(*batch_shape, n_factors * n_codes)


class ResMLPDiscreteActor(nn.Module):
    """Goal-conditioned categorical policy using ResMLP encoder."""

    hidden_dim: int
    num_blocks: int
    action_dim: int
    final_fc_init_scale: float = 1e-2

    @nn.compact
    def __call__(self, observations, goals=None, temperature=1.0):
        inputs = [observations]
        if goals is not None:
            inputs.append(goals)
        inputs = jnp.concatenate(inputs, axis=-1)
        x = ResMLPEncoder(hidden_dim=self.hidden_dim, num_blocks=self.num_blocks)(inputs)
        logits = nn.Dense(self.action_dim, kernel_init=default_init(self.final_fc_init_scale))(x)
        return distrax.Categorical(logits=logits / jnp.maximum(1e-6, temperature))


class MLPDiscreteActor(nn.Module):
    """Goal-conditioned categorical policy using an MLP."""

    hidden_dims: Sequence[int]
    action_dim: int
    layer_norm: bool = False
    final_fc_init_scale: float = 1e-2

    def setup(self):
        self.actor_net = MLP(self.hidden_dims, activate_final=True, layer_norm=self.layer_norm)
        self.logit_net = nn.Dense(self.action_dim, kernel_init=default_init(self.final_fc_init_scale))

    def __call__(self, observations, goals=None, temperature=1.0):
        inputs = [observations]
        if goals is not None:
            inputs.append(goals)
        inputs = jnp.concatenate(inputs, axis=-1)
        outputs = self.actor_net(inputs)
        logits = self.logit_net(outputs)
        return distrax.Categorical(logits=logits / jnp.maximum(1e-6, temperature))


class MLPFactorizedDiscreteActor(nn.Module):
    """Goal-conditioned factorized categorical policy using an MLP.

    Outputs n_factors independent categorical distributions over n_codes each.
    """

    hidden_dims: Sequence[int]
    n_factors: int
    n_codes: int
    layer_norm: bool = False
    final_fc_init_scale: float = 1e-2

    def setup(self):
        self.actor_net = MLP(self.hidden_dims, activate_final=True, layer_norm=self.layer_norm)
        self.logit_net = nn.Dense(
            self.n_factors * self.n_codes, kernel_init=default_init(self.final_fc_init_scale)
        )

    def __call__(self, observations, goals=None, temperature=1.0):
        inputs = [observations]
        if goals is not None:
            inputs.append(goals)
        inputs = jnp.concatenate(inputs, axis=-1)
        outputs = self.actor_net(inputs)
        logits = self.logit_net(outputs)
        # Reshape to (batch, n_factors, n_codes) for independent categoricals
        batch_shape = logits.shape[:-1]
        logits = logits.reshape(*batch_shape, self.n_factors, self.n_codes)
        return distrax.Categorical(logits=logits / jnp.maximum(1e-6, temperature))


class ResMLPFactorizedDiscreteActor(nn.Module):
    """Goal-conditioned factorized categorical policy using ResMLP encoder."""

    hidden_dim: int
    num_blocks: int
    n_factors: int
    n_codes: int
    final_fc_init_scale: float = 1e-2

    @nn.compact
    def __call__(self, observations, goals=None, temperature=1.0):
        inputs = [observations]
        if goals is not None:
            inputs.append(goals)
        inputs = jnp.concatenate(inputs, axis=-1)
        x = ResMLPEncoder(hidden_dim=self.hidden_dim, num_blocks=self.num_blocks)(inputs)
        logits = nn.Dense(
            self.n_factors * self.n_codes, kernel_init=default_init(self.final_fc_init_scale)
        )(x)
        # Reshape to (batch, n_factors, n_codes) for independent categoricals
        batch_shape = logits.shape[:-1]
        logits = logits.reshape(*batch_shape, self.n_factors, self.n_codes)
        return distrax.Categorical(logits=logits / jnp.maximum(1e-6, temperature))


class SubgoalEncoder(nn.Module):
    """Option-style subgoal encoder: (s, s', g) -> z with Gumbel-Softmax output."""

    use_resmlp: bool
    mlp_hidden_dims: Sequence[int]
    layer_norm: bool
    resmlp_hidden_dim: int
    resmlp_num_blocks: int
    z_dim: int = 8
    gumbel_temperature: float = 1.0
    gumbel_hard: bool = True

    @nn.compact
    def __call__(self, x):
        if self.use_resmlp:
            x = ResMLPEncoder(hidden_dim=self.resmlp_hidden_dim, num_blocks=self.resmlp_num_blocks)(x)
            logits = nn.Dense(self.z_dim)(x)
        else:
            logits = MLP((*self.mlp_hidden_dims, self.z_dim), activate_final=False, layer_norm=self.layer_norm)(x)
        rng = self.make_rng('gumbel')
        return gumbel_softmax(logits, rng, temperature=self.gumbel_temperature, hard=self.gumbel_hard)


class FactorizedSubgoalEncoder(nn.Module):
    """Option-style subgoal encoder with factorized discrete output.

    Outputs n_factors independent categorical distributions, each with n_codes options.
    Total representation size is n_factors * n_codes (concatenated one-hots).
    Number of unique codes is n_codes^n_factors.
    """

    use_resmlp: bool
    mlp_hidden_dims: Sequence[int]
    layer_norm: bool
    resmlp_hidden_dim: int
    resmlp_num_blocks: int
    n_factors: int = 8
    n_codes: int = 8
    gumbel_temperature: float = 1.0
    gumbel_hard: bool = True

    @nn.compact
    def __call__(self, x):
        output_dim = self.n_factors * self.n_codes
        if self.use_resmlp:
            x = ResMLPEncoder(hidden_dim=self.resmlp_hidden_dim, num_blocks=self.resmlp_num_blocks)(x)
            logits = nn.Dense(output_dim)(x)
        else:
            logits = MLP((*self.mlp_hidden_dims, output_dim), activate_final=False, layer_norm=self.layer_norm)(x)
        rng = self.make_rng('gumbel')
        return factorized_gumbel_softmax(
            logits, rng, self.n_factors, self.n_codes,
            temperature=self.gumbel_temperature, hard=self.gumbel_hard
        )


class FactorizedCodebook(nn.Module):
    """Learned codebook for factorized discrete representations.

    Maps factorized one-hot vectors to learned embeddings.
    Input: (..., n_factors * n_codes) - concatenated one-hots
    Output: (..., embed_dim) - summed embeddings across factors
    """

    n_factors: int
    n_codes: int
    embed_dim: int
    agg: str = 'sum'  # 'sum' or 'concat'

    @nn.compact
    def __call__(self, z):
        """Convert one-hot codes to embeddings.

        Args:
            z: shape (..., n_factors * n_codes) - concatenated one-hots for each factor

        Returns:
            shape (..., embed_dim) if agg='sum', or (..., n_factors * embed_dim) if agg='concat'
        """
        batch_shape = z.shape[:-1]
        # Reshape to (..., n_factors, n_codes)
        z = z.reshape(*batch_shape, self.n_factors, self.n_codes)

        # Learned embedding table: (n_factors, n_codes, embed_dim)
        # Each factor has its own codebook
        codebook = self.param(
            'codebook',
            nn.initializers.normal(stddev=0.02),
            (self.n_factors, self.n_codes, self.embed_dim)
        )

        # z @ codebook: (..., n_factors, n_codes) @ (n_factors, n_codes, embed_dim)
        # -> (..., n_factors, embed_dim)
        embeddings = jnp.einsum('...fc,fcd->...fd', z, codebook)

        if self.agg == 'sum':
            # Sum across factors: (..., embed_dim)
            return embeddings.sum(axis=-2)
        else:
            # Concatenate across factors: (..., n_factors * embed_dim)
            return embeddings.reshape(*batch_shape, self.n_factors * self.embed_dim)


class GoalEncoder(nn.Module):
    """Encodes raw goal into a compact representation using shared MLP/ResMLP size."""

    use_resmlp: bool
    mlp_hidden_dims: Sequence[int]
    layer_norm: bool
    resmlp_hidden_dim: int
    resmlp_num_blocks: int
    output_dim: int = 64

    @nn.compact
    def __call__(self, x):
        if self.use_resmlp:
            x = ResMLPEncoder(hidden_dim=self.resmlp_hidden_dim, num_blocks=self.resmlp_num_blocks)(x)
            x = nn.Dense(self.output_dim)(x)
        else:
            x = MLP((*self.mlp_hidden_dims, self.output_dim), activate_final=False, layer_norm=self.layer_norm)(x)
        return x


class DiscreteLatentSHARSAAgent(flax.struct.PyTreeNode):
    """SHARSA agent with discrete latent subgoals."""

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

    def _encode_subgoal(self, observations, next_observations, goals, grad_params=None, rng=None):
        """Encode (current state, next state, goal) into latent subgoal z (option-style).

        By default, only uses s' (next state). Set use_subgoal_currstate=True and/or
        use_subgoal_truegoal=True to include current state s and/or goal g.
        """
        enc_next = self._encode_observations(next_observations, grad_params)

        # Build input based on which components are enabled
        inputs = [enc_next]  # s' is always included
        if self.config['use_subgoal_currstate']:
            enc = self._encode_observations(observations, grad_params)
            inputs.insert(0, enc)  # prepend s to maintain [s, s', g] order
        if self.config['use_subgoal_truegoal']:
            inputs.append(goals)  # append g

        enc_input = jnp.concatenate(inputs, axis=-1)
        rng = rng if rng is not None else self.rng
        return self.network.select('subgoal_encoder')(
            enc_input, params=grad_params, rngs={'gumbel': rng}
        )

    def high_value_loss(self, batch, grad_params, rng):
        """Compute the high-level SARSA value loss."""
        # Option-style subgoal encoding: (s, s', g) -> z
        z_target = self._encode_subgoal(
            batch['observations'],
            batch['high_value_next_observations'],
            batch['high_value_goals'],
            grad_params,
            rng,
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

    def high_critic_loss(self, batch, grad_params, rng):
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
            rng,
        )
        q1, q2 = self.network.select('high_critic')(
            batch['observations'], batch['high_value_goals'], z_actions, params=grad_params
        )

        if self.config['value_loss_type'] == 'squared':
            critic_loss = ((q1 - q) ** 2 + (q2 - q) ** 2).mean()
        elif self.config['value_loss_type'] == 'bce':
            q1_logit, q2_logit = q1, q2
            critic_loss = self.bce_loss(q1_logit, q).mean() + self.bce_loss(q2_logit, q).mean()

        z_norms = jnp.linalg.norm(z_actions, axis=-1)
        return critic_loss, {
            'critic_loss': critic_loss,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
            'z_norm_mean': z_norms.mean(),
            'z_norm_std': z_norms.std(),
        }

    def high_actor_loss(self, batch, grad_params, rng=None):
        """Compute the high-level categorical actor loss."""
        # Option-style subgoal encoding: (s, s', g) -> z
        # Don't pass grad_params - subgoal encoder shouldn't get gradients from actor loss
        z_target = self._encode_subgoal(
            batch['observations'],
            batch['high_actor_next_observations'],
            batch['high_actor_goals'],
            rng=rng,
        )
        z_target = jax.lax.stop_gradient(z_target)

        dist = self.network.select('high_actor')(batch['observations'], batch['high_actor_goals'], params=grad_params)

        if self.config['factorized']:
            # Factorized: z_target is (batch, n_factors * n_codes), reshape to (batch, n_factors, n_codes)
            n_factors = self.config['n_factors']
            n_codes = self.config['n_codes']
            z_target = z_target.reshape(-1, n_factors, n_codes)
            target_idx = jnp.argmax(z_target, axis=-1)  # (batch, n_factors)
            # dist is Categorical with logits (batch, n_factors, n_codes)
            # log_prob returns (batch, n_factors), sum over factors
            actor_loss = -dist.log_prob(target_idx).sum(axis=-1).mean()
        else:
            target_idx = jnp.argmax(z_target, axis=-1)
            actor_loss = -dist.log_prob(target_idx).mean()

        actor_info = {
            'actor_loss': actor_loss,
        }

        return actor_loss, actor_info

    def low_actor_loss(self, batch, grad_params, rng):
        """Compute the low-level flow BC actor loss."""
        batch_size, action_dim = batch['actions'].shape
        flow_rng, subgoal_rng = jax.random.split(rng, 2)
        x_rng, t_rng = jax.random.split(flow_rng, 2)

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
            subgoal_rng,
        )
        # Optionally convert to learned embeddings via codebook
        if self.config['factorized'] and self.config['use_codebook']:
            z_goals = self.network.select('codebook')(z_goals, params=grad_params)
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

        high_value_loss, high_value_info = self.high_value_loss(batch, grad_params, high_value_rng)
        for k, v in high_value_info.items():
            info[f'high_value/{k}'] = v

        high_critic_loss, high_critic_info = self.high_critic_loss(batch, grad_params, high_critic_rng)
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

        # High-level: categorical policy over discrete subgoals.
        high_dist = self.network.select('high_actor')(observations, goals, temperature=temperature)
        subgoal_idx = high_dist.sample(seed=high_seed)

        if self.config['factorized']:
            # Factorized: subgoal_idx is (batch, n_factors), one-hot each factor and concatenate
            n_factors = self.config['n_factors']
            n_codes = self.config['n_codes']
            # subgoal_idx shape: (*batch_shape, n_factors)
            subgoals = jax.nn.one_hot(subgoal_idx, n_codes, dtype=observations.dtype)
            # subgoals shape: (*batch_shape, n_factors, n_codes)
            # Flatten to (*batch_shape, n_factors * n_codes)
            subgoals = subgoals.reshape(*observations.shape[:-1], n_factors * n_codes)
            # Optionally convert to learned embeddings via codebook
            if self.config['use_codebook']:
                subgoals = self.network.select('codebook')(subgoals)
        else:
            subgoals = jax.nn.one_hot(subgoal_idx, self.config['z_dim'], dtype=observations.dtype)

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
        # Compute z_dim based on factorized vs non-factorized
        if config['factorized']:
            z_dim = config['n_factors'] * config['n_codes']
            config['z_dim'] = z_dim  # Set for consistency
        else:
            z_dim = config['z_dim']

        ex_z = np.zeros((ex_observations.shape[0], z_dim), dtype=np.float32)
        action_dim = ex_actions.shape[-1]
        goal_dim = ex_goals.shape[-1]
        obs_dim = ex_observations.shape[-1]
        mlp_hidden_dims = (config['mlp_hidden_dim'],) * config['mlp_num_layers']

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
            if config['factorized']:
                high_actor_def = ResMLPFactorizedDiscreteActor(
                    hidden_dim=config['resmlp_hidden_dim'],
                    num_blocks=config['resmlp_num_blocks'],
                    n_factors=config['n_factors'],
                    n_codes=config['n_codes'],
                )
            else:
                high_actor_def = ResMLPDiscreteActor(
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
                hidden_dims=mlp_hidden_dims,
                layer_norm=config['layer_norm'],
                num_ensembles=1,
            )
            high_critic_def = GCValue(
                hidden_dims=mlp_hidden_dims,
                layer_norm=config['layer_norm'],
                num_ensembles=2,
            )
            if config['factorized']:
                high_actor_def = MLPFactorizedDiscreteActor(
                    hidden_dims=mlp_hidden_dims,
                    n_factors=config['n_factors'],
                    n_codes=config['n_codes'],
                    layer_norm=config['layer_norm'],
                )
            else:
                high_actor_def = MLPDiscreteActor(
                    hidden_dims=mlp_hidden_dims,
                    action_dim=config['z_dim'],
                    layer_norm=config['layer_norm'],
                )
            low_actor_flow_def = ActorVectorField(
                hidden_dims=mlp_hidden_dims,
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
        if config['factorized']:
            subgoal_encoder_def = FactorizedSubgoalEncoder(
                use_resmlp=config['use_resmlp'],
                mlp_hidden_dims=mlp_hidden_dims,
                layer_norm=config['layer_norm'],
                resmlp_hidden_dim=config['resmlp_hidden_dim'],
                resmlp_num_blocks=config['resmlp_num_blocks'],
                n_factors=config['n_factors'],
                n_codes=config['n_codes'],
                gumbel_temperature=config['gumbel_temperature'],
                gumbel_hard=config['gumbel_hard'],
            )
        else:
            subgoal_encoder_def = SubgoalEncoder(
                use_resmlp=config['use_resmlp'],
                mlp_hidden_dims=mlp_hidden_dims,
                layer_norm=config['layer_norm'],
                resmlp_hidden_dim=config['resmlp_hidden_dim'],
                resmlp_num_blocks=config['resmlp_num_blocks'],
                z_dim=config['z_dim'],
                gumbel_temperature=config['gumbel_temperature'],
                gumbel_hard=config['gumbel_hard'],
            )
        # Build example input based on which components are enabled
        subgoal_inputs = [ex_enc]  # s' always included
        if config['use_subgoal_currstate']:
            subgoal_inputs.insert(0, ex_enc)  # prepend s
        if config['use_subgoal_truegoal']:
            subgoal_inputs.append(ex_goals)  # append g
        ex_subgoal_input = np.concatenate(subgoal_inputs, axis=-1)

        # Optionally create codebook for factorized discrete representations
        codebook_def = None
        if config['factorized'] and config['use_codebook']:
            codebook_def = FactorizedCodebook(
                n_factors=config['n_factors'],
                n_codes=config['n_codes'],
                embed_dim=config['codebook_embed_dim'],
                agg=config['codebook_agg'],
            )
            # Compute z embedding dimension based on aggregation
            if config['codebook_agg'] == 'sum':
                z_embed_dim = config['codebook_embed_dim']
            else:  # concat
                z_embed_dim = config['n_factors'] * config['codebook_embed_dim']
            ex_z_embed = np.zeros((ex_observations.shape[0], z_embed_dim), dtype=np.float32)
        else:
            ex_z_embed = ex_z  # Use raw one-hots

        # Optionally create goal encoder for low actor goal conditioning
        goal_encoder_def = None
        if config['low_actor_goal_conditioned'] and config['encode_goal']:
            goal_encoder_def = GoalEncoder(
                use_resmlp=config['use_resmlp'],
                mlp_hidden_dims=mlp_hidden_dims,
                layer_norm=config['layer_norm'],
                resmlp_hidden_dim=config['resmlp_hidden_dim'],
                resmlp_num_blocks=config['resmlp_num_blocks'],
                output_dim=config['goal_encoder_output_dim'],
            )
            ex_encoded_goal = np.zeros((ex_observations.shape[0], config['goal_encoder_output_dim']), dtype=np.float32)
            ex_low_actor_goal = np.concatenate([ex_z_embed, ex_encoded_goal], axis=-1)
        elif config['low_actor_goal_conditioned']:
            ex_low_actor_goal = np.concatenate([ex_z_embed, ex_goals], axis=-1)
        else:
            ex_low_actor_goal = ex_z_embed

        network_info = dict(
            subgoal_encoder=(subgoal_encoder_def, (ex_subgoal_input,)),
            high_value=(high_value_def, (ex_observations, ex_goals)),
            high_critic=(high_critic_def, (ex_observations, ex_goals, ex_z)),
            target_high_critic=(copy.deepcopy(high_critic_def), (ex_observations, ex_goals, ex_z)),
            high_actor=(high_actor_def, (ex_observations, ex_goals)),
            low_actor_flow=(low_actor_flow_def, (ex_observations, ex_low_actor_goal, ex_actions, ex_times)),
        )
        if encoder_def is not None:
            network_info['encoder'] = (encoder_def, (ex_observations,))
        if goal_encoder_def is not None:
            network_info['goal_encoder'] = (goal_encoder_def, (ex_goals,))
        if codebook_def is not None:
            network_info['codebook'] = (codebook_def, (ex_z,))
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        rng, gumbel_rng = jax.random.split(rng)
        network_params = network_def.init({'params': init_rng, 'gumbel': gumbel_rng}, **network_args)['params']
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
            agent_name='discrete_latent_sharsa',  # Agent name.
            lr=3e-4,  # Learning rate.
            batch_size=1024,  # Batch size.
            mlp_hidden_dim=256,  # Hidden width for all MLP networks.
            mlp_num_layers=4,  # Number of layers for all MLP networks.
            layer_norm=True,  # Whether to use layer normalization.
            discount=0.999,  # Discount factor.
            tau=0.005,  # Target network update rate.
            q_agg='min',  # Aggregation function for Q values.
            action_dim=ml_collections.config_dict.placeholder(int),  # Action dimension (set automatically).
            goal_dim=ml_collections.config_dict.placeholder(int),  # Goal dimension (set automatically).
            z_dim=8,  # Number of discrete subgoals (non-factorized mode).
            factorized=False,  # Whether to use factorized discrete representation.
            n_factors=8,  # Number of independent categorical factors (factorized mode).
            n_codes=8,  # Number of codes per factor (factorized mode). Total codes = n_codes^n_factors.
            use_codebook=False,  # Whether to use learned codebook embeddings instead of one-hots (factorized only).
            codebook_embed_dim=64,  # Embedding dimension per code in the codebook.
            codebook_agg='sum',  # How to aggregate embeddings across factors: 'sum' or 'concat'.
            gumbel_temperature=1.0,  # Gumbel-Softmax temperature for subgoal encoder.
            gumbel_hard=True,  # Whether to use straight-through one-hot subgoals.
            value_loss_type='bce',  # Value loss type ('squared' or 'bce').
            flow_steps=10,  # Number of flow steps.
            num_samples=32,  # Number of samples for the actor.
            discrete=False,  # Whether the action space is discrete.
            encoder='none',  # Encoder name ('none' for no encoder, or e.g., 'resmlp_small').
            freeze_encoder=True,  # Whether to freeze encoder weights (if encoder is used).
            high_actor_warmup_steps=0,  # Steps to freeze high actor (0 = no warmup).
            low_actor_goal_conditioned=False,  # Whether low actor sees the final goal in addition to z.
            encode_goal=False,  # Whether to encode the goal before passing to low actor.
            goal_encoder_output_dim=64,  # Output dimension of goal encoder.
            use_resmlp=False,  # Whether to use ResMLP architecture for all networks.
            resmlp_hidden_dim=64,  # Hidden dimension for all ResMLP networks.
            resmlp_num_blocks=8,  # Number of residual blocks (4 layers each).
            # Subgoal encoder input flags (default: s'-only, which works best).
            use_subgoal_currstate=False,  # Include current state s in subgoal encoder (s' -> z becomes (s, s') -> z).
            use_subgoal_truegoal=False,  # Include goal g in subgoal encoder (s' -> z becomes (s', g) -> z).
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
