import copy
from typing import Any, Sequence

import flax
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax

import flax.linen as nn

import distrax

from utils.encoders import encoder_modules, ResMLPEncoder
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import ActorVectorField, GCDiscreteActor, GCValue, MLP, ResMLPActorVectorField, ResMLPValue


class SubgoalEncoder(nn.Module):
    """Option-style subgoal encoder: (s, s', g) -> z using shared MLP/ResMLP size."""

    use_resmlp: bool
    mlp_hidden_dims: Sequence[int]
    layer_norm: bool
    resmlp_hidden_dim: int
    resmlp_num_blocks: int
    z_dim: int = 8

    @nn.compact
    def __call__(self, x):
        if self.use_resmlp:
            x = ResMLPEncoder(hidden_dim=self.resmlp_hidden_dim, num_blocks=self.resmlp_num_blocks)(x)
            x = nn.Dense(self.z_dim)(x)
        else:
            x = MLP((*self.mlp_hidden_dims, self.z_dim), activate_final=False, layer_norm=self.layer_norm)(x)
        return x


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


class StateProbe(nn.Module):
    """Probe network that regresses state from latent subgoal z.

    Used as a diagnostic tool to understand what information is captured in z.
    Predicts both s' (current state) and oracle(s') from z.
    """

    hidden_dims: Sequence[int]
    output_dim: int
    layer_norm: bool = True

    @nn.compact
    def __call__(self, z):
        return MLP((*self.hidden_dims, self.output_dim), activate_final=False, layer_norm=self.layer_norm)(z)


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

    @staticmethod
    def _cube_triple_probe_breakdown(pred_state, next_obs):
        """Log cube-triple state slices based on cube_env.compute_observation()."""
        obs_dim = next_obs.shape[-1]
        if obs_dim != 46:
            return {}

        joint_pos_dim = 6
        joint_vel_dim = 6
        eff_pos_dim = 3
        eff_yaw_cos_dim = 1
        eff_yaw_sin_dim = 1
        gripper_open_dim = 1
        gripper_contact_dim = 1
        base_dim = (
            joint_pos_dim
            + joint_vel_dim
            + eff_pos_dim
            + eff_yaw_cos_dim
            + eff_yaw_sin_dim
            + gripper_open_dim
            + gripper_contact_dim
        )
        per_cube_dim = 3 + 4 + 1 + 1

        def mse(a, b):
            return jnp.mean((a - b) ** 2)

        eff_start = joint_pos_dim + joint_vel_dim
        eff_slice = slice(eff_start, eff_start + eff_pos_dim)
        eff_pos_loss = mse(pred_state[:, eff_slice], next_obs[:, eff_slice])

        cube_pos_losses = []
        cube_quat_losses = []
        cube_yaw_losses = []
        for i in range(3):
            start = base_dim + i * per_cube_dim
            pos_slice = slice(start, start + 3)
            quat_slice = slice(start + 3, start + 7)
            yaw_slice = slice(start + 7, start + 9)
            cube_pos_losses.append(mse(pred_state[:, pos_slice], next_obs[:, pos_slice]))
            cube_quat_losses.append(mse(pred_state[:, quat_slice], next_obs[:, quat_slice]))
            cube_yaw_losses.append(mse(pred_state[:, yaw_slice], next_obs[:, yaw_slice]))

        return {
            'state_effector_pos_loss': eff_pos_loss,
            'state_cube_pos_loss': jnp.mean(jnp.stack(cube_pos_losses)),
            'state_cube_quat_loss': jnp.mean(jnp.stack(cube_quat_losses)),
            'state_cube_yaw_loss': jnp.mean(jnp.stack(cube_yaw_losses)),
        }

    def _encode_observations(self, observations, grad_params=None):
        """Encode observations (identity if no encoder, otherwise use encoder)."""
        if self.config['encoder'] == 'none':
            return observations
        if self.config['freeze_encoder']:
            return self.network.select('encoder')(observations)
        else:
            return self.network.select('encoder')(observations, params=grad_params)

    def _encode_goals(self, goals, grad_params=None):
        """Encode goals (uses same encoder as observations)."""
        return self._encode_observations(goals, grad_params)

    def _encode_subgoal(self, observations, next_observations, goals, grad_params=None):
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
            enc_goals = self._encode_goals(goals, grad_params)
            inputs.append(enc_goals)  # append encoded g

        enc_input = jnp.concatenate(inputs, axis=-1)
        return self.network.select('subgoal_encoder')(enc_input, params=grad_params)

    def high_value_loss(self, batch, grad_params):
        """Compute the high-level SARSA value loss."""
        # Encode observations and goals for network inputs
        enc_obs = self._encode_observations(batch['observations'], grad_params)
        enc_goals = self._encode_goals(batch['high_value_goals'], grad_params)

        # Option-style subgoal encoding: (s, s', g) -> z
        z_target = self._encode_subgoal(
            batch['observations'],
            batch['high_value_next_observations'],
            batch['high_value_goals'],
            grad_params,
        )
        z_target = jax.lax.stop_gradient(z_target)
        # Target critic uses stop_gradient encoded inputs
        enc_obs_sg = jax.lax.stop_gradient(enc_obs)
        enc_goals_sg = jax.lax.stop_gradient(enc_goals)
        q1, q2 = self.network.select('target_high_critic')(
            enc_obs_sg, goals=enc_goals_sg, actions=z_target
        )
        if self.config['value_loss_type'] == 'bce':
            q1, q2 = jax.nn.sigmoid(q1), jax.nn.sigmoid(q2)

        if self.config['q_agg'] == 'min':
            q = jnp.minimum(q1, q2)
        elif self.config['q_agg'] == 'mean':
            q = (q1 + q2) / 2

        v = self.network.select('high_value')(enc_obs, enc_goals, params=grad_params)
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
        # Encode observations and goals for network inputs
        enc_obs = self._encode_observations(batch['observations'], grad_params)
        enc_goals = self._encode_goals(batch['high_value_goals'], grad_params)
        enc_next_obs = self._encode_observations(batch['high_value_next_observations'], grad_params)

        next_v = self.network.select('high_value')(
            jax.lax.stop_gradient(enc_next_obs),
            jax.lax.stop_gradient(enc_goals),
        )
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
            enc_obs, enc_goals, z_actions, params=grad_params
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
        """Compute the high-level flow BC actor loss."""
        # Encode observations and goals for network inputs
        enc_obs = self._encode_observations(batch['observations'], grad_params)
        enc_goals = self._encode_goals(batch['high_actor_goals'], grad_params)

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
            enc_obs, enc_goals, x_t, t, params=grad_params
        )

        actor_loss = jnp.mean((pred - y) ** 2)

        actor_info = {
            'actor_loss': actor_loss,
        }

        return actor_loss, actor_info

    def low_actor_loss(self, batch, grad_params, rng):
        """Compute the low-level actor loss (flow BC for continuous, cross-entropy for discrete)."""
        # Encode observations for network inputs
        enc_obs = self._encode_observations(batch['observations'], grad_params)

        # Option-style subgoal encoding: (s, s', g) -> z
        # Use high_actor_goals (final goal) to match inference-time z distribution
        subgoal_grad_params = None if self.config['policy_subgoal_stopgrad'] else grad_params
        z_goals = self._encode_subgoal(
            batch['observations'],
            batch['low_actor_goal_observations'],
            batch['high_actor_goals'],  # Must match high actor's goal conditioning
            subgoal_grad_params,
        )
        # Optionally condition low actor on the final goal as well
        if self.config['low_actor_goal_conditioned']:
            if self.config['encode_goal']:
                encoded_goal = self.network.select('goal_encoder')(
                    batch['high_actor_goals'], params=grad_params
                )
                z_goals = jnp.concatenate([z_goals, encoded_goal], axis=-1)
            else:
                # Encode goals before concatenating (needed for image observations)
                enc_goals = self._encode_goals(batch['high_actor_goals'], grad_params)
                z_goals = jnp.concatenate([z_goals, enc_goals], axis=-1)

        if self.config['discrete']:
            # Discrete actions: cross-entropy loss
            dist = self.network.select('low_actor')(enc_obs, z_goals, params=grad_params)
            actions = batch['actions']  # (batch_size,) integer indices
            log_probs = dist.log_prob(actions)
            actor_loss = -jnp.mean(log_probs)
            actor_info = {
                'actor_loss': actor_loss,
                'entropy': jnp.mean(dist.entropy()),
            }
        else:
            # Continuous actions: flow BC loss
            batch_size, action_dim = batch['actions'].shape
            x_rng, t_rng = jax.random.split(rng, 2)

            x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
            x_1 = batch['actions']
            t = jax.random.uniform(t_rng, (batch_size, 1))
            x_t = (1 - t) * x_0 + t * x_1
            y = x_1 - x_0

            pred = self.network.select('low_actor_flow')(enc_obs, z_goals, x_t, t, params=grad_params)
            actor_loss = jnp.mean((pred - y) ** 2)
            actor_info = {
                'actor_loss': actor_loss,
            }

        return actor_loss, actor_info

    def probe_loss(self, batch, grad_params):
        """Compute the probe loss for regressing s' from z.

        This is a diagnostic objective to understand what information is captured
        in the latent subgoal representation z. The probes are trained separately
        and do NOT affect the main network in any way.
        """
        # Compute z without grad_params (uses current network params, not being optimized here)
        # Then apply stop_gradient to ensure NO gradients flow back to the main network
        z = self._encode_subgoal(
            batch['observations'],
            batch['high_actor_next_observations'],
            batch['high_actor_goals'],
        )
        z = jax.lax.stop_gradient(z)  # Critical: isolates probes from main network

        # Get target states
        next_obs = batch['high_actor_next_observations']
        # Predict s' from z
        pred_state = self.network.select('state_probe')(z, params=grad_params)
        state_loss = jnp.mean((pred_state - next_obs) ** 2)
        total_probe_loss = self.config['probe_state_coef'] * state_loss

        probe_info = {
            'state_loss': state_loss,
            'total_loss': total_probe_loss,
        }
        probe_info.update(self._cube_triple_probe_breakdown(pred_state, next_obs))

        return total_probe_loss, probe_info

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

        # Probe loss for diagnostic analysis of z
        if self.config['use_probe']:
            probe_loss, probe_info = self.probe_loss(batch, grad_params)
            loss = loss + probe_loss
            for k, v in probe_info.items():
                info[f'probe/{k}'] = v

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

        # Encode observations and goals
        enc_observations = self._encode_observations(observations)
        enc_goals = self._encode_goals(goals)

        # High-level: rejection sampling.
        n_subgoals = jax.random.normal(
            high_seed,
            (
                self.config['num_samples'],
                *enc_observations.shape[:-1],
                self.config['z_dim'],
            ),
        )
        n_enc_observations = jnp.repeat(jnp.expand_dims(enc_observations, 0), self.config['num_samples'], axis=0)
        n_enc_goals = jnp.repeat(jnp.expand_dims(enc_goals, 0), self.config['num_samples'], axis=0)

        for i in range(self.config['flow_steps']):
            t = jnp.full((self.config['num_samples'], *enc_observations.shape[:-1], 1), i / self.config['flow_steps'])
            vels = self.network.select('high_actor_flow')(n_enc_observations, n_enc_goals, n_subgoals, t)
            n_subgoals = n_subgoals + vels / self.config['flow_steps']

        q = self.network.select('high_critic')(n_enc_observations, goals=n_enc_goals, actions=n_subgoals).min(axis=0)
        subgoals = n_subgoals[jnp.argmax(q)]

        # Low-level: behavioral cloning.
        # Optionally condition on the final goal as well
        if self.config['low_actor_goal_conditioned']:
            if self.config['encode_goal']:
                encoded_goal = self.network.select('goal_encoder')(goals)
                low_actor_goals = jnp.concatenate([subgoals, encoded_goal], axis=-1)
            else:
                low_actor_goals = jnp.concatenate([subgoals, enc_goals], axis=-1)
        else:
            low_actor_goals = subgoals

        if self.config['discrete']:
            # Discrete actions: sample from categorical distribution
            dist = self.network.select('low_actor')(enc_observations, low_actor_goals, temperature=temperature)
            actions = dist.sample(seed=low_seed)
        else:
            # Continuous actions: flow sampling
            actions = jax.random.normal(low_seed, (*enc_observations.shape[:-1], self.config['action_dim']))
            for i in range(self.config['flow_steps']):
                t = jnp.full((*enc_observations.shape[:-1], 1), i / self.config['flow_steps'])
                vels = self.network.select('low_actor_flow')(enc_observations, low_actor_goals, actions, t)
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
        batch_size = ex_observations.shape[0]
        ex_times = np.zeros((batch_size, 1), dtype=np.float32)  # Always (batch, 1) for flow timesteps
        ex_z = np.zeros((batch_size, config['z_dim']), dtype=np.float32)
        # Handle discrete vs continuous actions
        if config['discrete']:
            # For discrete actions, action_dim is the number of possible actions (derived from max action value)
            # main.py fills example_batch['actions'] with env.action_space.n - 1 to signal the action space size
            action_dim = int(np.max(ex_actions)) + 1
            ex_actions = ex_actions[:, np.newaxis]  # (batch,) -> (batch, 1) for any code expecting 2D
        else:
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
                hidden_dims=mlp_hidden_dims,
                layer_norm=config['layer_norm'],
                num_ensembles=1,
            )
            high_critic_def = GCValue(
                hidden_dims=mlp_hidden_dims,
                layer_norm=config['layer_norm'],
                num_ensembles=2,
            )
            high_actor_flow_def = ActorVectorField(
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
            ex_enc_goals = encoder_def.apply({'params': enc_shape_params}, ex_goals)
            enc_dim = ex_enc.shape[-1]
        else:
            encoder_def = None
            ex_enc = ex_observations
            ex_enc_goals = ex_goals
            enc_dim = obs_dim

        # Subgoal encoder: by default only s' -> z, optionally (s, s', g) -> z
        subgoal_encoder_def = SubgoalEncoder(
            use_resmlp=config['use_resmlp'],
            mlp_hidden_dims=mlp_hidden_dims,
            layer_norm=config['layer_norm'],
            resmlp_hidden_dim=config['resmlp_hidden_dim'],
            resmlp_num_blocks=config['resmlp_num_blocks'],
            z_dim=config['z_dim'],
        )
        # Build example input based on which components are enabled (all encoded)
        subgoal_inputs = [ex_enc]  # s' always included (encoded)
        if config['use_subgoal_currstate']:
            subgoal_inputs.insert(0, ex_enc)  # prepend s (encoded)
        if config['use_subgoal_truegoal']:
            subgoal_inputs.append(ex_enc_goals)  # append g (encoded)
        ex_subgoal_input = np.concatenate(subgoal_inputs, axis=-1)

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
            ex_low_actor_goal = np.concatenate([ex_z, ex_encoded_goal], axis=-1)
        elif config['low_actor_goal_conditioned']:
            # Goals are now encoded by the main encoder
            ex_low_actor_goal = np.concatenate([ex_z, ex_enc_goals], axis=-1)
        else:
            ex_low_actor_goal = ex_z

        # Create discrete low actor if needed
        if config['discrete']:
            low_actor_def = GCDiscreteActor(
                hidden_dims=mlp_hidden_dims,
                action_dim=action_dim,  # Number of discrete actions
            )

        # Networks now receive encoded observations and goals
        network_info = dict(
            subgoal_encoder=(subgoal_encoder_def, (ex_subgoal_input,)),
            high_value=(high_value_def, (ex_enc, ex_enc_goals)),
            high_critic=(high_critic_def, (ex_enc, ex_enc_goals, ex_z)),
            target_high_critic=(copy.deepcopy(high_critic_def), (ex_enc, ex_enc_goals, ex_z)),
            high_actor_flow=(high_actor_flow_def, (ex_enc, ex_enc_goals, ex_z, ex_times)),
        )
        # Add either discrete or continuous low actor
        if config['discrete']:
            network_info['low_actor'] = (low_actor_def, (ex_enc, ex_low_actor_goal))
        else:
            network_info['low_actor_flow'] = (low_actor_flow_def, (ex_enc, ex_low_actor_goal, ex_actions, ex_times))
        if encoder_def is not None:
            network_info['encoder'] = (encoder_def, (ex_observations,))
        if goal_encoder_def is not None:
            network_info['goal_encoder'] = (goal_encoder_def, (ex_goals,))

        # Probe networks for diagnostic analysis of z
        if config['use_probe']:
            probe_hidden_dims = (config['probe_hidden_dim'],) * config['probe_num_layers']
            # State probe: z -> s'
            state_probe_def = StateProbe(
                hidden_dims=probe_hidden_dims,
                output_dim=obs_dim,
                layer_norm=config['layer_norm'],
            )
            network_info['state_probe'] = (state_probe_def, (ex_z,))
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
            mlp_hidden_dim=256,  # Hidden width for all MLP networks.
            mlp_num_layers=4,  # Number of layers for all MLP networks.
            layer_norm=True,  # Whether to use layer normalization.
            discount=0.999,  # Discount factor.
            tau=0.005,  # Target network update rate.
            q_agg='min',  # Aggregation function for Q values.
            action_dim=ml_collections.config_dict.placeholder(int),  # Action dimension (set automatically).
            goal_dim=ml_collections.config_dict.placeholder(int),  # Goal dimension (set automatically).
            z_dim=8,  # Latent subgoal dimension.
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
            policy_subgoal_stopgrad=False,  # Whether to stop policy gradients into subgoal encoder.
            # Subgoal encoder input flags (default: s'-only, which works best).
            use_subgoal_currstate=False,  # Include current state s in subgoal encoder (s' -> z becomes (s, s') -> z).
            use_subgoal_truegoal=False,  # Include goal g in subgoal encoder (s' -> z becomes (s', g) -> z).
            # Probe hyperparameters (diagnostic tool to analyze z).
            use_probe=False,  # Whether to train probe networks.
            probe_hidden_dim=256,  # Hidden dimension for probe networks.
            probe_num_layers=2,  # Number of layers for probe networks.
            probe_state_coef=1.0,  # Coefficient for state probe loss (z -> s').
            # Only predict s' for probes; oracle probe removed.
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
