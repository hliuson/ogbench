import functools
from typing import Sequence, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp

from utils.networks import MLP


class FCLayer(nn.Module):
    """Fully connected layer with LayerNorm and Swish activation.

    From "1000 Layer Networks for Self-Supervised RL" (Blumenthal et al., 2025).
    https://arxiv.org/abs/2503.14858
    """

    features: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.features)(x)
        x = nn.LayerNorm()(x)
        x = nn.swish(x)
        return x


class ResMLPBlock(nn.Module):
    """Residual block with 4 consecutive FC layers.

    From "1000 Layer Networks for Self-Supervised RL" (Blumenthal et al., 2025).
    https://arxiv.org/abs/2503.14858
    """

    features: int

    @nn.compact
    def __call__(self, x):
        residual = x
        for _ in range(4):
            x = FCLayer(self.features)(x)
        return x + residual


class ResMLPEncoder(nn.Module):
    """Residual MLP encoder for state-based observations.

    From "1000 Layer Networks for Self-Supervised RL" (Blumenthal et al., 2025).
    https://arxiv.org/abs/2503.14858

    Architecture: FCLayer → [ResMLPBlock] × num_blocks
    - FCLayer: Dense → LayerNorm → Swish
    - ResMLPBlock: 4 × FCLayer + residual connection
    """

    hidden_dim: int = 256
    num_blocks: int = 2

    @nn.compact
    def __call__(self, x, train=True, cond_var=None):
        x = FCLayer(self.hidden_dim)(x)
        for _ in range(self.num_blocks):
            x = ResMLPBlock(self.hidden_dim)(x)
        return x


# Aliases for backward compatibility
MLPResnetBlock = ResMLPBlock
MLPEncoder = ResMLPEncoder


class ResnetStack(nn.Module):
    """ResNet stack module."""

    num_features: int
    num_blocks: int
    max_pooling: bool = True

    @nn.compact
    def __call__(self, x):
        initializer = nn.initializers.xavier_uniform()
        conv_out = nn.Conv(
            features=self.num_features,
            kernel_size=(3, 3),
            strides=1,
            kernel_init=initializer,
            padding='SAME',
        )(x)

        if self.max_pooling:
            conv_out = nn.max_pool(
                conv_out,
                window_shape=(3, 3),
                padding='SAME',
                strides=(2, 2),
            )

        for _ in range(self.num_blocks):
            block_input = conv_out
            conv_out = nn.relu(conv_out)
            conv_out = nn.Conv(
                features=self.num_features,
                kernel_size=(3, 3),
                strides=1,
                padding='SAME',
                kernel_init=initializer,
            )(conv_out)

            conv_out = nn.relu(conv_out)
            conv_out = nn.Conv(
                features=self.num_features,
                kernel_size=(3, 3),
                strides=1,
                padding='SAME',
                kernel_init=initializer,
            )(conv_out)
            conv_out += block_input

        return conv_out


class ImpalaEncoder(nn.Module):
    """IMPALA encoder."""

    width: int = 1
    stack_sizes: tuple = (16, 32, 32)
    num_blocks: int = 2
    dropout_rate: float = None
    mlp_hidden_dims: Sequence[int] = (512,)
    layer_norm: bool = False

    def setup(self):
        stack_sizes = self.stack_sizes
        self.stack_blocks = [
            ResnetStack(
                num_features=stack_sizes[i] * self.width,
                num_blocks=self.num_blocks,
            )
            for i in range(len(stack_sizes))
        ]
        if self.dropout_rate is not None:
            self.dropout = nn.Dropout(rate=self.dropout_rate)

    @nn.compact
    def __call__(self, x, train=True, cond_var=None):
        x = x.astype(jnp.float32) / 255.0

        conv_out = x

        for idx in range(len(self.stack_blocks)):
            conv_out = self.stack_blocks[idx](conv_out)
            if self.dropout_rate is not None:
                conv_out = self.dropout(conv_out, deterministic=not train)

        conv_out = nn.relu(conv_out)
        if self.layer_norm:
            conv_out = nn.LayerNorm()(conv_out)
        out = conv_out.reshape((*x.shape[:-3], -1))

        out = MLP(self.mlp_hidden_dims, activate_final=True, layer_norm=self.layer_norm)(out)

        return out

class DualEncoder(nn.Module):
    """Concatenate frozen high-level and trainable low-level encoders."""

    high_encoder: nn.Module
    low_encoder: nn.Module
    freeze_high: bool = False
    freeze_low: bool = False

    def __call__(self, x, train=True, cond_var=None):
        high = self.high_encoder(x, train=train, cond_var=cond_var)
        if self.freeze_high:
            high = jax.lax.stop_gradient(high)
        low = self.low_encoder(x, train=train, cond_var=cond_var)
        if self.freeze_low:
            low = jax.lax.stop_gradient(low)
        return jnp.concatenate([high, low], axis=-1)


class SplitEncoder(nn.Module):
    """Apply the same encoder to a concatenated input split in half."""

    encoder: nn.Module

    def __call__(self, x, train=True, cond_var=None):
        if x.shape[-1] % 2 != 0:
            raise ValueError(f'Input dim {x.shape[-1]} must be even to split state and goal.')
        first, second = jnp.split(x, 2, axis=-1)
        first_enc = self.encoder(first, train=train, cond_var=cond_var)
        second_enc = self.encoder(second, train=train, cond_var=cond_var)
        return jnp.concatenate([first_enc, second_enc], axis=-1)


class StopGradientEncoder(nn.Module):
    """Stop gradients through the wrapped encoder."""

    encoder: nn.Module

    def __call__(self, x, train=True, cond_var=None):
        out = self.encoder(x, train=train, cond_var=cond_var)
        return jax.lax.stop_gradient(out)


class GCEncoder(nn.Module):
    """Helper module to handle inputs to goal-conditioned networks.

    It takes in observations (s) and goals (g) and returns the concatenation of `state_encoder(s)`, `goal_encoder(g)`,
    and `concat_encoder([s, g])`. It ignores the encoders that are not provided. This way, the module can handle both
    early and late fusion (or their variants) of state and goal information.
    """

    state_encoder: nn.Module = None
    goal_encoder: nn.Module = None
    concat_encoder: nn.Module = None

    @nn.compact
    def __call__(self, observations, goals=None, goal_encoded=False):
        """Returns the representations of observations and goals.

        If `goal_encoded` is True, `goals` is assumed to be already encoded representations. In this case, either
        `goal_encoder` or `concat_encoder` must be None.
        """
        reps = []
        if self.state_encoder is not None:
            reps.append(self.state_encoder(observations))
        if goals is not None:
            if goal_encoded:
                # Can't have both goal_encoder and concat_encoder in this case.
                assert self.goal_encoder is None or self.concat_encoder is None
                reps.append(goals)
            else:
                if self.goal_encoder is not None:
                    reps.append(self.goal_encoder(goals))
                if self.concat_encoder is not None:
                    reps.append(self.concat_encoder(jnp.concatenate([observations, goals], axis=-1)))
        reps = jnp.concatenate(reps, axis=-1)
        return reps


encoder_modules = {
    'impala': ImpalaEncoder,
    'impala_debug': functools.partial(ImpalaEncoder, num_blocks=1, stack_sizes=(4, 4)),
    'impala_small': functools.partial(ImpalaEncoder, num_blocks=1),
    'impala_large': functools.partial(ImpalaEncoder, stack_sizes=(64, 128, 128), mlp_hidden_dims=(1024,)),
    'mlp': ResMLPEncoder,
    'mlp_small': functools.partial(ResMLPEncoder, num_blocks=1),
    'mlp_large': functools.partial(ResMLPEncoder, hidden_dim=512, num_blocks=4),
    # Explicit resmlp aliases
    'resmlp': ResMLPEncoder,
    'resmlp_small': functools.partial(ResMLPEncoder, num_blocks=1),
    'resmlp_large': functools.partial(ResMLPEncoder, hidden_dim=512, num_blocks=4),
}
