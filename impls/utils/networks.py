from typing import Any, Optional, Sequence

import distrax
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp


def default_init(scale=1.0):
    """Default kernel initializer."""
    return nn.initializers.variance_scaling(scale, 'fan_avg', 'uniform')


def ensemblize(cls, num_qs, out_axes=0, **kwargs):
    """Ensemblize a module."""
    return nn.vmap(
        cls,
        variable_axes={'params': 0},
        split_rngs={'params': True},
        in_axes=None,
        out_axes=out_axes,
        axis_size=num_qs,
        **kwargs,
    )


class Identity(nn.Module):
    """Identity layer."""

    def __call__(self, x):
        return x


class MLP(nn.Module):
    """Multi-layer perceptron.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        activations: Activation function.
        activate_final: Whether to apply activation to the final layer.
        kernel_init: Kernel initializer.
        layer_norm: Whether to apply layer normalization.
    """

    hidden_dims: Sequence[int]
    activations: Any = nn.gelu
    activate_final: bool = False
    kernel_init: Any = default_init()
    layer_norm: bool = False

    @nn.compact
    def __call__(self, x):
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=self.kernel_init)(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                x = self.activations(x)
                if self.layer_norm:
                    x = nn.LayerNorm()(x)
        return x


class ScalarTimeEmbedding(nn.Module):
    """Sinusoidal positional embedding for a scalar."""

    num_freqs: int = 32
    max_period: float = 10000.0
    scale: float = 1.0

    @nn.compact
    def __call__(self, t):
        if t.shape[-1] != 1:
            t = t[..., None]
        half_dim = self.num_freqs
        inv_freq = jnp.exp(
            -jnp.log(self.max_period) * jnp.arange(half_dim, dtype=t.dtype) / half_dim
        )
        emb = t * inv_freq * self.scale
        return jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=-1)


class TimeConditioner(nn.Module):
    """Positional-embed each scalar time and sum MLP projections."""

    num_freqs: int = 32
    hidden_dim: int = 128
    out_dim: int = 128
    max_period: float = 10000.0
    scale: float = 1.0
    layer_norm: bool = False

    @nn.compact
    def __call__(self, times):
        if times.ndim == 1:
            times = times[..., None]
        outputs = []
        for i in range(times.shape[-1]):
            emb = ScalarTimeEmbedding(
                num_freqs=self.num_freqs,
                max_period=self.max_period,
                scale=self.scale,
            )(times[..., i : i + 1])
            proj = MLP(
                (self.hidden_dim, self.out_dim),
                activate_final=True,
                layer_norm=self.layer_norm,
            )(emb)
            outputs.append(proj)
        return sum(outputs)


class LengthNormalize(nn.Module):
    """Length normalization layer.

    It normalizes the input along the last dimension to have a length of sqrt(dim).
    """

    @nn.compact
    def __call__(self, x):
        return x / jnp.linalg.norm(x, axis=-1, keepdims=True) * jnp.sqrt(x.shape[-1])


class Param(nn.Module):
    """Scalar parameter module."""

    init_value: float = 0.0

    @nn.compact
    def __call__(self):
        return self.param('value', init_fn=lambda key: jnp.full((), self.init_value))


class LogParam(nn.Module):
    """Scalar parameter module with log scale."""

    init_value: float = 1.0

    @nn.compact
    def __call__(self):
        log_value = self.param('log_value', init_fn=lambda key: jnp.full((), jnp.log(self.init_value)))
        return jnp.exp(log_value)


class TransformedWithMode(distrax.Transformed):
    """Transformed distribution with mode calculation."""

    def mode(self):
        return self.bijector.forward(self.distribution.mode())


class RunningMeanStd(flax.struct.PyTreeNode):
    """Running mean and standard deviation.

    Attributes:
        eps: Epsilon value to avoid division by zero.
        mean: Running mean.
        var: Running variance.
        clip_max: Clip value after normalization.
        count: Number of samples.
    """

    eps: Any = 1e-6
    mean: Any = 1.0
    var: Any = 1.0
    clip_max: Any = 10.0
    count: int = 0

    def normalize(self, batch):
        batch = (batch - self.mean) / jnp.sqrt(self.var + self.eps)
        batch = jnp.clip(batch, -self.clip_max, self.clip_max)
        return batch

    def unnormalize(self, batch):
        return batch * jnp.sqrt(self.var + self.eps) + self.mean

    def update(self, batch):
        batch_mean, batch_var = jnp.mean(batch, axis=0), jnp.var(batch, axis=0)
        batch_count = len(batch)

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
        new_var = m_2 / total_count

        return self.replace(mean=new_mean, var=new_var, count=total_count)


class GCActor(nn.Module):
    """Goal-conditioned actor.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        action_dim: Action dimension.
        log_std_min: Minimum value of log standard deviation.
        log_std_max: Maximum value of log standard deviation.
        tanh_squash: Whether to squash the action with tanh.
        state_dependent_std: Whether to use state-dependent standard deviation.
        const_std: Whether to use constant standard deviation.
        final_fc_init_scale: Initial scale of the final fully-connected layer.
        gc_encoder: Optional GCEncoder module to encode the inputs.
    """

    hidden_dims: Sequence[int]
    action_dim: int
    log_std_min: Optional[float] = -5
    log_std_max: Optional[float] = 2
    tanh_squash: bool = False
    state_dependent_std: bool = False
    const_std: bool = True
    final_fc_init_scale: float = 1e-2
    gc_encoder: nn.Module = None

    def setup(self):
        self.actor_net = MLP(self.hidden_dims, activate_final=True)
        self.mean_net = nn.Dense(self.action_dim, kernel_init=default_init(self.final_fc_init_scale))
        if self.state_dependent_std:
            self.log_std_net = nn.Dense(self.action_dim, kernel_init=default_init(self.final_fc_init_scale))
        else:
            if not self.const_std:
                self.log_stds = self.param('log_stds', nn.initializers.zeros, (self.action_dim,))

    def __call__(
        self,
        observations,
        goals=None,
        goal_encoded=False,
        temperature=1.0,
    ):
        """Return the action distribution.

        Args:
            observations: Observations.
            goals: Goals (optional).
            goal_encoded: Whether the goals are already encoded.
            temperature: Scaling factor for the standard deviation.
        """
        if self.gc_encoder is not None:
            inputs = self.gc_encoder(observations, goals, goal_encoded=goal_encoded)
        else:
            inputs = [observations]
            if goals is not None:
                inputs.append(goals)
            inputs = jnp.concatenate(inputs, axis=-1)
        outputs = self.actor_net(inputs)

        means = self.mean_net(outputs)
        if self.state_dependent_std:
            log_stds = self.log_std_net(outputs)
        else:
            if self.const_std:
                log_stds = jnp.zeros_like(means)
            else:
                log_stds = self.log_stds

        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)

        distribution = distrax.MultivariateNormalDiag(loc=means, scale_diag=jnp.exp(log_stds) * temperature)
        if self.tanh_squash:
            distribution = TransformedWithMode(distribution, distrax.Block(distrax.Tanh(), ndims=1))

        return distribution


class GCDiscreteActor(nn.Module):
    """Goal-conditioned actor for discrete actions.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        action_dim: Action dimension.
        final_fc_init_scale: Initial scale of the final fully-connected layer.
        gc_encoder: Optional GCEncoder module to encode the inputs.
    """

    hidden_dims: Sequence[int]
    action_dim: int
    final_fc_init_scale: float = 1e-2
    gc_encoder: nn.Module = None

    def setup(self):
        self.actor_net = MLP(self.hidden_dims, activate_final=True)
        self.logit_net = nn.Dense(self.action_dim, kernel_init=default_init(self.final_fc_init_scale))

    def __call__(
        self,
        observations,
        goals=None,
        goal_encoded=False,
        temperature=1.0,
    ):
        """Return the action distribution.

        Args:
            observations: Observations.
            goals: Goals (optional).
            goal_encoded: Whether the goals are already encoded.
            temperature: Inverse scaling factor for the logits (set to 0 to get the argmax).
        """
        if self.gc_encoder is not None:
            inputs = self.gc_encoder(observations, goals, goal_encoded=goal_encoded)
        else:
            inputs = [observations]
            if goals is not None:
                inputs.append(goals)
            inputs = jnp.concatenate(inputs, axis=-1)
        outputs = self.actor_net(inputs)

        logits = self.logit_net(outputs)

        distribution = distrax.Categorical(logits=logits / jnp.maximum(1e-6, temperature))

        return distribution


class GCValue(nn.Module):
    """Goal-conditioned value/critic function.

    This module can be used for both value V(s, g) and critic Q(s, a, g) functions.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        output_dim: Output dimension (set to None for scalar output).
        mlp_class: MLP class.
        layer_norm: Whether to apply layer normalization.
        ensemble: Whether to ensemble the value function.
        num_ensembles: Number of ensemble components (overrides ensemble when set).
        gc_encoder: Optional GCEncoder module to encode the inputs.
    """

    hidden_dims: Sequence[int]
    output_dim: int = None
    mlp_class: Any = MLP
    layer_norm: bool = True
    ensemble: bool = True
    num_ensembles: int = None
    gc_encoder: nn.Module = None

    def setup(self):
        mlp_module = self.mlp_class
        num_ensembles = self.num_ensembles
        if num_ensembles is None:
            num_ensembles = 2 if self.ensemble else 1
        if num_ensembles > 1:
            mlp_module = ensemblize(mlp_module, num_ensembles)
        output_dim = self.output_dim if self.output_dim is not None else 1
        value_net = mlp_module((*self.hidden_dims, output_dim), activate_final=False, layer_norm=self.layer_norm)

        self.value_net = value_net

    def __call__(self, observations, goals=None, actions=None):
        """Return the value/critic function.

        Args:
            observations: Observations.
            goals: Goals (optional).
            actions: Actions (optional).
        """
        if self.gc_encoder is not None:
            if goals is None:
                inputs = [self.gc_encoder(observations)]
            else:
                inputs = [self.gc_encoder(observations, goals)]
        else:
            inputs = [observations]
            if goals is not None:
                inputs.append(goals)
        if actions is not None:
            inputs.append(actions)
        inputs = jnp.concatenate(inputs, axis=-1)

        v = self.value_net(inputs)
        if self.output_dim is None:
            v = v.squeeze(-1)

        return v


class GCDiscreteCritic(GCValue):
    """Goal-conditioned critic for discrete actions."""

    action_dim: int = None

    def __call__(self, observations, goals=None, actions=None):
        actions = jnp.eye(self.action_dim)[actions]
        return super().__call__(observations, goals, actions)


class GCBilinearValue(nn.Module):
    """Goal-conditioned bilinear value/critic function.

    This module computes the value function as V(s, g) = phi(s)^T psi(g) / sqrt(d) or the critic function as
    Q(s, a, g) = phi(s, a)^T psi(g) / sqrt(d), where phi and psi output d-dimensional vectors.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        latent_dim: Latent dimension.
        layer_norm: Whether to apply layer normalization.
        ensemble: Whether to ensemble the value function.
        value_exp: Whether to exponentiate the value. Useful for contrastive learning.
        state_encoder: Optional state encoder.
        goal_encoder: Optional goal encoder.
    """

    hidden_dims: Sequence[int]
    latent_dim: int
    layer_norm: bool = True
    ensemble: bool = True
    value_exp: bool = False
    state_encoder: nn.Module = None
    goal_encoder: nn.Module = None

    def setup(self):
        mlp_module = MLP
        if self.ensemble:
            mlp_module = ensemblize(mlp_module, 2)

        self.phi = mlp_module((*self.hidden_dims, self.latent_dim), activate_final=False, layer_norm=self.layer_norm)
        self.psi = mlp_module((*self.hidden_dims, self.latent_dim), activate_final=False, layer_norm=self.layer_norm)

    def __call__(self, observations, goals, actions=None, info=False):
        """Return the value/critic function.

        Args:
            observations: Observations.
            goals: Goals.
            actions: Actions (optional).
            info: Whether to additionally return the representations phi and psi.
        """
        if self.state_encoder is not None:
            observations = self.state_encoder(observations)
        if self.goal_encoder is not None:
            goals = self.goal_encoder(goals)

        if actions is None:
            phi_inputs = observations
        else:
            phi_inputs = jnp.concatenate([observations, actions], axis=-1)

        phi = self.phi(phi_inputs)
        psi = self.psi(goals)

        v = (phi * psi / jnp.sqrt(self.latent_dim)).sum(axis=-1)

        if self.value_exp:
            v = jnp.exp(v)

        if info:
            return v, phi, psi
        else:
            return v


class GCDiscreteBilinearCritic(GCBilinearValue):
    """Goal-conditioned bilinear critic for discrete actions."""

    action_dim: int = None

    def __call__(self, observations, goals=None, actions=None, info=False):
        actions = jnp.eye(self.action_dim)[actions]
        return super().__call__(observations, goals, actions, info)


class GCMRNValue(nn.Module):
    """Metric residual network (MRN) value function.

    This module computes the value function as the sum of a symmetric Euclidean distance and an asymmetric
    L^infinity-based quasimetric.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        latent_dim: Latent dimension.
        layer_norm: Whether to apply layer normalization.
        encoder: Optional state/goal encoder.
    """

    hidden_dims: Sequence[int]
    latent_dim: int
    layer_norm: bool = True
    encoder: nn.Module = None

    def setup(self):
        self.phi = MLP((*self.hidden_dims, self.latent_dim), activate_final=False, layer_norm=self.layer_norm)

    def __call__(self, observations, goals, is_phi=False, info=False):
        """Return the MRN value function.

        Args:
            observations: Observations.
            goals: Goals.
            is_phi: Whether the inputs are already encoded by phi.
            info: Whether to additionally return the representations phi_s and phi_g.
        """
        if is_phi:
            phi_s = observations
            phi_g = goals
        else:
            if self.encoder is not None:
                observations = self.encoder(observations)
                goals = self.encoder(goals)
            phi_s = self.phi(observations)
            phi_g = self.phi(goals)

        sym_s = phi_s[..., : self.latent_dim // 2]
        sym_g = phi_g[..., : self.latent_dim // 2]
        asym_s = phi_s[..., self.latent_dim // 2 :]
        asym_g = phi_g[..., self.latent_dim // 2 :]
        squared_dist = ((sym_s - sym_g) ** 2).sum(axis=-1)
        quasi = jax.nn.relu((asym_s - asym_g).max(axis=-1))
        v = jnp.sqrt(jnp.maximum(squared_dist, 1e-12)) + quasi

        if info:
            return v, phi_s, phi_g
        else:
            return v


class GCIQEValue(nn.Module):
    """Interval quasimetric embedding (IQE) value function.

    This module computes the value function as an IQE-based quasimetric.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        latent_dim: Latent dimension.
        dim_per_component: Dimension of each component in IQE (i.e., number of intervals in each group).
        layer_norm: Whether to apply layer normalization.
        encoder: Optional state/goal encoder.
    """

    hidden_dims: Sequence[int]
    latent_dim: int
    dim_per_component: int
    layer_norm: bool = True
    encoder: nn.Module = None

    def setup(self):
        self.phi = MLP((*self.hidden_dims, self.latent_dim), activate_final=False, layer_norm=self.layer_norm)
        self.alpha = Param()

    def __call__(self, observations, goals, is_phi=False, info=False):
        """Return the IQE value function.

        Args:
            observations: Observations.
            goals: Goals.
            is_phi: Whether the inputs are already encoded by phi.
            info: Whether to additionally return the representations phi_s and phi_g.
        """
        alpha = jax.nn.sigmoid(self.alpha())
        if is_phi:
            phi_s = observations
            phi_g = goals
        else:
            if self.encoder is not None:
                observations = self.encoder(observations)
                goals = self.encoder(goals)
            phi_s = self.phi(observations)
            phi_g = self.phi(goals)

        x = jnp.reshape(phi_s, (*phi_s.shape[:-1], -1, self.dim_per_component))
        y = jnp.reshape(phi_g, (*phi_g.shape[:-1], -1, self.dim_per_component))
        valid = x < y
        xy = jnp.concatenate(jnp.broadcast_arrays(x, y), axis=-1)
        ixy = xy.argsort(axis=-1)
        sxy = jnp.take_along_axis(xy, ixy, axis=-1)
        neg_inc_copies = jnp.take_along_axis(valid, ixy % self.dim_per_component, axis=-1) * jnp.where(
            ixy < self.dim_per_component, -1, 1
        )
        neg_inp_copies = jnp.cumsum(neg_inc_copies, axis=-1)
        neg_f = -1.0 * (neg_inp_copies < 0)
        neg_incf = jnp.concatenate([neg_f[..., :1], neg_f[..., 1:] - neg_f[..., :-1]], axis=-1)
        components = (sxy * neg_incf).sum(axis=-1)
        v = alpha * components.mean(axis=-1) + (1 - alpha) * components.max(axis=-1)

        if info:
            return v, phi_s, phi_g
        else:
            return v


class ResMLPNetwork(nn.Module):
    """ResMLP backbone network using residual blocks with LayerNorm and Swish.

    From "1000 Layer Networks for Self-Supervised RL" (Blumenthal et al., 2025).
    https://arxiv.org/abs/2503.14858

    Architecture: FCLayer → [ResMLPBlock] × num_blocks → Dense(output_dim)
    - FCLayer: Dense → LayerNorm → Swish
    - ResMLPBlock: 4 × FCLayer + residual connection

    Attributes:
        hidden_dim: Hidden dimension (same for all layers due to residual connections).
        num_blocks: Number of residual blocks.
        output_dim: Output dimension.
    """

    hidden_dim: int = 256
    num_blocks: int = 2
    output_dim: int = 1

    @nn.compact
    def __call__(self, x):
        # Initial projection to hidden_dim
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.LayerNorm()(x)
        x = nn.swish(x)

        # Residual blocks
        for _ in range(self.num_blocks):
            residual = x
            for _ in range(4):
                x = nn.Dense(self.hidden_dim)(x)
                x = nn.LayerNorm()(x)
                x = nn.swish(x)
            x = x + residual

        # Output projection
        x = nn.Dense(self.output_dim)(x)
        return x


class ResMLPValue(nn.Module):
    """Goal-conditioned value/critic function using ResMLP architecture.

    Attributes:
        hidden_dim: Hidden dimension for ResMLP.
        num_blocks: Number of residual blocks.
        num_ensembles: Number of ensemble components.
    """

    hidden_dim: int = 256
    num_blocks: int = 2
    num_ensembles: int = 1

    def setup(self):
        resmlp_module = ResMLPNetwork
        if self.num_ensembles > 1:
            resmlp_module = ensemblize(resmlp_module, self.num_ensembles)
        self.value_net = resmlp_module(
            hidden_dim=self.hidden_dim,
            num_blocks=self.num_blocks,
            output_dim=1,
        )

    def __call__(self, observations, goals=None, actions=None):
        """Return the value/critic function."""
        inputs = [observations]
        if goals is not None:
            inputs.append(goals)
        if actions is not None:
            inputs.append(actions)
        inputs = jnp.concatenate(inputs, axis=-1)

        v = self.value_net(inputs)
        v = v.squeeze(-1)
        return v


class ResMLPActorVectorField(nn.Module):
    """Actor vector field for flow policies using ResMLP architecture.

    Attributes:
        hidden_dim: Hidden dimension for ResMLP.
        num_blocks: Number of residual blocks.
        action_dim: Action dimension.
    """

    hidden_dim: int = 256
    num_blocks: int = 2
    action_dim: int = 1

    def setup(self):
        self.resmlp = ResMLPNetwork(
            hidden_dim=self.hidden_dim,
            num_blocks=self.num_blocks,
            output_dim=self.action_dim,
        )

    def __call__(self, observations, goals=None, actions=None, times=None, is_encoded=False):
        """Return the current vector."""
        if goals is None:
            inputs = observations
        else:
            inputs = jnp.concatenate([observations, goals], axis=-1)
        if times is None:
            inputs = jnp.concatenate([inputs, actions], axis=-1)
        else:
            inputs = jnp.concatenate([inputs, actions, times], axis=-1)

        v = self.resmlp(inputs)
        return v


class ActorVectorField(nn.Module):
    """Actor vector field for flow policies.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        action_dim: Action dimension.
        mlp_class: MLP class.
        activate_final: Whether to apply activation to the final layer.
        layer_norm: Whether to apply layer normalization.
        gc_encoder: Optional GCEncoder module to encode the inputs.
        time_encoder: Optional module to encode time inputs.
    """

    hidden_dims: Sequence[int]
    action_dim: int
    mlp_class: Any = MLP
    activate_final: bool = False
    layer_norm: bool = False
    gc_encoder: nn.Module = None
    time_encoder: nn.Module = None

    def setup(self):
        self.mlp = self.mlp_class(
            (*self.hidden_dims, self.action_dim), activate_final=False, layer_norm=self.layer_norm
        )

    @nn.compact
    def __call__(self, observations, goals=None, actions=None, times=None, is_encoded=False):
        """Return the current vector.

        Args:
            observations: Observations.
            goals: Goals (optional).
            actions: Current actions.
            times: Current times (optional).
            is_encoded: Whether the inputs are already encoded.
        """
        if not is_encoded and self.gc_encoder is not None:
            if goals is None:
                inputs = self.gc_encoder(observations)
            else:
                inputs = self.gc_encoder(observations, goals)
        else:
            if goals is None:
                inputs = observations
            else:
                inputs = jnp.concatenate([observations, goals], axis=-1)
        if times is None:
            inputs = jnp.concatenate([inputs, actions], axis=-1)
        else:
            if self.time_encoder is not None:
                times = self.time_encoder(times)
            inputs = jnp.concatenate([inputs, actions, times], axis=-1)

        v = self.mlp(inputs)

        return v


class ActorVectorFieldDualHead(nn.Module):
    """Actor vector field with dual heads (u, v) for iMF.

    Attributes:
        hidden_dims: Hidden layer dimensions for the shared trunk.
        action_dim: Action dimension.
        mlp_class: MLP class.
        layer_norm: Whether to apply layer normalization.
        gc_encoder: Optional GCEncoder module to encode the inputs.
        time_encoder: Optional module to encode time inputs.
        head_hidden_dims: Optional hidden dims for each head MLP. If None, use linear heads.
    """

    hidden_dims: Sequence[int]
    action_dim: int
    mlp_class: Any = MLP
    layer_norm: bool = False
    gc_encoder: nn.Module = None
    time_encoder: nn.Module = None
    head_hidden_dims: Optional[Sequence[int]] = None

    def setup(self):
        self.trunk = self.mlp_class(
            self.hidden_dims, activate_final=True, layer_norm=self.layer_norm
        )
        if self.head_hidden_dims:
            self.u_head = self.mlp_class(
                (*self.head_hidden_dims, self.action_dim),
                activate_final=False,
                layer_norm=self.layer_norm,
            )
            self.v_head = self.mlp_class(
                (*self.head_hidden_dims, self.action_dim),
                activate_final=False,
                layer_norm=self.layer_norm,
            )
        else:
            self.u_head = nn.Dense(self.action_dim, kernel_init=default_init())
            self.v_head = nn.Dense(self.action_dim, kernel_init=default_init())

    @nn.compact
    def __call__(self, observations, goals=None, actions=None, times=None, is_encoded=False):
        """Return (u, v) vector fields."""
        if not is_encoded and self.gc_encoder is not None:
            if goals is None:
                inputs = self.gc_encoder(observations)
            else:
                inputs = self.gc_encoder(observations, goals)
        else:
            if goals is None:
                inputs = observations
            else:
                inputs = jnp.concatenate([observations, goals], axis=-1)
        if times is None:
            inputs = jnp.concatenate([inputs, actions], axis=-1)
        else:
            if self.time_encoder is not None:
                times = self.time_encoder(times)
            inputs = jnp.concatenate([inputs, actions, times], axis=-1)

        feats = self.trunk(inputs)
        u = self.u_head(feats)
        v = self.v_head(feats)
        return u, v
