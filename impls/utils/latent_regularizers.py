"""Latent-space regularizers.

This file intentionally contains standalone utilities only. Nothing here is wired
into an agent yet.

The SIGReg implementation follows the same structure as the local lejepa
reference clone:
  - univariate Epps-Pulley statistic on projected samples
  - multivariate random slicing over unit Gaussian directions

Reference used during implementation:
  - /tmp/lejepa.eUgEdQ/lejepa/univariate/epps_pulley.py
  - /tmp/lejepa.eUgEdQ/lejepa/multivariate/slicing.py
"""

import jax
import jax.numpy as jnp
from typing import Optional


def _epps_pulley_grid(
    num_points: int = 17,
    t_max: float = 3.0,
    dtype=jnp.float32,
):
    """Build the integration grid used by the Epps-Pulley statistic.

    This mirrors the upstream lejepa implementation:
      - linearly spaced points on [0, t_max]
      - trapezoidal rule
      - positive-half integration with the Gaussian weight folded into the
        quadrature weights
    """
    if num_points < 3 or num_points % 2 == 0:
        raise ValueError(f"num_points must be an odd integer >= 3, got {num_points}")
    if t_max <= 0:
        raise ValueError(f"t_max must be positive, got {t_max}")

    t = jnp.linspace(0.0, t_max, num_points, dtype=dtype)
    dt = jnp.asarray(t_max / (num_points - 1), dtype=dtype)
    base_weights = jnp.full((num_points,), 2.0 * dt, dtype=dtype)
    base_weights = base_weights.at[0].set(dt)
    base_weights = base_weights.at[-1].set(dt)
    phi = jnp.exp(-0.5 * jnp.square(t))
    weights = base_weights * phi
    return t, phi, weights


def epps_pulley_statistic(
    projected: jax.Array,
    num_points: int = 17,
    t_max: float = 3.0,
):
    """Compute the Epps-Pulley normality statistic on projected samples.

    Args:
      projected:
        Array shaped `(..., num_samples, num_slices)`. The sample axis is `-2`
        and the slice axis is `-1`.
      num_points:
        Number of trapezoidal integration points on `[0, t_max]`.
      t_max:
        Positive integration limit.

    Returns:
      Array shaped `(..., num_slices)` containing one statistic per slice.
    """
    if projected.ndim < 2:
        raise ValueError(f"projected must have shape (..., num_samples, num_slices), got {projected.shape}")

    projected = jnp.asarray(projected)
    num_samples = projected.shape[-2]
    t, phi, weights = _epps_pulley_grid(
        num_points=num_points,
        t_max=t_max,
        dtype=projected.dtype,
    )

    # Shape: (..., num_samples, num_slices, num_points)
    xt = projected[..., :, :, None] * t
    cos_mean = jnp.mean(jnp.cos(xt), axis=-3)
    sin_mean = jnp.mean(jnp.sin(xt), axis=-3)

    err = jnp.square(cos_mean - phi) + jnp.square(sin_mean)
    return jnp.tensordot(err, weights, axes=([-1], [0])) * num_samples


def sample_unit_gaussian_slices(
    rng: jax.Array,
    dim: int,
    num_slices: int,
    dtype=jnp.float32,
):
    """Sample random unit vectors from a Gaussian and normalize columns."""
    if dim <= 0:
        raise ValueError(f"dim must be positive, got {dim}")
    if num_slices <= 0:
        raise ValueError(f"num_slices must be positive, got {num_slices}")

    directions = jax.random.normal(rng, (dim, num_slices), dtype=dtype)
    norms = jnp.linalg.norm(directions, axis=0, keepdims=True)
    norms = jnp.maximum(norms, jnp.asarray(1e-8, dtype=dtype))
    return directions / norms


def sliced_epps_pulley_statistics(
    embeddings: jax.Array,
    rng: jax.Array,
    num_slices: int = 32,
    num_points: int = 17,
    t_max: float = 3.0,
    clip_value: Optional[float] = None,
):
    """Compute one Epps-Pulley statistic per random slice.

    Args:
      embeddings:
        Array shaped `(..., num_samples, dim)`.
      rng:
        PRNG key used to sample slicing directions.
      num_slices:
        Number of random unit vectors.
      num_points:
        Number of Epps-Pulley integration points.
      t_max:
        Positive integration limit.
      clip_value:
        Optional lower threshold. Statistics below this value are zeroed, which
        mirrors lejepa's optional clipping behavior.

    Returns:
      Array shaped `(..., num_slices)`.
    """
    if embeddings.ndim < 2:
        raise ValueError(f"embeddings must have shape (..., num_samples, dim), got {embeddings.shape}")

    embeddings = jnp.asarray(embeddings)
    directions = sample_unit_gaussian_slices(
        rng,
        dim=embeddings.shape[-1],
        num_slices=num_slices,
        dtype=embeddings.dtype,
    )
    projected = jnp.einsum('...nd,dk->...nk', embeddings, directions)
    stats = epps_pulley_statistic(
        projected,
        num_points=num_points,
        t_max=t_max,
    )
    if clip_value is not None:
        clip_value = jnp.asarray(clip_value, dtype=stats.dtype)
        stats = jnp.where(stats < clip_value, 0.0, stats)
    return stats


def sliced_epps_pulley_loss(
    embeddings: jax.Array,
    rng: jax.Array,
    num_slices: int = 32,
    num_points: int = 17,
    t_max: float = 3.0,
    clip_value: Optional[float] = None,
    reduction: str = "mean",
):
    """Aggregate the sliced Epps-Pulley statistics into a scalar loss."""
    stats = sliced_epps_pulley_statistics(
        embeddings,
        rng,
        num_slices=num_slices,
        num_points=num_points,
        t_max=t_max,
        clip_value=clip_value,
    )
    if reduction == "mean":
        return jnp.mean(stats)
    if reduction == "sum":
        return jnp.sum(stats)
    if reduction in ("none", None):
        return stats
    raise ValueError(f"Unknown reduction {reduction!r}")
