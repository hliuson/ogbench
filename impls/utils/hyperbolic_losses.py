"""Hyperbolic losses for representation learning in JAX.

This module provides Lorentz-model geometry utilities and loss functions
adapted for generic representation learning pipelines:
- hyperbolic contrastive loss over paired samples
- hyperbolic entailment cone loss
- joint objective combining both terms
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp


Array = jax.Array


def _positive_curvature(curvature: float | Array, dtype: jnp.dtype, eps: float = 1e-8) -> Array:
    """Return a numerically-stable positive curvature tensor."""
    curv = jnp.asarray(curvature, dtype=dtype)
    return jnp.maximum(curv, eps)


def exp_map0(x: Array, curvature: float | Array = 1.0, eps: float = 1e-8) -> Array:
    """Map Euclidean tangent vectors at origin to Lorentz hyperboloid space."""
    curv = _positive_curvature(curvature, x.dtype, eps=eps)
    rc_xnorm = jnp.sqrt(curv) * jnp.linalg.norm(x, axis=-1, keepdims=True)
    sinh_input = jnp.clip(rc_xnorm, a_min=eps, a_max=math.asinh(2**15))
    return jnp.sinh(sinh_input) * x / jnp.clip(rc_xnorm, a_min=eps)


def pairwise_lorentz_inner(x: Array, y: Array, curvature: float | Array = 1.0, eps: float = 1e-8) -> Array:
    """Pairwise Lorentzian inner product for hyperboloid space components."""
    curv = _positive_curvature(curvature, x.dtype, eps=eps)
    x_time = jnp.sqrt(1.0 / curv + jnp.sum(x**2, axis=-1, keepdims=True))
    y_time = jnp.sqrt(1.0 / curv + jnp.sum(y**2, axis=-1, keepdims=True))
    return x @ y.T - x_time @ y_time.T


def pairwise_hyperbolic_distance(
    x: Array,
    y: Array,
    curvature: float | Array = 1.0,
    eps: float = 1e-8,
) -> Array:
    """Pairwise geodesic distance on the Lorentz hyperboloid."""
    curv = _positive_curvature(curvature, x.dtype, eps=eps)
    c_xyl = -curv * pairwise_lorentz_inner(x, y, curvature=curv, eps=eps)
    distance = jnp.arccosh(jnp.clip(c_xyl, a_min=1.0 + eps))
    return distance / jnp.sqrt(curv)


def entailment_half_aperture(
    x: Array,
    curvature: float | Array = 1.0,
    min_radius: float = 0.1,
    eps: float = 1e-8,
) -> Array:
    """Half-aperture angle for entailment cones with apex at each point in `x`."""
    curv = _positive_curvature(curvature, x.dtype, eps=eps)
    norm = jnp.linalg.norm(x, axis=-1)
    asin_input = 2.0 * min_radius / (norm * jnp.sqrt(curv) + eps)
    asin_input = jnp.clip(asin_input, a_min=-1.0 + eps, a_max=1.0 - eps)
    return jnp.arcsin(asin_input)


def oxy_angle(x: Array, y: Array, curvature: float | Array = 1.0, eps: float = 1e-8) -> Array:
    """Exterior angle at `x` in hyperbolic triangle Oxy (O = origin)."""
    curv = _positive_curvature(curvature, x.dtype, eps=eps)

    x_time = jnp.sqrt(1.0 / curv + jnp.sum(x**2, axis=-1))
    y_time = jnp.sqrt(1.0 / curv + jnp.sum(y**2, axis=-1))
    c_xyl = curv * (jnp.sum(x * y, axis=-1) - x_time * y_time)

    acos_numer = y_time + c_xyl * x_time
    acos_denom = jnp.sqrt(jnp.clip(c_xyl**2 - 1.0, a_min=eps))
    acos_input = acos_numer / (jnp.linalg.norm(x, axis=-1) * acos_denom + eps)
    acos_input = jnp.clip(acos_input, a_min=-1.0 + eps, a_max=1.0 - eps)
    return jnp.arccos(acos_input)


def _cross_entropy_from_logits(logits: Array, targets: Array) -> Array:
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    row_index = jnp.arange(logits.shape[0])
    return -jnp.mean(log_probs[row_index, targets])


def _validate_targets(targets: Array, batch_size: int, _num_classes: int) -> Array:
    if targets.ndim != 1:
        raise ValueError(f'Expected 1D targets, got shape {targets.shape}.')
    if targets.shape[0] != batch_size:
        raise ValueError(f'Expected targets shape ({batch_size},), got {targets.shape}.')
    if not jnp.issubdtype(targets.dtype, jnp.integer):
        raise ValueError(f'Expected integer targets, got {targets.dtype}.')
    return targets.astype(jnp.int32)


def hyperbolic_contrastive_loss(
    query_embeddings: Array,
    key_embeddings: Array,
    *,
    curvature: float | Array = 1.0,
    logit_scale: float | Array = 1.0,
    target_indices: Array | None = None,
    reverse_target_indices: Array | None = None,
    symmetric: bool = True,
    project_inputs: bool = False,
    eps: float = 1e-8,
) -> tuple[Array, dict[str, Array]]:
    """Hyperbolic contrastive loss using negative Lorentz geodesic distance.

    Args:
        query_embeddings: Array of shape `(Bq, D)` for anchor/query features.
        key_embeddings: Array of shape `(Bk, D)` for key/positive-negative pool.
        curvature: Positive scalar curvature value.
        logit_scale: Multiplicative scale on logits (temperature inverse).
        target_indices: Optional integer tensor of shape `(Bq,)`. If `None`,
            diagonal targets are used and `Bq == Bk` is required.
        reverse_target_indices: Optional targets for key-to-query direction,
            used only when `symmetric=True`.
        symmetric: If `True`, average query->key and key->query cross entropy.
        project_inputs: If `True`, project Euclidean inputs via `exp_map0`.
        eps: Numerical stability epsilon.
    """
    if query_embeddings.ndim != 2 or key_embeddings.ndim != 2:
        raise ValueError(
            'Expected 2D embeddings for contrastive loss: '
            f'got {query_embeddings.shape} and {key_embeddings.shape}.'
        )
    if query_embeddings.shape[1] != key_embeddings.shape[1]:
        raise ValueError(
            f'Embedding dimensions must match, got {query_embeddings.shape[1]} and {key_embeddings.shape[1]}.'
        )

    query = exp_map0(query_embeddings, curvature=curvature, eps=eps) if project_inputs else query_embeddings
    key = exp_map0(key_embeddings, curvature=curvature, eps=eps) if project_inputs else key_embeddings

    num_queries = query.shape[0]
    num_keys = key.shape[0]
    using_diagonal_targets = target_indices is None
    if using_diagonal_targets:
        if num_queries != num_keys:
            raise ValueError(
                'Diagonal targets require equal batch sizes, '
                f'got query batch {num_queries} and key batch {num_keys}.'
            )
        target_indices = jnp.arange(num_queries, dtype=jnp.int32)
    else:
        target_indices = _validate_targets(target_indices, num_queries, num_keys)

    logits_qk = -pairwise_hyperbolic_distance(query, key, curvature=curvature, eps=eps)
    scale = jnp.asarray(logit_scale, dtype=logits_qk.dtype)
    scaled_qk = scale * logits_qk

    qk_loss = _cross_entropy_from_logits(scaled_qk, target_indices)
    qk_acc = jnp.mean(jnp.argmax(logits_qk, axis=-1) == target_indices)
    logits_pos = jnp.mean(logits_qk[jnp.arange(num_queries), target_indices])
    full_mean = jnp.mean(logits_qk)
    neg_count = logits_qk.size - num_queries
    pos_sum = jnp.sum(logits_qk[jnp.arange(num_queries), target_indices])
    logits_neg = jnp.where(
        neg_count > 0,
        (jnp.sum(logits_qk) - pos_sum) / neg_count,
        0.0,
    )

    total_loss = qk_loss
    metrics = {
        'contrastive/query_to_key_loss': qk_loss,
        'contrastive/query_to_key_accuracy': qk_acc,
        'contrastive/logits_mean': full_mean,
        'contrastive/logits_pos': logits_pos,
        'contrastive/logits_neg': logits_neg,
    }

    if symmetric:
        logits_kq = -pairwise_hyperbolic_distance(key, query, curvature=curvature, eps=eps)
        if reverse_target_indices is None:
            if using_diagonal_targets and num_queries == num_keys:
                reverse_target_indices = jnp.arange(num_keys, dtype=jnp.int32)
            else:
                raise ValueError(
                    'Symmetric loss with custom `target_indices` requires '
                    '`reverse_target_indices` explicitly.'
                )
        reverse_target_indices = _validate_targets(reverse_target_indices, num_keys, num_queries)
        scaled_kq = scale * logits_kq
        kq_loss = _cross_entropy_from_logits(scaled_kq, reverse_target_indices)
        kq_acc = jnp.mean(jnp.argmax(logits_kq, axis=-1) == reverse_target_indices)
        total_loss = 0.5 * (qk_loss + kq_loss)
        metrics['contrastive/key_to_query_loss'] = kq_loss
        metrics['contrastive/key_to_query_accuracy'] = kq_acc
    else:
        metrics['contrastive/key_to_query_loss'] = jnp.asarray(0.0, dtype=qk_loss.dtype)
        metrics['contrastive/key_to_query_accuracy'] = jnp.asarray(0.0, dtype=qk_loss.dtype)

    metrics['contrastive/loss'] = total_loss
    metrics['contrastive/logit_scale'] = scale

    return total_loss, metrics


def hyperbolic_entailment_loss(
    entailing_embeddings: Array,
    entailed_embeddings: Array,
    *,
    curvature: float | Array = 1.0,
    min_radius: float = 0.1,
    pair_mask: Array | None = None,
    project_inputs: bool = False,
    eps: float = 1e-8,
) -> tuple[Array, dict[str, Array]]:
    """Hyperbolic entailment cone loss.

    `entailing_embeddings[i]` should entail `entailed_embeddings[i]`.
    """
    if entailing_embeddings.ndim != 2 or entailed_embeddings.ndim != 2:
        raise ValueError(
            'Expected 2D embeddings for entailment loss: '
            f'got {entailing_embeddings.shape} and {entailed_embeddings.shape}.'
        )
    if entailing_embeddings.shape != entailed_embeddings.shape:
        raise ValueError(
            'Entailing and entailed tensors must have the same shape, '
            f'got {entailing_embeddings.shape} and {entailed_embeddings.shape}.'
        )

    entailing = exp_map0(entailing_embeddings, curvature=curvature, eps=eps) if project_inputs else entailing_embeddings
    entailed = exp_map0(entailed_embeddings, curvature=curvature, eps=eps) if project_inputs else entailed_embeddings

    angle = oxy_angle(entailing, entailed, curvature=curvature, eps=eps)
    aperture = entailment_half_aperture(entailing, curvature=curvature, min_radius=min_radius, eps=eps)
    violations = jnp.maximum(angle - aperture, 0.0)

    if pair_mask is None:
        entail_loss = jnp.mean(violations)
        active_fraction = jnp.mean(violations > 0.0)
    else:
        if pair_mask.ndim != 1 or pair_mask.shape[0] != violations.shape[0]:
            raise ValueError(
                f'Expected pair_mask shape ({violations.shape[0]},), got {pair_mask.shape}.'
            )
        mask = pair_mask.astype(violations.dtype)
        denom = jnp.maximum(jnp.sum(mask), 1.0)
        entail_loss = jnp.sum(mask * violations) / denom
        active_fraction = jnp.sum(mask * (violations > 0.0)) / denom

    metrics = {
        'entailment/loss': entail_loss,
        'entailment/angle_mean': jnp.mean(angle),
        'entailment/aperture_mean': jnp.mean(aperture),
        'entailment/violation_mean': jnp.mean(violations),
        'entailment/active_fraction': active_fraction,
    }
    return entail_loss, metrics


def hyperbolic_representation_loss(
    query_embeddings: Array,
    key_embeddings: Array,
    *,
    curvature: float | Array = 1.0,
    logit_scale: float | Array = 1.0,
    target_indices: Array | None = None,
    reverse_target_indices: Array | None = None,
    symmetric: bool = True,
    entailing_embeddings: Array | None = None,
    entailed_embeddings: Array | None = None,
    entailment_weight: float = 0.0,
    entailment_min_radius: float = 0.1,
    entailment_pair_mask: Array | None = None,
    project_contrastive_inputs: bool = False,
    project_entailment_inputs: bool = False,
    eps: float = 1e-8,
) -> tuple[Array, dict[str, Array]]:
    """Joint hyperbolic objective for contrastive and entailment supervision.

    This function is designed as a drop-in utility for any agent that can
    provide paired representations.
    """
    contrastive_loss, contrastive_metrics = hyperbolic_contrastive_loss(
        query_embeddings,
        key_embeddings,
        curvature=curvature,
        logit_scale=logit_scale,
        target_indices=target_indices,
        reverse_target_indices=reverse_target_indices,
        symmetric=symmetric,
        project_inputs=project_contrastive_inputs,
        eps=eps,
    )

    total_loss = contrastive_loss
    metrics = dict(contrastive_metrics)

    if entailment_weight > 0.0:
        if entailing_embeddings is None or entailed_embeddings is None:
            raise ValueError(
                'Entailment supervision requested (`entailment_weight > 0`) but '
                '`entailing_embeddings`/`entailed_embeddings` were not both provided.'
            )
        entailment_loss, entailment_metrics = hyperbolic_entailment_loss(
            entailing_embeddings,
            entailed_embeddings,
            curvature=curvature,
            min_radius=entailment_min_radius,
            pair_mask=entailment_pair_mask,
            project_inputs=project_entailment_inputs,
            eps=eps,
        )
        total_loss = total_loss + entailment_weight * entailment_loss
        metrics.update(entailment_metrics)
    else:
        metrics['entailment/loss'] = jnp.asarray(0.0, dtype=total_loss.dtype)
        metrics['entailment/angle_mean'] = jnp.asarray(0.0, dtype=total_loss.dtype)
        metrics['entailment/aperture_mean'] = jnp.asarray(0.0, dtype=total_loss.dtype)
        metrics['entailment/violation_mean'] = jnp.asarray(0.0, dtype=total_loss.dtype)
        metrics['entailment/active_fraction'] = jnp.asarray(0.0, dtype=total_loss.dtype)

    metrics['loss/contrastive'] = contrastive_loss
    metrics['loss/entailment_weight'] = jnp.asarray(entailment_weight, dtype=total_loss.dtype)
    metrics['loss/total'] = total_loss
    metrics['hyperbolic/curvature'] = jnp.asarray(curvature, dtype=total_loss.dtype)

    return total_loss, metrics
