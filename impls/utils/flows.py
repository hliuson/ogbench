from typing import Callable, Dict, Optional, Tuple, Union

import jax
import jax.numpy as jnp


def _logit_normal(rng: jax.Array, shape: Tuple[int, ...], mean: float, std: float) -> jax.Array:
    return jax.nn.sigmoid(jax.random.normal(rng, shape) * std + mean)


def _sample_imf_times(
    rng: jax.Array,
    batch_size: int,
    r_equals_t_prob: float = 0.0,
    logit_normal: bool = True,
    logit_mean: float = -0.4,
    logit_std: float = 1.0,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """Sample (r, t) with r <= t and optional diagonal mass (r == t).

    This matches the official iMF schedule: logit-normal times and a fixed
    proportion of r=t ("flow-matching" samples).

    Args:
        r_equals_t_prob: Proportion of the batch with r == t (data_proportion in iMF).
        logit_normal: If True, use logit-normal distribution (official iMF).
                      If False, use uniform distribution.
        logit_mean: Mean of the normal distribution before sigmoid.
        logit_std: Std of the normal distribution before sigmoid.
    """
    rng, t_rng, r_rng = jax.random.split(rng, 3)
    if logit_normal:
        t = _logit_normal(t_rng, (batch_size, 1), logit_mean, logit_std)
        r = _logit_normal(r_rng, (batch_size, 1), logit_mean, logit_std)
    else:
        t = jax.random.uniform(t_rng, (batch_size, 1))
        r = jax.random.uniform(r_rng, (batch_size, 1))

    t_hi = jnp.maximum(t, r)
    r_lo = jnp.minimum(t, r)
    t, r = t_hi, r_lo

    if r_equals_t_prob > 0:
        data_size = int(batch_size * r_equals_t_prob)
        fm_mask = (jnp.arange(batch_size) < data_size).reshape(batch_size, 1)
        r = jnp.where(fm_mask, t, r)
    else:
        fm_mask = jnp.zeros((batch_size, 1), dtype=bool)

    return r, t, fm_mask


def _adaptive_loss_weight(loss: jax.Array, eps: float = 0.01, p: float = 1.0) -> jax.Array:
    """Adaptive loss weighting (official iMF): loss / (loss + eps)^p."""
    adp_wt = (loss + eps) ** p
    return loss / jax.lax.stop_gradient(adp_wt)


def _sample_cfg_scale(
    rng: jax.Array,
    batch_size: int,
    w_min: float = 1.0,
    w_max: float = 1.0,
    cfg_beta: float = 1.0,
) -> jax.Array:
    """Sample CFG scale w with the official iMF power distribution.

    When w_min == 1.0, w is sampled in [1, 1 + s_max] with s_max = w_max - 1.
    For w_min != 1.0, we shift the distribution to [w_min, w_max].
    """
    if w_max <= w_min:
        return jnp.full((batch_size, 1), w_min)

    u = jax.random.uniform(rng, (batch_size, 1), minval=0.0, maxval=1.0, dtype=jnp.float32)
    s_max = jnp.asarray(w_max - w_min, jnp.float32)
    b = jnp.asarray(cfg_beta, jnp.float32)

    if cfg_beta == 1.0:
        s = jnp.exp(u * jnp.log1p(s_max))
    else:
        log_base = (1.0 - b) * jnp.log1p(s_max)
        log_inner = jnp.log1p(u * jnp.expm1(log_base))
        s = jnp.exp(log_inner / (1.0 - b))

    return s + (w_min - 1.0)


def _sample_cfg_interval(
    rng: jax.Array,
    batch_size: int,
    fm_mask: Optional[jax.Array] = None,
) -> Tuple[jax.Array, jax.Array]:
    """Sample CFG interval [t_min, t_max] (official iMF)."""
    rng_start, rng_end = jax.random.split(rng)
    t_min = jax.random.uniform(
        rng_start, (batch_size, 1), minval=0.0, maxval=0.5, dtype=jnp.float32
    )
    t_max = jax.random.uniform(
        rng_end, (batch_size, 1), minval=0.5, maxval=1.0, dtype=jnp.float32
    )

    if fm_mask is not None:
        t_min = jnp.where(fm_mask, 0.0, t_min)
        t_max = jnp.where(fm_mask, 1.0, t_max)

    return t_min, t_max


def _ensure_like(x: Union[jax.Array, float, int, None], ref: jax.Array) -> jax.Array:
    if x is None:
        return jnp.ones_like(ref)
    if isinstance(x, (float, int)):
        return jnp.full_like(ref, float(x))
    x = jnp.asarray(x, dtype=ref.dtype)
    if x.shape == ref.shape:
        return x
    if x.ndim < ref.ndim:
        x = x.reshape((1,) * (ref.ndim - x.ndim) + x.shape)
    return jnp.broadcast_to(x, ref.shape)


def _pack_times(t: jax.Array, r: jax.Array, w: jax.Array, t_min: jax.Array, t_max: jax.Array) -> jax.Array:
    """Pack times as [t, h=t-r, w, t_min, t_max] (official iMF conditioning)."""
    h = t - r
    return jnp.concatenate([t, h, w, t_min, t_max], axis=-1)


def _pack_v_times(t: jax.Array, w: jax.Array) -> jax.Array:
    """Pack times for v prediction: [t, 0, w, 0, 1]."""
    zeros = jnp.zeros_like(t)
    ones = jnp.ones_like(t)
    return jnp.concatenate([t, zeros, w, zeros, ones], axis=-1)


def _split_uv(out):
    if isinstance(out, tuple):
        return out[0], out[1]
    return out, None


def _apply_drop(cond_out, uncond_out, drop_mask):
    if isinstance(cond_out, tuple):
        return tuple(jnp.where(drop_mask, u_uncond, u_cond) for u_cond, u_uncond in zip(cond_out, uncond_out))
    return jnp.where(drop_mask, uncond_out, cond_out)


def _sample_drop_mask(rng: jax.Array, batch_size: int, drop_prob: float) -> jax.Array:
    if drop_prob <= 0.0:
        return jnp.zeros((batch_size, 1), dtype=bool)
    rand_mask = jax.random.uniform(rng, shape=(batch_size,)) < drop_prob
    num_drop = jnp.sum(rand_mask).astype(jnp.int32)
    return (jnp.arange(batch_size) < num_drop).reshape(batch_size, 1)


def imf_loss(
    rng: jax.Array,
    x: jax.Array,
    vector_field_fn: Callable[[jax.Array, jax.Array], jax.Array],
    r_equals_t_prob: float = 0.5,
    w: Union[jax.Array, float, None] = None,
    logit_normal: bool = True,
    logit_mean: float = -0.4,
    logit_std: float = 1.0,
    adaptive_loss: bool = True,
    adaptive_loss_eps: float = 0.01,
    adaptive_loss_p: float = 1.0,
) -> Tuple[jax.Array, Dict[str, jax.Array]]:
    """iMF loss without classifier-free guidance (CFG).

    This matches the official iMF objective with w=1 (no guidance).
    The vector_field_fn can optionally return (u, v). If it only returns u,
    the auxiliary v-head loss is omitted.

    Args:
        rng: PRNG key.
        x: Target samples (batch, dim).
        vector_field_fn: Callable that maps (z, times=[t,h,w,t_min,t_max]) -> u (or (u, v)).
        r_equals_t_prob: Proportion of r==t samples (data_proportion).
        w: Optional guidance scale to include as a time input (default 1.0).
        logit_normal: If True, use logit-normal distribution (official iMF).
        logit_mean: Mean of the normal distribution before sigmoid.
        logit_std: Std of the normal distribution before sigmoid.
        adaptive_loss: If True, use adaptive loss weighting (official iMF).
        adaptive_loss_eps: Epsilon for adaptive loss weighting.
        adaptive_loss_p: Power for adaptive loss weighting.
    """
    batch_size, dim = x.shape
    rng, noise_rng, time_rng = jax.random.split(rng, 3)

    e = jax.random.normal(noise_rng, (batch_size, dim), dtype=x.dtype)
    r, t, _ = _sample_imf_times(
        time_rng,
        batch_size,
        r_equals_t_prob=r_equals_t_prob,
        logit_normal=logit_normal,
        logit_mean=logit_mean,
        logit_std=logit_std,
    )

    w = _ensure_like(w, t)
    t_min = jnp.zeros_like(t)
    t_max = jnp.ones_like(t)

    z_t = (1 - t) * x + t * e
    v_t = e - x

    # Predicted v used as JVP tangent (official iMF)
    out_tt = vector_field_fn(z_t, _pack_v_times(t, w))
    _, v_head = _split_uv(out_tt)
    v_c = v_head if v_head is not None else out_tt
    has_v = v_head is not None

    def u_fn(z_in, t_in, r_in):
        times = _pack_times(t_in, r_in, w, t_min, t_max)
        out = vector_field_fn(z_in, times)
        if has_v:
            u_out, v_out = _split_uv(out)
            return u_out, v_out
        return out

    dtdt = jnp.ones_like(t)
    dtdr = jnp.zeros_like(t)
    if has_v:
        u, dudt, v_pred = jax.jvp(u_fn, (z_t, t, r), (v_c, dtdt, dtdr), has_aux=True)
    else:
        u, dudt = jax.jvp(u_fn, (z_t, t, r), (v_c, dtdt, dtdr))
        v_pred = None

    V = u + (t - r) * jax.lax.stop_gradient(dudt)
    v_g = jax.lax.stop_gradient(v_t)

    loss_u = jnp.sum((V - v_g) ** 2, axis=-1)
    if adaptive_loss:
        loss_u = _adaptive_loss_weight(loss_u, eps=adaptive_loss_eps, p=adaptive_loss_p)

    loss = loss_u
    loss_v = None
    if v_pred is not None:
        loss_v = jnp.sum((v_pred - v_g) ** 2, axis=-1)
        if adaptive_loss:
            loss_v = _adaptive_loss_weight(loss_v, eps=adaptive_loss_eps, p=adaptive_loss_p)
        loss = loss + loss_v

    loss = jnp.mean(loss)

    info = {
        'loss': loss,
        'loss_u': jnp.mean((V - v_g) ** 2),
    }
    if v_pred is not None:
        info['loss_v'] = jnp.mean((v_pred - v_g) ** 2)

    return loss, info


def imf_cfg_loss(
    rng: jax.Array,
    x: jax.Array,
    cond_fn: Callable[[jax.Array, jax.Array], jax.Array],
    uncond_fn: Callable[[jax.Array, jax.Array], jax.Array],
    r_equals_t_prob: float = 0.5,
    w_min: float = 1.0,
    w_max: float = 1.0,
    w_power: float = 1.0,
    cfg_beta: Optional[float] = None,
    class_dropout_prob: float = 0.1,
    logit_normal: bool = True,
    logit_mean: float = -0.4,
    logit_std: float = 1.0,
    adaptive_loss: bool = True,
    adaptive_loss_eps: float = 0.01,
    adaptive_loss_p: float = 1.0,
) -> Tuple[jax.Array, Dict[str, jax.Array]]:
    """iMF loss with classifier-free guidance (CFG).

    Matches the official iMF implementation. The network can optionally return
    (u, v); otherwise, the auxiliary v-head loss is omitted.

    Key points:
    - Uses predicted v_c as JVP tangent (not v_g)
    - Computes V = u + (t-r)*du/dt and regresses to guided target v_g
    - Uses logit-normal time sampling and adaptive loss weighting
    - Includes CFG interval sampling and class dropout
    """
    batch_size, dim = x.shape
    rng, noise_rng, time_rng, w_rng, interval_rng, drop_rng = jax.random.split(rng, 6)

    e = jax.random.normal(noise_rng, (batch_size, dim), dtype=x.dtype)
    r, t, fm_mask = _sample_imf_times(
        time_rng,
        batch_size,
        r_equals_t_prob=r_equals_t_prob,
        logit_normal=logit_normal,
        logit_mean=logit_mean,
        logit_std=logit_std,
    )

    z_t = (1 - t) * x + t * e
    v_t = e - x

    t_min, t_max = _sample_cfg_interval(interval_rng, batch_size, fm_mask=fm_mask)

    if cfg_beta is None:
        cfg_beta = w_power
    w = _sample_cfg_scale(w_rng, batch_size, w_min=w_min, w_max=w_max, cfg_beta=cfg_beta).astype(x.dtype)

    # Compute guided velocity target v_g and conditioned velocity v_c
    out_c = cond_fn(z_t, _pack_v_times(t, w))
    out_u = uncond_fn(z_t, _pack_v_times(t, jnp.ones_like(w)))
    _, v_c0 = _split_uv(out_c)
    _, v_u = _split_uv(out_u)
    v_c0 = v_c0 if v_c0 is not None else out_c
    v_u = v_u if v_u is not None else out_u

    v_g_fm = v_t + (1.0 - 1.0 / w) * (v_c0 - v_u)

    w_interval = jnp.where((t >= t_min) & (t <= t_max), w, 1.0)
    out_c_int = cond_fn(z_t, _pack_v_times(t, w_interval))
    _, v_c = _split_uv(out_c_int)
    v_c = v_c if v_c is not None else out_c_int

    v_g = v_t + (1.0 - 1.0 / w_interval) * (v_c - v_u)
    v_g = jnp.where(fm_mask, v_g_fm, v_g)

    drop_mask = _sample_drop_mask(drop_rng, batch_size, class_dropout_prob)
    if class_dropout_prob > 0.0:
        v_g = jnp.where(drop_mask, v_t, v_g)

    has_v = isinstance(out_c_int, tuple)

    def u_fn(z_in, t_in, r_in):
        times = _pack_times(t_in, r_in, w, t_min, t_max)
        if class_dropout_prob > 0.0:
            out_cond = cond_fn(z_in, times)
            out_uncond = uncond_fn(z_in, times)
            out = _apply_drop(out_cond, out_uncond, drop_mask)
        else:
            out = cond_fn(z_in, times)
        if has_v:
            u_out, v_out = _split_uv(out)
            return u_out, v_out
        return out

    dtdt = jnp.ones_like(t)
    dtdr = jnp.zeros_like(t)
    if has_v:
        u, dudt, v_pred = jax.jvp(u_fn, (z_t, t, r), (v_c, dtdt, dtdr), has_aux=True)
    else:
        u, dudt = jax.jvp(u_fn, (z_t, t, r), (v_c, dtdt, dtdr))
        v_pred = None

    V = u + (t - r) * jax.lax.stop_gradient(dudt)
    v_g = jax.lax.stop_gradient(v_g)

    loss_u = jnp.sum((V - v_g) ** 2, axis=-1)
    if adaptive_loss:
        loss_u = _adaptive_loss_weight(loss_u, eps=adaptive_loss_eps, p=adaptive_loss_p)

    loss = loss_u
    if v_pred is not None:
        loss_v = jnp.sum((v_pred - v_g) ** 2, axis=-1)
        if adaptive_loss:
            loss_v = _adaptive_loss_weight(loss_v, eps=adaptive_loss_eps, p=adaptive_loss_p)
        loss = loss + loss_v
    else:
        loss_v = None

    loss = jnp.mean(loss)

    info = {
        'loss': loss,
        'loss_u': jnp.mean((V - v_g) ** 2),
    }
    if v_pred is not None:
        info['loss_v'] = jnp.mean((v_pred - v_g) ** 2)

    return loss, info


def imf_one_shot_sample(
    rng: jax.Array,
    sample_shape: Tuple[int, ...],
    vector_field_fn: Callable[[jax.Array, jax.Array], jax.Array],
    dtype: jnp.dtype = jnp.float32,
    w: Union[jax.Array, float, None] = None,
    t_min: Union[jax.Array, float, None] = None,
    t_max: Union[jax.Array, float, None] = None,
) -> jax.Array:
    """One-shot iMF sampling: z = e - f(e, r=0, t=1, w)."""
    z = jax.random.normal(rng, sample_shape, dtype=dtype)
    r = jnp.zeros((*sample_shape[:-1], 1), dtype=z.dtype)
    t = jnp.ones((*sample_shape[:-1], 1), dtype=z.dtype)

    w = _ensure_like(w, t)

    if t_min is None:
        t_min = jnp.zeros_like(t)
    else:
        t_min = _ensure_like(t_min, t)
    if t_max is None:
        t_max = jnp.ones_like(t)
    else:
        t_max = _ensure_like(t_max, t)

    times = _pack_times(t, r, w, t_min, t_max)
    out = vector_field_fn(z, times)
    if isinstance(out, tuple):
        out = out[0]
    return z - out
