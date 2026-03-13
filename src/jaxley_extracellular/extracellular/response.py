"""Response feature extraction from voltage traces.

All functions are pure JAX and compatible with ``jax.vmap`` / ``jax.jit``.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jaxtyping import Array


def detect_spike(v_trace: Array, threshold_mV: float = 0.0) -> Array:
    """Return True (scalar bool array) if *v_trace* crosses *threshold_mV*.

    Parameters
    ----------
    v_trace : Array, shape ``(T,)``
        Membrane voltage over time at a single recording site.
    threshold_mV : float
        Voltage threshold for spike detection (default 0 mV).

    Returns
    -------
    Array, shape ``()``, dtype bool
    """
    return jnp.any(v_trace >= threshold_mV)


def spike_latency_steps(v_trace: Array, threshold_mV: float = 0.0) -> Array:
    """Index of first threshold crossing, or ``T`` if no spike.

    Parameters
    ----------
    v_trace : Array, shape ``(T,)``
    threshold_mV : float

    Returns
    -------
    Array, shape ``()``, dtype int32
        Index of first sample >= threshold.  Equals ``len(v_trace)`` when
        there is no spike (sentinel value, avoids branching for vmap).
    """
    above = v_trace >= threshold_mV
    # argmax on a bool array returns the first True index; returns 0 if none.
    first_idx = jnp.argmax(above)
    # Disambiguate: if above[0] is False and argmax returned 0, no spike.
    no_spike = ~jnp.any(above)
    return jnp.where(no_spike, v_trace.shape[0], first_idx)


def spike_latency_ms(
    v_trace: Array,
    dt_ms: float,
    threshold_mV: float = 0.0,
) -> Array:
    """Latency to first spike in milliseconds.

    Parameters
    ----------
    v_trace : Array, shape ``(T,)``
    dt_ms : float
        Simulation timestep in ms.
    threshold_mV : float

    Returns
    -------
    Array, shape ``()``, dtype float32
        Time of first threshold crossing.  Equals ``T * dt_ms`` (total
        trace duration) if no spike is detected, keeping the output
        shape fixed for vmap.
    """
    idx = spike_latency_steps(v_trace, threshold_mV)
    return idx.astype(jnp.float32) * dt_ms


def spike_count(v_trace: Array, threshold_mV: float = 0.0) -> Array:
    """Count upward threshold crossings in *v_trace*.

    Parameters
    ----------
    v_trace : Array, shape ``(T,)``
    threshold_mV : float

    Returns
    -------
    Array, shape ``()``, dtype int32
    """
    above = v_trace >= threshold_mV
    crossings = above[1:] & ~above[:-1]
    return jnp.sum(crossings).astype(jnp.int32)


def mean_isi_ms(
    v_trace: Array,
    dt_ms: float,
    threshold_mV: float = 0.0,
) -> Array:
    """Mean inter-spike interval in milliseconds.

    Parameters
    ----------
    v_trace : Array, shape ``(T,)``
    dt_ms : float
    threshold_mV : float

    Returns
    -------
    Array, shape ``()``, dtype float32
        Sentinel ``-1.0`` if fewer than 2 spikes.
    """
    T = v_trace.shape[0]
    above = v_trace >= threshold_mV
    crossings = above[1:] & ~above[:-1]  # shape (T-1,)
    n_spikes = jnp.sum(crossings)
    # Sort crossing indices to front: True positions get their index, False get T.
    indices = jnp.where(crossings, jnp.arange(T - 1), T)
    sorted_idx = jnp.sort(indices)
    # First crossing = sorted_idx[0], last = sorted_idx[n_spikes - 1]
    first = sorted_idx[0]
    last = sorted_idx[jnp.maximum(n_spikes - 1, 0)]
    span_ms = (last - first).astype(jnp.float32) * dt_ms
    isi = span_ms / jnp.maximum(n_spikes - 1, 1).astype(jnp.float32)
    result: Array = jnp.where(n_spikes >= 2, isi, jnp.float32(-1.0))  # pyright: ignore[reportAssignmentType]
    return result


def firing_rate_hz(
    v_trace: Array,
    dt_ms: float,
    threshold_mV: float = 0.0,
) -> Array:
    """Firing rate in Hz (spike count / trace duration).

    Parameters
    ----------
    v_trace : Array, shape ``(T,)``
    dt_ms : float
    threshold_mV : float

    Returns
    -------
    Array, shape ``()``, dtype float32
        ``0.0`` if no spikes.
    """
    n = spike_count(v_trace, threshold_mV)
    duration_s = v_trace.shape[0] * dt_ms / 1000.0
    return n.astype(jnp.float32) / jnp.float32(duration_s)  # type: ignore[no-any-return]


def extract_response_features(
    v_trace: Array,
    dt_ms: float,
    threshold_mV: float = 0.0,
) -> dict[str, Array]:
    """Compute standard response features for one recording site.

    Parameters
    ----------
    v_trace : Array, shape ``(T,)``
    dt_ms : float
    threshold_mV : float

    Returns
    -------
    dict with keys:
        spiked         -- bool scalar
        latency_ms     -- float scalar (trace duration if no spike)
        vmax           -- float scalar, peak depolarisation
        vmin           -- float scalar, peak hyperpolarisation
        spike_count    -- int32 scalar
        mean_isi_ms    -- float scalar (-1.0 if < 2 spikes)
        firing_rate_hz -- float scalar (0.0 if no spikes)
    """
    return {
        "spiked": detect_spike(v_trace, threshold_mV),
        "latency_ms": spike_latency_ms(v_trace, dt_ms, threshold_mV),
        "vmax": jnp.max(v_trace),
        "vmin": jnp.min(v_trace),
        "spike_count": spike_count(v_trace, threshold_mV),
        "mean_isi_ms": mean_isi_ms(v_trace, dt_ms, threshold_mV),
        "firing_rate_hz": firing_rate_hz(v_trace, dt_ms, threshold_mV),
    }


# ---------------------------------------------------------------------------
# Batched helper: extract features from a (B, T) voltage array
# ---------------------------------------------------------------------------


def extract_response_features_batch(
    v_batch: Array,
    dt_ms: float,
    threshold_mV: float = 0.0,
) -> dict[str, Array]:
    """``vmap``-ped version of :func:`extract_response_features`.

    Parameters
    ----------
    v_batch : Array, shape ``(B, T)``
    dt_ms : float
    threshold_mV : float

    Returns
    -------
    dict with keys mapping to Arrays of shape ``(B,)``.
    """
    extractor = partial(
        extract_response_features,
        dt_ms=dt_ms,
        threshold_mV=threshold_mV,
    )
    return jax.vmap(extractor)(v_batch)
