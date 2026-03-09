"""Response feature extraction from voltage traces.

All functions are pure JAX and compatible with ``jax.vmap`` / ``jax.jit``.
"""

from __future__ import annotations

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

    Returns ``T_ms`` (total trace duration) if no spike is detected,
    keeping the output shape fixed for vmap.
    """
    idx = spike_latency_steps(v_trace, threshold_mV)
    return idx.astype(jnp.float32) * dt_ms


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
        spiked      -- bool scalar
        latency_ms  -- float scalar (trace duration if no spike)
        vmax        -- float scalar, peak depolarisation
        vmin        -- float scalar, peak hyperpolarisation
    """
    return {
        "spiked": detect_spike(v_trace, threshold_mV),
        "latency_ms": spike_latency_ms(v_trace, dt_ms, threshold_mV),
        "vmax": jnp.max(v_trace),
        "vmin": jnp.min(v_trace),
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
    return jax.vmap(
        lambda v: extract_response_features(v, dt_ms, threshold_mV)
    )(v_batch)
