"""Clinical stimulation waveform generators.

All functions return JAX arrays suitable for ``point_source_potential``
electrode_current argument.  Waveforms represent the *electrode current*
in microamps (uA).

Convention: *cathodic* current is **negative** (current flows from tissue
into the electrode), which is the depolarising polarity for extracellular
stimulation.
"""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array

# ---------------------------------------------------------------------------
# Core pulse train generator
# ---------------------------------------------------------------------------


def make_pulse_train(
    amplitude_uA: float,
    pulse_width_ms: float,
    dt_ms: float,
    T_ms: float,
    *,
    frequency_hz: float = 0.0,
    cathodic: bool = True,
    biphasic: bool = False,
    interphase_ms: float = 0.0,
    delay_ms: float = 0.0,
) -> Array:
    """Rectangular pulse train (single pulse when *frequency_hz* is 0).

    Parameters
    ----------
    amplitude_uA : float
        Peak current magnitude (positive value; sign set by *cathodic*).
    pulse_width_ms : float
        Duration of each pulse phase in ms.
    dt_ms : float
        Simulation timestep in ms.
    T_ms : float
        Total waveform duration in ms.
    frequency_hz : float
        Pulse repetition frequency.  ``0`` (default) means a single pulse.
    cathodic : bool
        If True (default), leading/only phase is cathodic (negative).
    biphasic : bool
        If True, each pulse has a charge-balancing second phase.
    interphase_ms : float
        Gap between the two phases of a biphasic pulse (default 0).
    delay_ms : float
        Time before first pulse onset (default 0).

    Returns
    -------
    Array, shape ``(int(T_ms / dt_ms),)``
    """
    T = int(T_ms / dt_ms)
    pw = int(pulse_width_ms / dt_ms)
    ip = int(interphase_ms / dt_ms)
    onset = int(delay_ms / dt_ms)
    sign1 = -1.0 if cathodic else 1.0

    period = int(1000.0 / (frequency_hz * dt_ms)) if frequency_hz > 0.0 else T

    w = jnp.zeros((T,))
    t = onset
    while t < T:
        end1 = min(t + pw, T)
        w = w.at[t:end1].set(sign1 * amplitude_uA)
        if biphasic:
            t2 = t + pw + ip
            end2 = min(t2 + pw, T)
            w = w.at[t2:end2].set(-sign1 * amplitude_uA)
        t += period
    return w


# ---------------------------------------------------------------------------
# Single-pulse convenience wrappers
# ---------------------------------------------------------------------------


def make_monophasic_pulse(
    amplitude_uA: float,
    pulse_width_ms: float,
    dt_ms: float,
    T_ms: float,
    *,
    cathodic: bool = True,
    delay_ms: float = 0.0,
) -> Array:
    """Single-phase rectangular pulse.

    Thin wrapper around :func:`make_pulse_train` with ``frequency_hz=0``
    and ``biphasic=False``.
    """
    return make_pulse_train(
        amplitude_uA,
        pulse_width_ms,
        dt_ms,
        T_ms,
        cathodic=cathodic,
        delay_ms=delay_ms,
    )


def make_biphasic_pulse(
    amplitude_uA: float,
    pulse_width_ms: float,
    dt_ms: float,
    T_ms: float,
    *,
    cathodic_first: bool = True,
    interphase_ms: float = 0.0,
    delay_ms: float = 0.0,
) -> Array:
    """Charge-balanced symmetric biphasic rectangular pulse.

    Thin wrapper around :func:`make_pulse_train` with ``frequency_hz=0``
    and ``biphasic=True``.
    """
    return make_pulse_train(
        amplitude_uA,
        pulse_width_ms,
        dt_ms,
        T_ms,
        cathodic=cathodic_first,
        biphasic=True,
        interphase_ms=interphase_ms,
        delay_ms=delay_ms,
    )


# ---------------------------------------------------------------------------
# Batch grid for vmap sweeps
# ---------------------------------------------------------------------------


def make_biphasic_grid(
    amplitudes_uA: Array,
    pulse_widths_ms: Array,
    dt_ms: float,
    T_ms: float,
    *,
    cathodic_first: bool = True,
    interphase_ms: float = 0.0,
    delay_ms: float = 0.0,
) -> tuple[Array, Array, Array]:
    """Build a (N, T) waveform batch over amplitude x pulse-width grid.

    Returns
    -------
    waveforms : Array, shape ``(N, T)``
        Stacked waveform vectors ready for ``jax.vmap``.
    grid_amps : Array, shape ``(N,)``
        Amplitude for each row.
    grid_pws : Array, shape ``(N,)``
        Pulse width (ms) for each row.
    """
    rows = []
    amps_out = []
    pws_out = []
    for pw in pulse_widths_ms:
        for amp in amplitudes_uA:
            w = make_biphasic_pulse(
                float(amp),
                float(pw),
                dt_ms,
                T_ms,
                cathodic_first=cathodic_first,
                interphase_ms=interphase_ms,
                delay_ms=delay_ms,
            )
            rows.append(w)
            amps_out.append(float(amp))
            pws_out.append(float(pw))
    return (
        jnp.stack(rows),
        jnp.array(amps_out),
        jnp.array(pws_out),
    )
