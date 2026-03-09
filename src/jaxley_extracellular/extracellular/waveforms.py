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
# Elementary pulse shapes
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

    Parameters
    ----------
    amplitude_uA : float
        Peak current magnitude (positive value; sign set by *cathodic*).
    pulse_width_ms : float
        Duration of the pulse in ms.
    dt_ms : float
        Simulation timestep in ms.
    T_ms : float
        Total waveform duration in ms.
    cathodic : bool
        If True (default), current is negative (depolarising).
    delay_ms : float
        Time before pulse onset (default 0).

    Returns
    -------
    Array, shape ``(int(T_ms / dt_ms),)``
    """
    T = int(T_ms / dt_ms)
    sign = -1.0 if cathodic else 1.0
    onset = int(delay_ms / dt_ms)
    offset = onset + int(pulse_width_ms / dt_ms)
    return jnp.zeros((T,)).at[onset:offset].set(sign * amplitude_uA)


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

    Parameters
    ----------
    amplitude_uA : float
        Peak current magnitude (positive value).
    pulse_width_ms : float
        Duration of *each* phase in ms.
    dt_ms : float
        Simulation timestep in ms.
    T_ms : float
        Total waveform duration in ms.
    cathodic_first : bool
        If True (default), leading phase is cathodic (negative).
    interphase_ms : float
        Gap between the two phases (default 0).
    delay_ms : float
        Time before first phase onset (default 0).

    Returns
    -------
    Array, shape ``(int(T_ms / dt_ms),)``
    """
    T = int(T_ms / dt_ms)
    pw = int(pulse_width_ms / dt_ms)
    ip = int(interphase_ms / dt_ms)
    onset = int(delay_ms / dt_ms)

    sign1 = -1.0 if cathodic_first else 1.0
    w = jnp.zeros((T,))
    w = w.at[onset : onset + pw].set(sign1 * amplitude_uA)
    w = w.at[onset + pw + ip : onset + 2 * pw + ip].set(-sign1 * amplitude_uA)
    return w


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
