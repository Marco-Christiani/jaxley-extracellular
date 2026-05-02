"""Shared helpers for ECS cross-solver parity scripts (NEURON vs Jaxley).

This module is cell-agnostic: it only depends on numpy and plotly, so both
the NEURON-side and Jaxley-side scripts can import it without dragging in
the other simulator. Cell-specific setup (Jaxley Branch / Cell construction,
NEURON template loading, per-cell channel insertion) lives in the sibling
modules `hh.py` and `bbp_pyr.py`.

What's in here:
    - Stimulus waveform generator (rectangular pulse in uA).
    - Analytical point-source extracellular potential at arbitrary
      compartment positions.
    - Parity metrics (per-segment RMSE / MAE / Pearson r, time-grid
      interpolation to align solver outputs).
    - A plotly figure builder with a common layout: four panels showing
      matched voltage traces, per-segment RMSE, and the phi_e snapshot at
      stimulation peak.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Stimulus and field primitives

@dataclass(frozen=True)
class PulseSpec:
    """A single-pulse extracellular current waveform in uA.

    amp_uA < 0 is cathodic. Width / delay in ms. Time axis is generated
    from these plus the overall (dt_ms, t_max_ms) grid on the caller side.
    """
    amp_uA: float
    width_ms: float
    delay_ms: float


def time_grid(dt_ms: float, t_max_ms: float) -> np.ndarray:
    """Uniform time grid shared by both solvers. Inclusive of the endpoint."""
    n = round(t_max_ms / dt_ms) + 1
    return np.arange(n) * dt_ms


def waveform_uA(t: np.ndarray, pulse: PulseSpec) -> np.ndarray:
    """Rectangular pulse waveform in uA, sampled on t."""
    on = (t >= pulse.delay_ms) & (t < pulse.delay_ms + pulse.width_ms)
    w = np.zeros_like(t, dtype=float)
    w[on] = pulse.amp_uA
    return w


def phi_e_per_compartment(
    comp_xyz_um: np.ndarray,
    wave_uA: np.ndarray,
    electrode_xyz_um: Sequence[float],
    sigma_S_m: float,
    min_distance_um: float = 1.0,
) -> np.ndarray:
    """Analytical point-source extracellular potential at each compartment.

    Matches the project's `point_source_potential` convention:
        phi_e [mV] = I [uA] * 1e3 / (4 pi sigma [S/m] r [um])

    Args:
        comp_xyz_um: (Ncomp, 3) compartment centre coordinates in um.
        wave_uA: (T,) electrode current trace in uA.
        electrode_xyz_um: length-3 tuple/array of electrode position in um.
        sigma_S_m: extracellular conductivity.
        min_distance_um: floor to avoid singular division if a compartment
            centre happens to sit exactly on top of the electrode.

    Returns:
        phi_e: (Ncomp, T) extracellular potential in mV.
    """
    elec = np.asarray(electrode_xyz_um, dtype=float)
    diff = comp_xyz_um - elec[None, :]
    r = np.sqrt((diff ** 2).sum(axis=1))       # (Ncomp,) in um
    r = np.maximum(r, min_distance_um)
    prefactor = 1e3 / (4.0 * np.pi * sigma_S_m * r)  # (Ncomp,) in mV/uA
    return prefactor[:, None] * wave_uA[None, :]     # (Ncomp, T) in mV


# Parity metrics

@dataclass(frozen=True)
class ParityMetrics:
    """Per-segment parity metrics computed on a common time grid.

    Both voltage arrays must be (Nseg, T). The metrics are computed after
    linearly interpolating the "other" array onto the "reference" t grid,
    which is how we handle the case where the two solvers' output
    timestamps drift by rounding.
    """
    rmse_per_seg: np.ndarray   # (Nseg,)
    mae_per_seg: np.ndarray    # (Nseg,)
    r_per_seg: np.ndarray      # (Nseg,)
    resid: np.ndarray          # (Nseg, T) signed residual
    t_ref: np.ndarray          # reference time axis (Nseg-independent)
    v_ref: np.ndarray          # reference voltage (Nseg, T), not interpolated
    v_other_on_ref: np.ndarray  # the other solver on t_ref (Nseg, T)


def compute_metrics(
    t_ref: np.ndarray, v_ref: np.ndarray,
    t_other: np.ndarray, v_other: np.ndarray,
) -> ParityMetrics:
    """Align v_other onto t_ref's time grid, then compute per-segment metrics.

    Reference conventionally = NEURON (higher-precision CPU), other = Jaxley.
    """
    if v_ref.shape[0] != v_other.shape[0]:
        raise ValueError(
            f"segment count mismatch: v_ref {v_ref.shape} vs v_other {v_other.shape}"
        )
    n_seg = v_ref.shape[0]
    v_other_on_ref = np.stack(
        [np.interp(t_ref, t_other, v_other[i]) for i in range(n_seg)], axis=0,
    )
    resid = v_other_on_ref - v_ref
    rmse = np.sqrt(np.mean(resid ** 2, axis=1))
    mae = np.mean(np.abs(resid), axis=1)
    r = np.array([
        float(np.corrcoef(v_other_on_ref[i], v_ref[i])[0, 1])
        for i in range(n_seg)
    ])
    return ParityMetrics(
        rmse_per_seg=rmse, mae_per_seg=mae, r_per_seg=r,
        resid=resid, t_ref=t_ref, v_ref=v_ref,
        v_other_on_ref=v_other_on_ref,
    )


def print_metrics_summary(m: ParityMetrics, *, label: str = "parity") -> None:
    print(f"[{label}] segments: {m.rmse_per_seg.shape[0]}")
    print(f"[{label}] RMSE per-seg: min {m.rmse_per_seg.min():.3f}, "
          f"median {np.median(m.rmse_per_seg):.3f}, "
          f"max {m.rmse_per_seg.max():.3f} mV")
    print(f"[{label}] MAE  per-seg: min {m.mae_per_seg.min():.3f}, "
          f"median {np.median(m.mae_per_seg):.3f}, "
          f"max {m.mae_per_seg.max():.3f} mV")
    print(f"[{label}] Pearson r: min {m.r_per_seg.min():.4f}, "
          f"median {np.median(m.r_per_seg):.4f}")


# Figure

def plot_ecs_parity_sites(
    *,
    metrics: ParityMetrics,
    phi_sites: np.ndarray,
    site_names: Sequence[str],
    out_path: Path,
    title_main: str,
    title_subtitle: str | None = None,
    primary_site: str = "soma",
    secondary_site: str = "apical",
    scale: float = 2.0,
) -> None:
    """Four-panel figure for named-site parity (e.g. BBP morphology).

    Designed for a handful of anatomically-named recording sites rather
    than a long segment array. Panels:
        1 (top-left):  voltage at `primary_site` (NEURON vs Jaxley).
        2 (top-right): per-site RMSE bar chart.
        3 (bot-left):  voltage at `secondary_site`.
        4 (bot-right): phi_e at each site over time.

    phi_sites has shape (Nsites, T); site ordering must match metrics.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    t = metrics.t_ref
    v_n = metrics.v_ref
    v_j = metrics.v_other_on_ref
    rmse = metrics.rmse_per_seg
    names = list(site_names)

    i_primary = names.index(primary_site)
    i_secondary = names.index(secondary_site) if secondary_site in names else -1

    fig = make_subplots(
        rows=2, cols=2,
        vertical_spacing=0.18, horizontal_spacing=0.12,
        subplot_titles=(
            f"Voltage at {primary_site}: NEURON vs Jaxley",
            "Per-site RMSE",
            f"Voltage at {secondary_site}" if i_secondary >= 0 else "Voltage (secondary)",
            "phi_e at each site over time",
        ),
    )

    fig.add_trace(go.Scatter(x=t, y=v_n[i_primary], mode="lines",
                             line=dict(color="black", width=1.8),
                             name="NEURON"),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=v_j[i_primary], mode="lines",
                             line=dict(color="#d62728", width=1.6, dash="dash"),
                             name="Jaxley"),
                  row=1, col=1)
    fig.update_xaxes(title_text="time (ms)", row=1, col=1)
    fig.update_yaxes(title_text="V (mV)", row=1, col=1)

    fig.add_trace(go.Bar(x=names, y=rmse,
                         marker_color="#1f77b4", showlegend=False),
                  row=1, col=2)
    fig.update_yaxes(title_text="RMSE (mV)", row=1, col=2)

    if i_secondary >= 0:
        fig.add_trace(go.Scatter(x=t, y=v_n[i_secondary], mode="lines",
                                 line=dict(color="black", width=1.8),
                                 showlegend=False),
                      row=2, col=1)
        fig.add_trace(go.Scatter(x=t, y=v_j[i_secondary], mode="lines",
                                 line=dict(color="#d62728", width=1.6, dash="dash"),
                                 showlegend=False),
                      row=2, col=1)
        fig.update_xaxes(title_text="time (ms)", row=2, col=1)
        fig.update_yaxes(title_text="V (mV)", row=2, col=1)

    # A stable palette for the four most common BBP section names.
    palette = {"soma": "#1f77b4", "apical": "#d62728",
               "basal": "#2ca02c", "axon": "#ff7f0e"}
    for i, name in enumerate(names):
        fig.add_trace(go.Scatter(x=t, y=phi_sites[i], mode="lines",
                                 line=dict(color=palette.get(name, "#888"), width=1.6),
                                 name=name, legendgroup="phi"),
                      row=2, col=2)
    fig.update_xaxes(title_text="time (ms)", row=2, col=2)
    fig.update_yaxes(title_text="phi_e (mV)", row=2, col=2)

    title = title_main
    if title_subtitle:
        title = f"{title_main}<br><sub>{title_subtitle}</sub>"

    fig.update_layout(
        width=1100, height=680, title=title,
        legend=dict(orientation="h", yanchor="top", y=-0.12,
                    xanchor="center", x=0.5),
        margin=dict(t=100, b=90, l=70, r=30),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(out_path), scale=scale)
    print(f"Saved {out_path}")


def plot_ecs_parity(
    *,
    metrics: ParityMetrics,
    phi_e: np.ndarray,
    seg_positions: np.ndarray,
    seg_position_label: str,
    pulse: PulseSpec,
    electrode_xyz_um: Sequence[float],
    sigma_S_m: float,
    dt_ms: float,
    out_path: Path,
    title_main: str,
    title_subtitle: str | None = None,
    scale: float = 2.0,
) -> None:
    """Build the shared four-panel parity figure and save as PNG.

    Panels:
        1 (top-left):  voltage at the mid-indexed segment, NEURON vs Jaxley.
        2 (top-right): per-segment RMSE bar chart.
        3 (bot-left):  voltage at segment 0.
        4 (bot-right): phi_e snapshot at stimulus peak across segments.

    seg_positions: 1D array of length Nseg, values used for the x-axis of
        panels 2 and 4. For a straight cable this is arc length (um); for
        a morphology this is compartment index.
    seg_position_label: x-axis label for panels 2 and 4.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    t = metrics.t_ref
    v_n = metrics.v_ref
    v_j = metrics.v_other_on_ref
    rmse = metrics.rmse_per_seg
    n_seg = v_n.shape[0]
    midseg = n_seg // 2

    fig = make_subplots(
        rows=2, cols=2,
        vertical_spacing=0.18, horizontal_spacing=0.12,
        subplot_titles=(
            f"Voltage at segment {midseg} of {n_seg}",
            "Per-segment RMSE  |V_jaxley - V_neuron|",
            "Voltage at segment 0",
            "phi_e snapshot (stim peak)",
        ),
    )

    fig.add_trace(go.Scatter(x=t, y=v_n[midseg], mode="lines",
                             line=dict(color="black", width=1.8),
                             name="NEURON"),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=v_j[midseg], mode="lines",
                             line=dict(color="#d62728", width=1.6, dash="dash"),
                             name="Jaxley"),
                  row=1, col=1)
    fig.update_xaxes(title_text="time (ms)", row=1, col=1)
    fig.update_yaxes(title_text="V (mV)", row=1, col=1)

    fig.add_trace(go.Bar(x=seg_positions, y=rmse,
                         marker_color="#1f77b4", showlegend=False),
                  row=1, col=2)
    fig.update_xaxes(title_text=seg_position_label, row=1, col=2)
    fig.update_yaxes(title_text="RMSE (mV)", row=1, col=2)

    fig.add_trace(go.Scatter(x=t, y=v_n[0], mode="lines",
                             line=dict(color="black", width=1.8),
                             showlegend=False),
                  row=2, col=1)
    fig.add_trace(go.Scatter(x=t, y=v_j[0], mode="lines",
                             line=dict(color="#d62728", width=1.6, dash="dash"),
                             showlegend=False),
                  row=2, col=1)
    fig.update_xaxes(title_text="time (ms)", row=2, col=1)
    fig.update_yaxes(title_text="V (mV)", row=2, col=1)

    peak_idx = int((pulse.delay_ms + 0.5 * pulse.width_ms) / dt_ms)
    peak_idx = max(0, min(peak_idx, phi_e.shape[1] - 1))
    fig.add_trace(go.Scatter(x=seg_positions, y=phi_e[:, peak_idx],
                             mode="lines+markers",
                             line=dict(color="#2ca02c", width=2),
                             showlegend=False),
                  row=2, col=2)
    fig.update_xaxes(title_text=seg_position_label, row=2, col=2)
    fig.update_yaxes(title_text="phi_e (mV)", row=2, col=2)

    title = title_main
    if title_subtitle:
        title = f"{title_main}<br><sub>{title_subtitle}</sub>"

    fig.update_layout(
        width=1100, height=680,
        title=title,
        legend=dict(orientation="h", yanchor="top", y=-0.12,
                    xanchor="center", x=0.5),
        margin=dict(t=100, b=90, l=70, r=30),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(out_path), scale=scale)
    print(f"Saved {out_path}")
