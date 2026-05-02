"""Generate the paper figures from locked result artifacts.

Consumes the curated artifact set under ``results/paper_package/data/``
(pinned Apr 27--29) rather than top-level ``results/*.npz``, which can be
overwritten by smoke tests or re-runs. Outputs land in ``paper/figures/``
next to ``paper.tex`` so the figure pipeline is self-contained under
``paper/``.

Run from project root:

    python -m paper.make_figures              # all figures
    python -m paper.make_figures --which throughput
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from scripts.parity_common import waveform_metrics
from scripts.parity_ecs import common as ecs_common

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "results" / "paper_package" / "data"
OUT_DIR = PROJECT_ROOT / "paper" / "figures"

# Centralised strings + protocol constants. Functions reference CONFIG
# rather than inline literals so a rename or hardware change is a single
# edit. Numbers that come from data (dt, ncomps, n_devices, RMSE etc.)
# are still read from the npz at runtime.
CONFIG: dict[str, object] = {
    "cell_label": "BBP L2/3 Pyr",
    "morph_short": "cADpyr229",
    "neuron_label": "NEURON (CPU)",
    "jaxley_label": "Jaxley (TPU)",
    # Threshold (nA) for sub-vs-supra labelling in the intracellular parity
    # figure. Matches the protocol amplitudes 0.1 (sub) and 0.5 (supra).
    "subthresh_amp_nA": 0.2,
    # Step-current protocol window (ms), shaded in intracellular parity.
    "stim_window_ms": (50.0, 150.0),
}


def _n_devices(npz: object, default: int = 1) -> int:
    """Read ``n_devices`` from an npz, defaulting if the field is absent.

    Older single-chip throughput runs predate the field; newer multi-chip
    runs have it. Returning a default keeps figure code uniform.
    """
    return int(npz["n_devices"]) if "n_devices" in npz.files else default


def _human_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f} s"
    if seconds < 3600:
        return f"{seconds / 60:.1f} min"
    if seconds < 86400:
        return f"{seconds / 3600:.2f} h"
    return f"{seconds / 86400:.2f} d"


def make_bbp_intracellular_parity(
    out_path: Path,
    *,
    jaxley_npz: Path = DATA_DIR / "parity_bbp_pyr_jaxley.npz",
    neuron_npz: Path = DATA_DIR / "parity_bbp_pyr_neuron.npz",
    scale: float = 2.0,
) -> None:
    jp = np.load(jaxley_npz)
    nn = np.load(neuron_npz)
    t = jp["t"]
    amps = tuple(float(x) for x in jp["amps"].tolist())
    dt_ms = float(t[1] - t[0])  # dt is implicit in the time grid for parity npzs
    sub_amp = float(CONFIG["subthresh_amp_nA"])
    stim_t0, stim_t1 = (float(x) for x in CONFIG["stim_window_ms"])

    fig = make_subplots(
        rows=len(amps),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.09,
        subplot_titles=tuple(
            f"Step current amplitude {amp} nA"
            + ("  (subthreshold)" if amp < sub_amp else "  (suprathreshold)")
            for amp in amps
        ),
    )

    for row, amp in enumerate(amps, start=1):
        key = f"v_{amp}"
        vn = nn[key]
        vj = jp[key]
        m = waveform_metrics(t, vn, t, vj)

        fig.add_vrect(
            x0=stim_t0,
            x1=stim_t1,
            fillcolor="rgba(200,200,200,0.25)",
            line_width=0,
            layer="below",
            row=row,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=t,
                y=vn,
                mode="lines",
                name=CONFIG["neuron_label"],
                line=dict(color="black", width=1.6),
                legendgroup="neuron",
                showlegend=(row == 1),
            ),
            row=row,
            col=1,
        )
        color = "#d62728" if amp >= sub_amp else "#1f77b4"
        fig.add_trace(
            go.Scatter(
                x=t,
                y=vj,
                mode="lines",
                name=CONFIG["jaxley_label"],
                line=dict(color=color, width=1.6, dash="dash"),
                legendgroup="jaxley",
                showlegend=(row == 1),
            ),
            row=row,
            col=1,
        )
        metrics_text = (
            f"RMSE = {m['rmse_mV']:.2f} mV<br>"
            f"MAE = {m['mae_mV']:.2f} mV<br>"
            f"max|err| = {m['max_abs_mV']:.2f} mV<br>"
            f"r = {m['pearson_r']:.4f}"
        )
        ax_suffix = "" if row == 1 else str(row)
        fig.add_annotation(
            xref=f"x{ax_suffix} domain",
            yref=f"y{ax_suffix} domain",
            x=0.99,
            y=0.98,
            xanchor="right",
            yanchor="top",
            text=metrics_text,
            showarrow=False,
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#888",
            borderwidth=1,
            borderpad=4,
            font=dict(size=11, family="monospace"),
            row=row,
            col=1,
        )
        fig.update_yaxes(title_text="V_soma  [mV]", row=row, col=1)

    fig.update_xaxes(title_text="time  [ms]", row=len(amps), col=1)
    fig.update_layout(
        height=620,
        width=900,
        showlegend=True,
        legend=dict(orientation="h", yanchor="top", y=-0.12, xanchor="center", x=0.5),
        title=(
            f"{CONFIG['cell_label']} parity: "
            f"{CONFIG['jaxley_label']} vs {CONFIG['neuron_label']}"
        ),
        margin=dict(t=70, b=90, l=70, r=30),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(out_path), scale=scale)


def make_bbp_ecs_parity(
    out_path: Path,
    *,
    jaxley_npz: Path = DATA_DIR / "parity_bbp_ecs_jaxley.npz",
    neuron_npz: Path = DATA_DIR / "parity_bbp_ecs_neuron.npz",
    scale: float = 2.0,
) -> None:
    j = np.load(jaxley_npz)
    n = np.load(neuron_npz)
    metrics = ecs_common.compute_metrics(
        t_ref=n["t"],
        v_ref=n["v"],
        t_other=j["t"],
        v_other=j["v"],
    )
    site_names = [str(x) for x in n["site_names"].tolist()]
    title_main = (
        f"ECS parity on {CONFIG['cell_label']}: "
        f"{CONFIG['jaxley_label']} vs {CONFIG['neuron_label']}"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ecs_common.plot_ecs_parity_sites(
        metrics=metrics,
        phi_sites=n["phi_sites"],
        site_names=site_names,
        out_path=out_path,
        title_main=title_main,
        primary_site="soma",
        secondary_site="apical",
        scale=scale,
    )


def make_throughput_bbp(
    out_path: Path,
    *,
    neuron_npz: Path = DATA_DIR / "throughput_bbp_neuron.npz",
    jaxley_npz: Path = DATA_DIR / "throughput_bbp_jaxley.npz",
    scale: float = 2.0,
) -> None:
    n = np.load(neuron_npz)
    j = np.load(jaxley_npz)

    n_n = n["n_values"].astype(float)
    n_wall = n["wall_s"].astype(float)
    n_per_sim = float(n["per_sim_s"].mean())
    # Linear-extrapolation reference line, capped at the largest measured
    # NEURON N (no extrapolation past the data we actually have).
    n_grid = np.geomspace(n_n.min(), n_n.max(), 50)
    n_extrap = n_grid * n_per_sim

    j_n = j["n_values"].astype(float)
    j_steady = j["cached_s"].astype(float)
    j_per_sim_steady = j_steady / j_n

    # Headline comparison: same N on both sides, both directly measured.
    j_max_idx = int(np.argmax(j_n))
    j_n_max = int(j_n[j_max_idx])
    j_steady_max = float(j_steady[j_max_idx])
    j_per_sim_steady_max = float(j_per_sim_steady[j_max_idx])

    # NEURON wall at the same N (measured if N is in n_n, else linear-extrap).
    if j_n_max in n_n.astype(int).tolist():
        neuron_at_n = float(n_wall[np.searchsorted(n_n, j_n_max)])
        neuron_label = "measured"
    else:
        neuron_at_n = j_n_max * n_per_sim
        neuron_label = "linear extrapolation"
    speedup_total = neuron_at_n / j_steady_max

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=n_n,
            y=n_wall,
            mode="markers+lines",
            name=f"{CONFIG['neuron_label']} - measured",
            marker=dict(color="#1f77b4", size=11, symbol="circle"),
            line=dict(color="#1f77b4", width=2.5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=n_grid,
            y=n_extrap,
            mode="lines",
            name=f"{CONFIG['neuron_label']} - linear extrapolation",
            line=dict(color="#1f77b4", width=2, dash="dash"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=j_n,
            y=j_steady,
            mode="markers+lines",
            name=f"{CONFIG['jaxley_label']} - JIT-warm",
            marker=dict(color="#d62728", size=11, symbol="diamond", line=dict(color="black", width=0.8)),
            line=dict(color="#d62728", width=2.5),
        )
    )
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.02,
        y=0.98,
        xanchor="left",
        yanchor="top",
        text=(
            f"<b>At N = {j_n_max:,} experiments:</b><br>"
            f"{CONFIG['neuron_label']}  ~{_human_time(neuron_at_n)}  ({neuron_label})<br>"
            f"{CONFIG['jaxley_label']}  ~{_human_time(j_steady_max)}"
            f"  ({j_per_sim_steady_max*1000:.1f} ms/sim, JIT warm)<br>"
            f"<b>speedup ~{speedup_total:.0f}x</b>"
        ),
        showarrow=False,
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="#888",
        borderwidth=1,
        borderpad=8,
        font=dict(size=12),
    )
    y_ticks = [1, 10, 60, 600, 3600, 36000, 86400]
    fig.update_xaxes(title_text="number of independent simulations N", type="log", dtick=1)
    fig.update_yaxes(
        title_text="total wall time (log scale)",
        type="log",
        tickvals=y_ticks,
        ticktext=[_human_time(v) for v in y_ticks],
    )
    fig.update_layout(
        width=900,
        height=560,
        title=(
            f"{CONFIG['cell_label']} throughput: "
            f"{CONFIG['neuron_label']} vs {CONFIG['jaxley_label']}"
        ),
        legend=dict(orientation="h", yanchor="bottom", y=-0.35, xanchor="center", x=0.5),
        margin=dict(t=70, b=140, l=90, r=40),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(out_path), scale=scale)


def make_throughput_v5p4_scaling(
    out_path: Path,
    *,
    cheap_npz: Path = DATA_DIR / "throughput_v5p4_cheap.npz",
    full_npz: Path = DATA_DIR / "throughput_v5p4_full.npz",
    ceiling_npz: Path = DATA_DIR / "throughput_v5p4_ceiling.npz",
    scale: float = 2.0,
) -> None:
    """Plot Jaxley batch-throughput on a v5litepod-4 pod-slice.

    Two curves on shared log--log axes:
      cheap   = ncomp=2 (post-expansion ~700 comps/cell), B in [4, 1024]
      full    = ncomp=50 (~17500 comps/cell), B in [4, 4096],
                merging the ncomp50 sweep and the ceiling probe at the
                overlap point B=256.
    """
    cheap = np.load(cheap_npz)
    fullL = np.load(full_npz)
    ceil_ = np.load(ceiling_npz)

    cheap_B = cheap["n_values"].astype(float)
    cheap_cps = cheap_B / cheap["cached_s"].astype(float)

    # Merge the full-fidelity sweeps at the overlap point B=256
    full_B = fullL["n_values"].astype(float)
    full_cps = full_B / fullL["cached_s"].astype(float)
    ceil_B = ceil_["n_values"].astype(float)
    ceil_cps = ceil_B / ceil_["cached_s"].astype(float)
    # ceiling overlaps with full at B=256; trust the ceiling run for the
    # large-B portion since it is the run that reached 4096 in one go.
    keep = full_B < ceil_B.min()
    full_merged_B = np.concatenate([full_B[keep], ceil_B])
    full_merged_cps = np.concatenate([full_cps[keep], ceil_cps])

    peak_idx = int(np.argmax(full_merged_cps))
    peak_B = int(full_merged_B[peak_idx])
    peak_cps = float(full_merged_cps[peak_idx])

    fig = go.Figure()
    cheap_n_dev = _n_devices(cheap)
    full_n_dev = _n_devices(fullL)
    fig.add_trace(
        go.Scatter(
            x=cheap_B,
            y=cheap_cps,
            mode="markers+lines",
            name=f"cheap cells ({int(cheap['ncomps'])} comps/cell)",
            marker=dict(color="#1f77b4", size=10, symbol="circle"),
            line=dict(color="#1f77b4", width=2.2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=full_merged_B,
            y=full_merged_cps,
            mode="markers+lines",
            name=f"full fidelity ({int(fullL['ncomps'])} comps/cell)",
            marker=dict(color="#d62728", size=10, symbol="diamond"),
            line=dict(color="#d62728", width=2.2),
        )
    )
    fig.add_annotation(
        x=np.log10(peak_B),
        y=np.log10(peak_cps),
        xref="x",
        yref="y",
        text=f"peak {peak_cps:.2f} cells/s @ B = {peak_B}",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=1.5,
        ax=40,
        ay=-30,
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="#888",
        borderwidth=1,
        borderpad=4,
        font=dict(size=11),
    )
    fig.update_xaxes(title_text="batch size B (cells per vmap dispatch)", type="log", dtick=1)
    fig.update_yaxes(title_text="throughput (cells per second)", type="log")
    n_dev = max(cheap_n_dev, full_n_dev)  # both runs are on the same pod
    fig.update_layout(
        width=900,
        height=520,
        title=f"{CONFIG['jaxley_label']} throughput, {n_dev}-chip pod-slice",
        legend=dict(orientation="h", yanchor="bottom", y=-0.30, xanchor="center", x=0.5),
        margin=dict(t=70, b=120, l=90, r=40),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(out_path), scale=scale)


def _verification_data() -> dict[str, np.ndarray]:
    """Run the operator round-trip + convergence checks and return raw arrays.

    Mirrors ``tests/test_ecs_operator_consistency.py``'s
    ``test_operator_equivalence_cable_random_v`` (round-trip) and
    ``test_analytical_activating_function_convergence`` (O(dx^2)). Cheap
    and deterministic; we run inline rather than caching to keep the
    figure step self-contained.
    """
    os.environ.setdefault("JAX_ENABLE_X64", "1")
    import jax
    import jax.numpy as jnp
    import jaxley as jx
    from jaxley.solver_voltage import _voltage_vectorfield  # type: ignore[attr-defined]

    from jaxley_extracellular.extracellular.discretization import (
        build_voltage_operator_G,
    )
    from jaxley_extracellular.extracellular.jaxley_adapter import (
        ensure_compartment_centers,
        get_compartment_xyz,
    )

    jax.config.update("jax_enable_x64", True)

    # Helper: construct a uniform cable with consistent xyz coordinates.
    def _uniform_cable(ncomp: int, total_length_um: float = 4000.0) -> jx.Branch:
        comp_len = total_length_um / ncomp
        comp = jx.Compartment()
        branch = jx.Branch(comp, ncomp=ncomp)
        branch.set("length", comp_len)
        branch.set("radius", 1.0)
        branch.set("axial_resistivity", 100.0)
        branch.set("capacitance", 1.0)
        branch.xyzr[0] = np.array([[0, 0, 0, 1.0], [total_length_um, 0, 0, 1.0]])
        branch.compute_compartment_centers()
        branch.to_jax()
        return branch

    # ---- Round-trip on a 50-compartment cable, random voltages ----
    branch = _uniform_cable(ncomp=50)
    params = branch.get_all_parameters(pstate=[])
    G = np.asarray(build_voltage_operator_G(branch, params), dtype=np.float64)

    edges = branch.base._comp_edges
    sinks = np.asarray(edges["sink"].to_list())
    sources = np.asarray(edges["source"].to_list())
    types = np.asarray(edges["type"].to_list())
    n_nodes = int(branch.base._n_nodes)
    idx = np.asarray(branch.base._internal_node_inds)
    axial = params["axial_conductances"]["v"]

    rng = np.random.default_rng(7)
    v = rng.standard_normal(50) * 20 - 65
    Gv = G @ v
    v_full = jnp.zeros(n_nodes).at[idx].set(jnp.asarray(v))
    vf = _voltage_vectorfield(
        v_full,
        jnp.zeros(n_nodes),
        jnp.zeros(n_nodes),
        axial,
        sinks,
        sources,
        types,
        n_nodes,
    )
    vf_comps = np.asarray(vf[idx])
    rel_err = np.abs(Gv - vf_comps) / (np.abs(vf_comps) + 1e-30)

    # ---- Convergence: error vs dx across ncomp = {50, 100, 200, 400} ----
    total_length = 4000.0
    sigma = 0.3
    y_e = 1000.0
    x_e = total_length / 2.0
    ncomps = [50, 100, 200, 400]
    dxs: list[float] = []
    errors: list[float] = []
    for nc in ncomps:
        b = _uniform_cable(nc, total_length)
        p = b.get_all_parameters(pstate=[])
        Gnc = np.asarray(build_voltage_operator_G(b, p), dtype=np.float64)
        ensure_compartment_centers(b)
        x = np.asarray(get_compartment_xyz(b), dtype=np.float64)[:, 0]
        dx = float(x[1] - x[0])
        g_ax = Gnc[nc // 2, nc // 2 - 1]
        C = 1e3 / (4 * np.pi * sigma)
        phi_e = C / np.sqrt((x - x_e) ** 2 + y_e**2)
        Gphi = Gnc @ phi_e
        dxv = x - x_e
        r2 = dxv**2 + y_e**2
        d2phi = C * (2 * dxv**2 - y_e**2) / r2**2.5
        analytical = g_ax * dx**2 * d2phi
        lo, hi = int(nc * 0.45), int(nc * 0.55)
        err = float(
            np.max(np.abs(Gphi[lo:hi] - analytical[lo:hi]) /
                   (np.abs(analytical[lo:hi]) + 1e-30))
        )
        dxs.append(dx)
        errors.append(err)

    return {
        "roundtrip_rel_err": rel_err,
        "convergence_dx_um": np.array(dxs),
        "convergence_rel_err": np.array(errors),
        "convergence_ncomps": np.array(ncomps),
    }


def make_verification(
    out_path: Path,
    *,
    scale: float = 2.0,
) -> None:
    """Two-panel verification figure: round-trip + O(dx^2) convergence."""
    d = _verification_data()
    rel_err = d["roundtrip_rel_err"]
    dxs = d["convergence_dx_um"]
    errors = d["convergence_rel_err"]
    ncomps_arr = d["convergence_ncomps"]

    # Reference slope-2 line, anchored at the largest dx.
    dx_ref = np.linspace(dxs.min(), dxs.max(), 50)
    err_ref = errors[0] * (dx_ref / dxs[0]) ** 2

    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.55, 0.45],
        horizontal_spacing=0.12,
        subplot_titles=(
            "Round-trip: per-compartment relative error",
            "Convergence: |G phi - analytical| vs Delta x",
        ),
    )

    fig.add_trace(
        go.Histogram(
            x=np.log10(np.maximum(rel_err, 1e-20)),
            nbinsx=25,
            marker=dict(color="#1f77b4", line=dict(color="black", width=0.4)),
            name="round-trip error",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.add_vline(
        x=float(np.log10(np.median(np.maximum(rel_err, 1e-20)))),
        line=dict(color="black", width=1, dash="dot"),
        row=1,
        col=1,
        annotation_text=f"median = {np.median(rel_err):.1e}",
        annotation_position="top right",
    )
    fig.update_xaxes(title_text="log10(relative error)", row=1, col=1)
    fig.update_yaxes(title_text="compartment count", row=1, col=1)

    fig.add_trace(
        go.Scatter(
            x=dxs,
            y=errors,
            mode="markers+lines+text",
            marker=dict(color="#d62728", size=11, symbol="diamond"),
            line=dict(color="#d62728", width=2),
            text=[f"N = {n}" for n in ncomps_arr.tolist()],
            textposition="top right",
            name="measured",
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=dx_ref,
            y=err_ref,
            mode="lines",
            line=dict(color="#999", width=1.5, dash="dash"),
            name="slope = 2",
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    fig.update_xaxes(
        title_text="Delta x (um)",
        type="log",
        row=1,
        col=2,
    )
    fig.update_yaxes(
        title_text="relative error",
        type="log",
        row=1,
        col=2,
    )

    ratios = [errors[i - 1] / errors[i] for i in range(1, len(errors))]
    fig.add_annotation(
        xref="x2",
        yref="y2",
        x=np.log10(dxs.min() * 2),
        y=np.log10(errors[-1] * 5),
        text=(
            "ratio per doubling: "
            + ", ".join(f"{r:.2f}x" for r in ratios)
        ),
        showarrow=False,
        bgcolor="rgba(255,255,255,0.85)",
        bordercolor="#888",
        borderwidth=1,
        borderpad=4,
        font=dict(size=11),
    )

    fig.update_layout(
        width=1100,
        height=440,
        title="Numerical verification of the discrete activating function",
        margin=dict(t=80, b=70, l=80, r=40),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(out_path), scale=scale)


RECEIPTS_DIR = PROJECT_ROOT / "results" / "paper_package" / "receipts"


def make_gradient_arrow(
    out_path: Path,
    *,
    receipt_path: Path = RECEIPTS_DIR / "gradient_receipt_ecs.txt",
    scale: float = 2.0,
) -> None:
    """2-D projection of the BBP Pyr morphology with the position-gradient arrow.

    Parses electrode position and gradient triple from the canonical
    receipt at ``results/paper_package/receipts/gradient_receipt_ecs.txt``
    so any numbers in the figure track the receipt and the paper text.
    """
    import re

    from jaxley_extracellular.bbp.cell_factory import make_pyr_cell

    text = receipt_path.read_text()

    elec_match = re.search(
        r"Electrode xyz\s*\(um\)\s*=\s*\(([^)]+)\)", text
    )
    if elec_match is None:
        raise ValueError(f"could not parse electrode xyz from {receipt_path}")
    elec_xyz = np.array([float(s) for s in elec_match.group(1).split(",")])

    grad_match = re.search(
        r"d\(peak_mV\)/d\(electrode_xyz\):\s*\[([^\]]+)\]", text
    )
    if grad_match is None:
        raise ValueError(f"could not parse gradient from {receipt_path}")
    grad = np.array([float(s) for s in grad_match.group(1).split()])

    cell = make_pyr_cell(ncomp=2, max_branch_len=100.0)
    nodes = cell.nodes
    xyz = nodes[["x", "y", "z"]].to_numpy(dtype=float)
    soma_mask = nodes["soma"].to_numpy().astype(bool) if "soma" in nodes.columns else np.zeros(len(nodes), dtype=bool)

    # Project onto the y-z plane: the dorsoventral axis (y) is most
    # informative for the (0, 100, 0) electrode placement, and z spreads
    # the dendrite tree. We render the morphology as a single faint
    # silhouette rather than per-section because Jaxley's SWC import does
    # not always preserve the apical/axon/basal split for BBP cells. The
    # soma centre is annotated explicitly.
    morph_y, morph_z = xyz[:, 1], xyz[:, 2]
    soma_y = float(np.mean(morph_y[soma_mask])) if soma_mask.any() else 0.0
    soma_z = float(np.mean(morph_z[soma_mask])) if soma_mask.any() else 0.0

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=morph_y,
            y=morph_z,
            mode="markers",
            marker=dict(color="#888", size=3, opacity=0.6),
            name="morphology",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[soma_y],
            y=[soma_z],
            mode="markers+text",
            marker=dict(color="black", size=12, symbol="circle", line=dict(width=2)),
            text=["soma"],
            textposition="bottom center",
            name="soma",
            showlegend=False,
        )
    )

    # Electrode marker.
    fig.add_trace(
        go.Scatter(
            x=[elec_xyz[1]],
            y=[elec_xyz[2]],
            mode="markers+text",
            marker=dict(color="black", size=14, symbol="x", line=dict(width=2)),
            text=["electrode"],
            textposition="top right",
            name="electrode",
            showlegend=False,
        )
    )

    # Gradient arrow: +grad is the direction that *increases* v_soma_peak,
    # so an electrode displacement along +grad raises the soma response.
    direction = grad / np.linalg.norm(grad)
    arrow_len = max(40.0, 0.2 * float(np.ptp(morph_y)))
    tip = elec_xyz[1:] + direction[1:] * arrow_len
    fig.add_annotation(
        x=tip[0],
        y=tip[1],
        ax=elec_xyz[1],
        ay=elec_xyz[2],
        xref="x",
        yref="y",
        axref="x",
        ayref="y",
        showarrow=True,
        arrowhead=3,
        arrowsize=1.5,
        arrowwidth=2.5,
        arrowcolor="#cc7700",
    )
    fig.add_annotation(
        x=tip[0],
        y=tip[1],
        text=(
            f"<b>d v_soma,peak / d x_e</b> = "
            f"({grad[0]:+.2f}, {grad[1]:+.2f}, {grad[2]:+.2f}) mV/um"
        ),
        showarrow=False,
        xanchor="left",
        yanchor="bottom",
        xshift=14,
        yshift=10,
        bgcolor="rgba(255,255,255,0.85)",
        bordercolor="#888",
        borderwidth=1,
        borderpad=4,
        font=dict(size=11),
    )

    fig.update_xaxes(title_text="y (um)", scaleanchor="y", scaleratio=1.0)
    fig.update_yaxes(title_text="z (um)")
    fig.update_layout(
        width=820,
        height=720,
        title=f"Position gradient on {CONFIG['cell_label']}",
        margin=dict(t=70, b=70, l=70, r=30),
        showlegend=False,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(out_path), scale=scale)


def make_strength_duration(
    out_path: Path,
    *,
    sweep_zarr: Path = DATA_DIR / "sweep_strength_duration.zarr",
    saturation_frac: float = 0.99,
    scale: float = 2.0,
) -> None:
    """Strength-duration curves on the HH cable.

    One panel per electrode distance; one curve per polarity. Threshold
    points within ``saturation_frac`` of the binary-search upper bracket
    are treated as ``no spike`` and dropped from the curve, since they
    indicate the search hit the ceiling rather than a real threshold.
    """
    import json

    import xarray as xr

    ds = xr.open_zarr(sweep_zarr, consolidated=False)
    cfg = json.loads(ds.attrs["config_json"])
    amp_hi = float(cfg["amp_hi"])
    df = ds.to_dataframe().reset_index()
    df["saturated"] = df["threshold_uA"] >= saturation_frac * amp_hi

    distances = sorted(float(x) for x in df["electrode_distance_um"].unique())
    polarities = ["monophasic_cathodic", "biphasic_cathodic_first", "monophasic_anodic"]
    polarity_label = {
        "monophasic_cathodic": "monophasic cathodic",
        "biphasic_cathodic_first": "biphasic (cathodic-first)",
        "monophasic_anodic": "monophasic anodic",
    }
    polarity_color = {
        "monophasic_cathodic": "#1f77b4",
        "biphasic_cathodic_first": "#2ca02c",
        "monophasic_anodic": "#d62728",
    }

    n_panels = len(distances)
    cols = 2 if n_panels >= 4 else n_panels
    rows = (n_panels + cols - 1) // cols
    fig = make_subplots(
        rows=rows,
        cols=cols,
        shared_yaxes=True,
        shared_xaxes=True,
        horizontal_spacing=0.04,
        vertical_spacing=0.12,
        subplot_titles=tuple(f"d = {d:g} um" for d in distances),
    )

    for idx, d in enumerate(distances):
        row = idx // cols + 1
        col = idx % cols + 1
        for w in polarities:
            sub = df[(df["electrode_distance_um"] == d) & (df["waveform_type"] == w)]
            sub = sub.sort_values("pulse_width_ms")
            fired = sub[~sub["saturated"]]
            color = polarity_color[w]
            label = polarity_label[w]
            fig.add_trace(
                go.Scatter(
                    x=fired["pulse_width_ms"],
                    y=fired["threshold_uA"],
                    mode="markers+lines",
                    name=label,
                    legendgroup=w,
                    showlegend=(idx == 0),
                    marker=dict(color=color, size=7),
                    line=dict(color=color, width=2),
                ),
                row=row,
                col=col,
            )
        # Bracket ceiling as a dashed reference line. Saturated points are
        # implicit (anything that would land at this line did not spike).
        fig.add_hline(
            y=amp_hi,
            line=dict(color="#999", width=1, dash="dot"),
            row=row,
            col=col,
        )

    for col in range(1, cols + 1):
        fig.update_xaxes(
            title_text="pulse width (ms)",
            type="log",
            row=rows,
            col=col,
        )
    for row in range(1, rows + 1):
        fig.update_yaxes(title_text="|A*| (uA)", type="log", row=row, col=1)

    fig.update_layout(
        width=900,
        height=620,
        title="Strength-duration on the HH cable",
        legend=dict(orientation="h", yanchor="bottom", y=-0.18, xanchor="center", x=0.5),
        margin=dict(t=70, b=120, l=80, r=30),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(out_path), scale=scale)


def make_biphasic_grid(
    out_path: Path,
    *,
    grid_zarr: Path = DATA_DIR / "grid_biphasic_d100.zarr",
    scale: float = 2.0,
) -> None:
    """Charge-balanced biphasic grid heatmap on the BBP Pyr cell.

    One panel per anodic amplitude $A_p$. Each panel is a $(T_p, T_n)$
    heatmap of the maximum soma voltage observed across the four tested
    pulse-train frequencies; cells whose $\\mathit{vmax}$ crosses
    $0\\,$mV correspond to a spike at some frequency in the sweep.
    """
    import json

    import xarray as xr

    ds = xr.open_zarr(grid_zarr, consolidated=False)
    cfg = json.loads(ds.attrs["config_json"])
    df = ds.to_dataframe().reset_index()

    # Marginalise frequency by taking the max vmax_mV per (Ap, Tp, Tn).
    agg = df.groupby(["ap_uA", "tp_us", "tn_us"]).agg(
        vmax_mV=("vmax_mV", "max"),
        spiked_any=("spiked", "max"),
    ).reset_index()

    aps = sorted(agg["ap_uA"].unique().tolist())
    tps = sorted(agg["tp_us"].unique().tolist())
    tns = sorted(agg["tn_us"].unique().tolist())
    vmin = float(agg["vmax_mV"].min())
    vmax = float(agg["vmax_mV"].max())

    fig = make_subplots(
        rows=1,
        cols=len(aps),
        shared_yaxes=True,
        horizontal_spacing=0.03,
        subplot_titles=tuple(f"Ap = {a:g} uA" for a in aps),
    )

    for col, a in enumerate(aps, start=1):
        sub = agg[agg["ap_uA"] == a].pivot(index="tn_us", columns="tp_us", values="vmax_mV")
        sub = sub.reindex(index=tns, columns=tps)
        fig.add_trace(
            go.Heatmap(
                x=[f"{int(v)}" for v in tps],
                y=[f"{int(v)}" for v in tns],
                z=sub.values,
                zmin=vmin,
                zmax=vmax,
                colorscale="RdBu_r",
                zmid=0.0,
                showscale=(col == len(aps)),
                colorbar=dict(title="max v_soma (mV)", thickness=14, len=0.85),
                hoverongaps=False,
            ),
            row=1,
            col=col,
        )
        # Symmetric-biphasic diagonal Tp=Tn as visual reference.
        fig.add_shape(
            type="line",
            xref=f"x{'' if col == 1 else col} domain",
            yref=f"y{'' if col == 1 else col} domain",
            x0=0,
            x1=1,
            y0=0,
            y1=1,
            line=dict(color="black", width=1, dash="dot"),
            row=1,
            col=col,
        )
        fig.update_xaxes(title_text="Tp (us)", row=1, col=col)

    fig.update_yaxes(title_text="Tn (us)", row=1, col=1)
    fig.update_layout(
        width=1200,
        height=380,
        title=f"Charge-balanced biphasic grid on {CONFIG['cell_label']}",
        margin=dict(t=70, b=70, l=70, r=80),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(out_path), scale=scale)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--which",
        choices=("all", "bbp-parity", "bbp-ecs", "throughput", "throughput-scaling",
                 "strength-duration", "biphasic-grid", "verification",
                 "gradient-arrow"),
        default="all",
    )
    args = parser.parse_args()

    if args.which in ("all", "bbp-parity"):
        make_bbp_intracellular_parity(OUT_DIR / "bbp_intracellular_parity.png")
    if args.which in ("all", "bbp-ecs"):
        make_bbp_ecs_parity(OUT_DIR / "bbp_ecs_parity.png")
    if args.which in ("all", "throughput"):
        make_throughput_bbp(OUT_DIR / "bbp_throughput.png")
    if args.which in ("all", "throughput-scaling"):
        make_throughput_v5p4_scaling(OUT_DIR / "throughput_v5p4_scaling.png")
    if args.which in ("all", "strength-duration"):
        make_strength_duration(OUT_DIR / "strength_duration.png")
    if args.which in ("all", "biphasic-grid"):
        make_biphasic_grid(OUT_DIR / "biphasic_grid.png")
    if args.which in ("all", "verification"):
        make_verification(OUT_DIR / "verification.png")
    if args.which in ("all", "gradient-arrow"):
        make_gradient_arrow(OUT_DIR / "gradient_arrow.png")


if __name__ == "__main__":
    main()
