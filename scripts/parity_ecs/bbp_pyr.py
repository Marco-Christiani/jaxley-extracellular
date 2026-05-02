"""NEURON vs Jaxley ECS parity on the BBP L2/3 pyramidal cell.

Extends the HH-cable parity (``scripts.parity_ecs.hh``) to a morphologically
accurate cell with the full BBP channel set. Same pipeline on both sides:

    * Load the BBP cADpyr229 cell (Jaxley: ``make_pyr_cell``; NEURON: the
      BBP HOC template bundle).
    * Place a single point-source electrode a fixed distance above the soma.
    * Apply a brief cathodic pulse, compute phi_e analytically at each
      compartment centre.
    * Jaxley drives the cell via our ECS equivalent-current pipeline.
      NEURON drives via its ``extracellular`` mechanism + ``Vector.play``.
    * Record voltage at four matched morphological sites (soma, apical
      trunk, basal dendrite, axon) and compare traces.

Usage:
    # Jaxley side (default env)
    python -m scripts.parity_ecs.bbp_pyr --side jaxley

    # NEURON side (dedicated NEURON shell)
    nix develop .#neuron
    cd reference/bbp/simulation && nrnivmodl mechanisms
    cd /path/to/jaxley-extracellular
    neuron-python -m scripts.parity_ecs.bbp_pyr --side neuron

    # Compare + figure (default env, both npzs must exist)
    python -m scripts.parity_ecs.bbp_pyr --side compare

Outputs:
    results/parity_bbp_ecs_{jaxley,neuron}.npz
    slides.local/figures/ecs_parity_bbp.png
"""

from __future__ import annotations

import argparse
import itertools
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from scripts.neuron_bbp_common import load_bbp_pyr_cell, pyr_cell_dir
from scripts.parity_common import (
    interp_xyz_on_polyline,
    interpolate_branch_voltage,
    provenance_fields,
    segment_center_xyz,
)
from scripts.parity_ecs import common

OUT_DIR = Path(__file__).resolve().parent.parent.parent / "results"
FIGURE_OUT = (Path(__file__).resolve().parent.parent.parent
              / "slides.local" / "figures" / "ecs_parity_bbp.png")

# --- shared simulation parameters -------------------------------------------
DT_MS       = 0.025
T_MAX_MS    = 15.0
V_INIT_MV   = -75.0
CELSIUS_C   = 34.0

# Electrode placed above the soma. The BBP Pyr soma sits near the origin,
# apical trunk climbs along +y, so placing the electrode at +y aligns with
# the dendritic "antenna" and produces a recognisable response.
ELEC_XYZ_UM = (0.0, 100.0, 0.0)
SIGMA_S_M   = 0.3

# Cathodic pulse amplitude defaults to subthreshold so the cell sits in
# the linear regime where cross-solver parity is cleanest. Suprathreshold
# (e.g. -100 uA) drives produce spike-timing drift of ~1 ms between
# NEURON and our pipeline. The residuals are the same family as the BBP
# intracellular parity caveat (Ih distance gradient, f32 vs f64), which dominates
# pointwise RMSE even though the waveform shape matches. The two stim
# spec variants are exposed here as named constants so follow-up
# experiments can toggle without re-reading the script.
PULSE_SUB = common.PulseSpec(amp_uA=-8.0,   width_ms=1.0, delay_ms=2.0)
PULSE_SUPRA = common.PulseSpec(amp_uA=-100.0, width_ms=1.0, delay_ms=2.0)
PULSE = PULSE_SUB

# Four matched recording sites across the morphology. Names must agree
# between the Jaxley and NEURON sides so the comparison pairs them by key.
RECORD_SITES: tuple[str, ...] = ("soma", "apical", "basal", "axon")

JAXLEY_SITE_PROXY_TARGETED = "projected_neuron_site_xyz_within_group_polyline"
JAXLEY_SITE_PROXY_FALLBACK = "projected_group_branch0_midpoint_within_group_polyline"
NEURON_SITE_PROXY = "section_midpoint_group0_x_0p5"
SITE_XYZ_TOL_UM = 1.0
PHI_SITE_RTOL = 0.05
PHI_SITE_ATOL_MV = 0.05


def _project_point_to_polyline(poly_xyz: np.ndarray, point_xyz: np.ndarray) -> tuple[float, np.ndarray, float]:
    """Project a point onto a polyline.

    Returns:
        (frac, xyz, dist_um) where frac is fractional arc length in [0, 1].
    """
    seg_len = np.sqrt(((poly_xyz[1:] - poly_xyz[:-1]) ** 2).sum(axis=1))
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = float(cum[-1])
    if total <= 0:
        q0 = poly_xyz[0].copy()
        return 0.0, q0, float(np.linalg.norm(point_xyz - q0))

    best_dist = float("inf")
    best_frac = 0.0
    best_xyz = poly_xyz[0].copy()

    for i, (a, b) in enumerate(itertools.pairwise(poly_xyz)):
        ab = b - a
        denom = float(np.dot(ab, ab))
        t = 0.0 if denom == 0.0 else float(np.clip(np.dot(point_xyz - a, ab) / denom, 0.0, 1.0))
        q = a + t * ab
        dist = float(np.linalg.norm(point_xyz - q))
        if dist < best_dist:
            best_dist = dist
            best_xyz = q
            best_frac = float((cum[i] + t * seg_len[i]) / total)

    return best_frac, best_xyz, best_dist


def _load_site_targets(site_targets_npz: Path) -> dict[str, np.ndarray]:
    """Load named site xyz targets from a saved parity npz."""
    d = np.load(site_targets_npz, allow_pickle=True)
    names = [str(x) for x in d["site_names"].tolist()]
    xyz = np.asarray(d["site_xyz"], dtype=float)
    out = {name: xyz_i for name, xyz_i in zip(names, xyz, strict=True)}
    missing = [name for name in RECORD_SITES if name not in out]
    if missing:
        raise RuntimeError(f"{site_targets_npz} missing site targets for {missing}")
    return {name: out[name].copy() for name in RECORD_SITES}


def _default_jaxley_site_targets(cell) -> dict[str, np.ndarray]:
    """Fallback targets: midpoint of branch 0 within each named group."""
    out: dict[str, np.ndarray] = {}
    for name in RECORD_SITES:
        poly_xyz = getattr(cell, name).branch(0).xyzr[0][:, :3].astype(float)
        out[name] = interp_xyz_on_polyline(poly_xyz, 0.5)
    return out


def _pick_jaxley_site_specs(cell, site_targets: dict[str, np.ndarray]) -> list[dict[str, object]]:
    """For each named site, choose the closest branch + arc-length location in that group."""
    specs: list[dict[str, object]] = []
    for name in RECORD_SITES:
        target_xyz = np.asarray(site_targets[name], dtype=float)
        group = getattr(cell, name)
        branch_ids = sorted(int(x) for x in group.nodes["local_branch_index"].unique())

        best: dict[str, object] | None = None
        for branch_id in branch_ids:
            poly_xyz = group.branch(branch_id).xyzr[0][:, :3].astype(float)
            frac, xyz, dist_um = _project_point_to_polyline(poly_xyz, target_xyz)
            cand = {
                "name": name,
                "target_xyz": target_xyz,
                "branch_id": branch_id,
                "frac": frac,
                "xyz": xyz,
                "dist_um": dist_um,
            }
            if best is None or float(cand["dist_um"]) < float(best["dist_um"]):
                best = cand
        assert best is not None
        specs.append(best)
    return specs


def run_jaxley(out_path: Path, site_targets_npz: Path | None = None) -> None:
    os.environ.setdefault("JAX_ENABLE_X64", "1")

    import jax
    import jax.numpy as jnp
    import jaxley as jx

    from jaxley_extracellular.bbp.cell_factory import make_pyr_cell
    from jaxley_extracellular.extracellular.discretization import build_voltage_operator_G
    from jaxley_extracellular.extracellular.equivalent_current import phi_e_to_ecs_nA
    from jaxley_extracellular.extracellular.field import point_source_potential

    print("BBP Pyr ECS parity (Jaxley side). jax:", jax.__version__,
          "backend:", jax.default_backend())

    cell = make_pyr_cell(ncomp=2, max_branch_len=100.0)
    cell.set("v", V_INIT_MV)
    cell.record("v", verbose=False)

    comp_xyz = cell.nodes[["x", "y", "z"]].to_numpy()
    n_comp = comp_xyz.shape[0]
    print(f"  cell: {n_comp} compartments, recording all compartments for site interpolation")

    if site_targets_npz is not None and site_targets_npz.exists():
        print(f"  site targets: {site_targets_npz}")
        site_targets = _load_site_targets(site_targets_npz)
        site_proxy = JAXLEY_SITE_PROXY_TARGETED
    else:
        print("  site targets: fallback to branch-0 midpoints")
        site_targets = _default_jaxley_site_targets(cell)
        site_proxy = JAXLEY_SITE_PROXY_FALLBACK

    site_specs = _pick_jaxley_site_specs(cell, site_targets)

    t = common.time_grid(DT_MS, T_MAX_MS)
    wave = common.waveform_uA(t, PULSE)

    phi_e_mV = np.asarray(point_source_potential(
        comp_xyz=comp_xyz,
        electrode_positions=jnp.asarray(ELEC_XYZ_UM)[None, :],
        electrode_currents=jnp.asarray(wave)[None, :],
        sigma=SIGMA_S_M,
    ))  # (Ncomp, T)

    # Equivalent-current conversion through the real ECS pipeline, then
    # package into Jaxley's data_stimuli format in a single call. Passing
    # the full (Ncomp, T) matrix to cell.data_stimulate distributes the
    # per-compartment traces across the cell without per-segment iteration.
    params = cell.get_all_parameters(pstate=[])
    G = build_voltage_operator_G(cell, params)
    cm = jnp.asarray(cell.nodes["capacitance"].to_numpy())
    area = jnp.asarray(cell.nodes["area"].to_numpy())
    i_ecs_nA = phi_e_to_ecs_nA(jnp.asarray(phi_e_mV), G, cm, area)
    data_stimuli = cell.data_stimulate(i_ecs_nA, data_stimuli=None, verbose=False)

    # Integrate with checkpoint_lengths for memory safety on the backward
    # pass (not strictly needed for forward-only, kept for consistency with
    # the other BBP scripts in this repo).
    v_full = np.asarray(jx.integrate(
        cell, delta_t=DT_MS, t_max=T_MAX_MS, data_stimuli=data_stimuli,
    ))  # (n_compartments, T)

    site_v_rows: list[np.ndarray] = []
    site_xyz_rows: list[np.ndarray] = []
    site_branch_ids: list[int] = []
    site_fracs: list[float] = []
    site_project_dist_um: list[float] = []
    site_support_gci: list[np.ndarray] = []
    site_support_w: list[np.ndarray] = []
    for spec in site_specs:
        name = str(spec["name"])
        branch_id = int(spec["branch_id"])
        frac = float(spec["frac"])
        branch = getattr(cell, name).branch(branch_id)
        trace, gci_pair, w_pair = interpolate_branch_voltage(v_full, branch, frac)
        site_v_rows.append(trace)
        site_xyz_rows.append(np.asarray(spec["xyz"], dtype=float))
        site_branch_ids.append(branch_id)
        site_fracs.append(frac)
        site_project_dist_um.append(float(spec["dist_um"]))
        site_support_gci.append(gci_pair)
        site_support_w.append(w_pair)

    v = np.stack(site_v_rows, axis=0)
    site_xyz_arr = np.stack(site_xyz_rows, axis=0)
    site_branch_id_arr = np.array(site_branch_ids, dtype=int)
    site_frac_arr = np.array(site_fracs, dtype=float)
    site_project_dist_um_arr = np.array(site_project_dist_um, dtype=float)
    site_support_gci_arr = np.stack(site_support_gci, axis=0)
    site_support_w_arr = np.stack(site_support_w, axis=0)

    phi_sites = np.asarray(point_source_potential(
        comp_xyz=site_xyz_arr,
        electrode_positions=jnp.asarray(ELEC_XYZ_UM)[None, :],
        electrode_currents=jnp.asarray(wave)[None, :],
        sigma=SIGMA_S_M,
    ))
    i_ecs_nA = np.asarray(i_ecs_nA)

    t_out = np.arange(v.shape[1]) * DT_MS

    import jax
    backend = jax.default_backend()
    OUT_DIR.mkdir(exist_ok=True)
    np.savez(
        out_path, t=t_out, v=v,
        site_names=np.array(RECORD_SITES),
        site_xyz=site_xyz_arr,
        site_branch_id=site_branch_id_arr,
        site_frac=site_frac_arr,
        site_project_dist_um=site_project_dist_um_arr,
        site_support_gci=site_support_gci_arr,
        site_support_weight=site_support_w_arr,
        phi_sites=phi_sites,
        wave_uA=wave,
        electrode_xyz=np.array(ELEC_XYZ_UM), sigma=SIGMA_S_M,
        pulse=np.array([PULSE.amp_uA, PULSE.width_ms, PULSE.delay_ms]),
        dt_ms=DT_MS, t_max_ms=T_MAX_MS, v_init_mv=V_INIT_MV,
        site_proxy=np.array(site_proxy),
        n_comp=n_comp,
        **provenance_fields(
            platform=backend,
            hardware_label=f"{backend} single-device",
            script_path=__file__,
        ),
    )
    print(f"Saved {out_path}  v.shape={v.shape}")


# NEURON side

def _neuron_record_handles(cell):
    """NEURON equivalents of the four Jaxley recording sites.

    BBP template exposes `cell.soma`, `cell.apic`, `cell.dend` (basal),
    `cell.axon` as section lists. We pick index 0 at 0.5 for each.
    """
    return {
        "soma":   cell.soma[0](0.5),
        "apical": cell.apic[0](0.5),
        "basal":  cell.dend[0](0.5),
        "axon":   cell.axon[0](0.5),
    }


def run_neuron(out_path: Path) -> None:
    from neuron import h  # type: ignore[import-not-found]

    print(f"BBP Pyr ECS parity (NEURON side). Loading template from {pyr_cell_dir()}...")
    cell = load_bbp_pyr_cell(h)
    h.celsius = CELSIUS_C

    # Gather every segment across every section. We need xyz for each so we
    # can compute phi_e, and a handle so we can insert extracellular and
    # set e_extracellular.
    all_segs: list = []
    for sec in cell.all:
        sec.insert("extracellular")
        for seg in sec:
            seg.xraxial[0] = 1e9
            seg.xg[0]      = 1e9
            seg.xc[0]      = 0.0
            all_segs.append(seg)

    xyz = np.array([segment_center_xyz(s) for s in all_segs])
    print(f"  {len(all_segs)} segments, xyz range "
          f"x={xyz[:,0].min():.0f}..{xyz[:,0].max():.0f}, "
          f"y={xyz[:,1].min():.0f}..{xyz[:,1].max():.0f}, "
          f"z={xyz[:,2].min():.0f}..{xyz[:,2].max():.0f}")

    t = common.time_grid(DT_MS, T_MAX_MS)
    wave = common.waveform_uA(t, PULSE)
    phi = common.phi_e_per_compartment(
        comp_xyz_um=xyz,
        wave_uA=wave,
        electrode_xyz_um=ELEC_XYZ_UM,
        sigma_S_m=SIGMA_S_M,
    )  # (Nseg, T)

    t_vec = h.Vector(t)
    phi_vecs = []  # keep refs alive for the duration of h.run
    for seg, row in zip(all_segs, phi, strict=False):
        vec = h.Vector(row.astype(np.float64))
        vec.play(seg._ref_e_extracellular, t_vec, True)
        phi_vecs.append(vec)

    handles = _neuron_record_handles(cell)
    rec_vecs = {name: h.Vector().record(seg._ref_v) for name, seg in handles.items()}
    rec_t = h.Vector().record(h._ref_t)

    # Compute xyz + phi_e for each recording site directly. We don't try to
    # match back into all_segs by identity because NEURON's segment objects
    # are not interned. `sec(x)` returns a fresh Python proxy each call,
    # so `id()` comparisons fail. Re-running the pt3d interpolation per
    # handle is cheap and eliminates the lookup.
    site_xyz_list: list[Sequence[float]] = []
    phi_site_rows: list[np.ndarray] = []
    for name in RECORD_SITES:
        seg = handles[name]
        site_xyz = segment_center_xyz(seg)
        site_xyz_list.append(site_xyz)
        # Analytical phi_e at that xyz, same formula as common.phi_e_per_compartment.
        site_xyz_arr_one = np.array(site_xyz)[None, :]
        phi_site_rows.append(common.phi_e_per_compartment(
            comp_xyz_um=site_xyz_arr_one,
            wave_uA=wave,
            electrode_xyz_um=ELEC_XYZ_UM,
            sigma_S_m=SIGMA_S_M,
        )[0])
    site_xyz_arr = np.array(site_xyz_list)
    phi_sites = np.stack(phi_site_rows, axis=0)

    h.cvode_active(0)
    h.secondorder = 0
    h.dt = DT_MS
    h.finitialize(V_INIT_MV)
    h.continuerun(T_MAX_MS)

    t_arr = np.asarray(rec_t)
    v_arr = np.stack([np.asarray(rec_vecs[name]) for name in RECORD_SITES], axis=0)

    OUT_DIR.mkdir(exist_ok=True)
    np.savez(
        out_path, t=t_arr, v=v_arr,
        site_names=np.array(RECORD_SITES),
        site_xyz=site_xyz_arr,
        phi_sites=phi_sites,
        wave_uA=wave,
        electrode_xyz=np.array(ELEC_XYZ_UM), sigma=SIGMA_S_M,
        pulse=np.array([PULSE.amp_uA, PULSE.width_ms, PULSE.delay_ms]),
        dt_ms=DT_MS, t_max_ms=T_MAX_MS, v_init_mv=V_INIT_MV,
        site_proxy=np.array(NEURON_SITE_PROXY),
        n_seg=len(all_segs),
        **provenance_fields(
            platform="cpu",
            hardware_label="cpu single-core (NEURON serial)",
            script_path=__file__,
        ),
    )
    print(f"Saved {out_path}  v.shape={v_arr.shape}")


# Compare

def _print_site_alignment_report(j, n) -> bool:
    """Return True iff saved site coords/fields match tightly enough."""
    if "site_xyz" not in j.files or "site_xyz" not in n.files:
        print("\nSite alignment check skipped: site_xyz missing from one side.")
        return True
    if "phi_sites" not in j.files or "phi_sites" not in n.files:
        print("\nSite alignment check skipped: phi_sites missing from one side.")
        return True

    j_proxy = str(j["site_proxy"]) if "site_proxy" in j.files else "unknown"
    n_proxy = str(n["site_proxy"]) if "site_proxy" in n.files else "unknown"
    peak_idx = int(np.abs(n["wave_uA"]).argmax())

    print("\nSite alignment check:")
    print(f"  Jaxley proxy: {j_proxy}")
    print(f"  NEURON proxy: {n_proxy}")
    print(f"  tolerances : xyz <= {SITE_XYZ_TOL_UM:.1f} um, "
          f"phi rel <= {PHI_SITE_RTOL*100:.1f}% or abs <= {PHI_SITE_ATOL_MV:.2f} mV")
    print(f"\n{'site':<8}{'xyz dist (um)':>16}{'phi_j peak':>14}{'phi_n peak':>14}"
          f"{'phi rel %':>12}{'ok':>6}")

    ok = True
    for name, xyz_j, xyz_n, phi_j, phi_n in zip(
        j["site_names"], j["site_xyz"], n["site_xyz"], j["phi_sites"], n["phi_sites"],
        strict=True,
    ):
        xyz_dist = float(np.linalg.norm(xyz_j - xyz_n))
        phi_j_peak = float(phi_j[peak_idx])
        phi_n_peak = float(phi_n[peak_idx])
        phi_abs = abs(phi_j_peak - phi_n_peak)
        phi_rel = phi_abs / max(abs(phi_n_peak), 1e-12)
        row_ok = xyz_dist <= SITE_XYZ_TOL_UM and (
            phi_abs <= PHI_SITE_ATOL_MV or phi_rel <= PHI_SITE_RTOL
        )
        ok = ok and row_ok
        print(f"{name!s:<8}{xyz_dist:>16.3f}{phi_j_peak:>14.3f}{phi_n_peak:>14.3f}"
              f"{phi_rel*100:>11.2f}%{('yes' if row_ok else 'no'):>6}")
    return ok


def compare(
    jaxley_npz: Path,
    neuron_npz: Path,
    figure_out: Path,
    *,
    allow_site_mismatch: bool = False,
) -> None:
    j = np.load(jaxley_npz)
    n = np.load(neuron_npz)

    # Sanity-check site ordering matches between the two sides.
    j_names = [str(x) for x in j["site_names"].tolist()]
    n_names = [str(x) for x in n["site_names"].tolist()]
    if j_names != n_names:
        raise RuntimeError(f"site ordering mismatch: jaxley={j_names} neuron={n_names}")

    sites_ok = _print_site_alignment_report(j, n)
    if not sites_ok and not allow_site_mismatch:
        raise RuntimeError(
            "saved Jaxley and NEURON site metadata do not describe the same physical "
            "locations closely enough for a clean parity claim. Rerun with corrected "
            "site selection or pass --allow-site-mismatch to inspect legacy outputs"
        )

    metrics = common.compute_metrics(
        t_ref=n["t"], v_ref=n["v"],
        t_other=j["t"], v_other=j["v"],
    )
    common.print_metrics_summary(metrics, label="BBP Pyr ECS")

    # Per-site raw numbers for the table view.
    print("\nPer-site (site / RMSE / MAE / r):")
    for name, rmse, mae, r in zip(
        n_names, metrics.rmse_per_seg, metrics.mae_per_seg, metrics.r_per_seg,
        strict=True,
    ):
        print(f"  {name:8s}  RMSE {rmse:6.3f} mV   MAE {mae:6.3f} mV   r {r:.4f}")

    subtitle = (
        f"BBP cADpyr229 morphology; {PULSE.amp_uA:.0f} uA cathodic pulse "
        f"{PULSE.width_ms:.1f} ms at electrode {ELEC_XYZ_UM[1]:.0f} um above soma; "
        f"RMSE median {np.median(metrics.rmse_per_seg):.3f} mV, "
        f"max {metrics.rmse_per_seg.max():.3f} mV"
    )
    common.plot_ecs_parity_sites(
        metrics=metrics,
        phi_sites=n["phi_sites"],
        site_names=n_names,
        out_path=figure_out,
        title_main="ECS parity on BBP L2/3 Pyr: NEURON (CPU, f64) vs Jaxley (GPU)",
        title_subtitle=subtitle,
        primary_site="soma",
        secondary_site="apical",
    )


# Main

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--side", choices=("jaxley", "neuron", "compare"), required=True)
    p.add_argument("--figure-out", default=str(FIGURE_OUT))
    p.add_argument(
        "--allow-site-mismatch",
        action="store_true",
        help="Allow compare to proceed even when saved site coords/phi metadata disagree.",
    )
    args = p.parse_args()

    jaxley_npz = OUT_DIR / "parity_bbp_ecs_jaxley.npz"
    neuron_npz = OUT_DIR / "parity_bbp_ecs_neuron.npz"

    if args.side == "jaxley":
        run_jaxley(jaxley_npz, site_targets_npz=neuron_npz if neuron_npz.exists() else None)
    elif args.side == "neuron":
        run_neuron(neuron_npz)
    else:
        if not jaxley_npz.exists() or not neuron_npz.exists():
            print(f"Missing {jaxley_npz} or {neuron_npz}; run --side jaxley and "
                  "--side neuron first.", file=sys.stderr)
            sys.exit(2)
        compare(
            jaxley_npz,
            neuron_npz,
            Path(args.figure_out),
            allow_site_mismatch=args.allow_site_mismatch,
        )


if __name__ == "__main__":
    main()
