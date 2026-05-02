"""NEURON vs Jaxley ECS parity on a straight HH cable.

Usage:
    # Jaxley side (default env)
    python -m scripts.parity_ecs.hh --side jaxley

    # NEURON side (dedicated NEURON shell)
    nix develop .#neuron
    neuron-python -m scripts.parity_ecs.hh --side neuron

    # Compare + plot (default env, requires both npzs on disk)
    python -m scripts.parity_ecs.hh --side compare

Outputs:
    results/parity_hh_ecs_{jaxley,neuron}.npz
    slides.local/figures/ecs_parity_hh.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from scripts.parity_ecs import common

OUT_DIR = Path(__file__).resolve().parent.parent.parent / "results"
FIGURE_OUT = Path(__file__).resolve().parent.parent.parent \
    / "slides.local" / "figures" / "ecs_parity_hh.png"


# --- shared geometry / stim / integration -----------------------------------
L_UM          = 1250.0
NSEG          = 50
DIAM_UM       = 20.0
RA_OHM_CM     = 100.0
CM_UF_CM2     = 1.0
GNABAR_S_CM2  = 0.12
GKBAR_S_CM2   = 0.036
GLEAK_S_CM2   = 0.0003
E_NA_MV       = 50.0
E_K_MV        = -77.0
E_LEAK_MV     = -54.3
V_INIT_MV     = -65.0

ELEC_XYZ_UM   = (625.0, 100.0, 0.0)
SIGMA_S_M     = 0.3

DT_MS         = 0.025
T_MAX_MS      = 10.0
PULSE         = common.PulseSpec(amp_uA=-100.0, width_ms=1.0, delay_ms=2.0)


def segment_centers_um() -> np.ndarray:
    """x-coordinates of the NSEG segment centres along the straight cable."""
    edge = np.linspace(0.0, L_UM, NSEG + 1)
    return 0.5 * (edge[:-1] + edge[1:])


# Jaxley side

def run_jaxley(out_path: Path) -> None:
    import os
    os.environ.setdefault("JAX_ENABLE_X64", "1")

    import jax
    import jax.numpy as jnp
    import jaxley as jx
    from jaxley.channels import HH

    from jaxley_extracellular.extracellular.discretization import build_voltage_operator_G
    from jaxley_extracellular.extracellular.equivalent_current import phi_e_to_ecs_nA
    from jaxley_extracellular.extracellular.field import point_source_potential

    print("HH ECS parity (Jaxley side). jax:", jax.__version__,
          "backend:", jax.default_backend())

    comp = jx.Compartment()
    branch = jx.Branch(comp, ncomp=NSEG)
    branch.set("length", L_UM / NSEG)
    branch.set("radius", DIAM_UM / 2.0)
    branch.set("axial_resistivity", RA_OHM_CM)
    branch.set("capacitance", CM_UF_CM2)
    branch.set("v", V_INIT_MV)
    branch.insert(HH())

    branch.xyzr[0] = np.array(
        [[0.0, 0.0, 0.0, DIAM_UM / 2.0],
         [L_UM, 0.0, 0.0, DIAM_UM / 2.0]]
    )
    branch.compute_compartment_centers()
    branch.init_states()
    branch.to_jax()

    for i in range(NSEG):
        branch.comp(i).record(verbose=False)

    comp_xyz = branch.nodes[["x", "y", "z"]].to_numpy()
    t = common.time_grid(DT_MS, T_MAX_MS)
    wave = common.waveform_uA(t, PULSE)

    # Use our project's point_source_potential for the field (exercises the
    # real JAX-traceable code path, not the analytical helper in common.py).
    phi_e_mV = np.asarray(point_source_potential(
        comp_xyz=comp_xyz,
        electrode_positions=jnp.asarray(ELEC_XYZ_UM)[None, :],
        electrode_currents=jnp.asarray(wave)[None, :],
        sigma=SIGMA_S_M,
    ))

    # Run the equivalent-current conversion through the public API.
    params = branch.get_all_parameters(pstate=[])
    G = build_voltage_operator_G(branch, params)
    cm = jnp.asarray(branch.nodes["capacitance"].to_numpy())
    area = jnp.asarray(branch.nodes["area"].to_numpy())
    i_ecs_nA = np.asarray(phi_e_to_ecs_nA(jnp.asarray(phi_e_mV), G, cm, area))

    data_stimuli = None
    for i in range(NSEG):
        data_stimuli = branch.comp(i).data_stimulate(
            jnp.asarray(i_ecs_nA[i]), data_stimuli=data_stimuli, verbose=False,
        )

    v = np.asarray(jx.integrate(
        branch, delta_t=DT_MS, t_max=T_MAX_MS, data_stimuli=data_stimuli,
    ))
    t_out = np.arange(v.shape[1]) * DT_MS

    OUT_DIR.mkdir(exist_ok=True)
    np.savez(
        out_path, t=t_out, v=v,
        phi_e=phi_e_mV, i_ecs_nA=i_ecs_nA, wave_uA=wave,
        seg_centers=segment_centers_um(),
        electrode_xyz=np.array(ELEC_XYZ_UM), sigma=SIGMA_S_M,
        pulse=np.array([PULSE.amp_uA, PULSE.width_ms, PULSE.delay_ms]),
        dt_ms=DT_MS, t_max_ms=T_MAX_MS, v_init_mv=V_INIT_MV,
        geometry=np.array([L_UM, NSEG, DIAM_UM, RA_OHM_CM, CM_UF_CM2]),
    )
    print(f"Saved {out_path}  v.shape={v.shape}")


# NEURON side

def run_neuron(out_path: Path) -> None:
    from neuron import h  # type: ignore[import-not-found]

    h.load_file("stdrun.hoc")
    print("HH ECS parity (NEURON side).", h.nrnversion(1))

    cable = h.Section(name="cable")
    cable.L = L_UM
    cable.nseg = NSEG
    cable.diam = DIAM_UM
    cable.Ra = RA_OHM_CM
    cable.cm = CM_UF_CM2

    cable.insert("hh")
    for seg in cable:
        seg.hh.gnabar = GNABAR_S_CM2
        seg.hh.gkbar  = GKBAR_S_CM2
        seg.hh.gl     = GLEAK_S_CM2
        seg.hh.el     = E_LEAK_MV
        seg.ena = E_NA_MV
        seg.ek  = E_K_MV

    # Standard single-shell extracellular setup: imposed e_e, no periaxial.
    cable.insert("extracellular")
    for seg in cable:
        seg.xraxial[0] = 1e9
        seg.xg[0]      = 1e9
        seg.xc[0]      = 0.0

    centers = segment_centers_um()
    comp_xyz = np.stack([centers, np.zeros_like(centers), np.zeros_like(centers)],
                        axis=1)
    t = common.time_grid(DT_MS, T_MAX_MS)
    wave = common.waveform_uA(t, PULSE)
    phi = common.phi_e_per_compartment(
        comp_xyz_um=comp_xyz,
        wave_uA=wave,
        electrode_xyz_um=ELEC_XYZ_UM,
        sigma_S_m=SIGMA_S_M,
    )

    t_vec = h.Vector(t)
    phi_vecs = []  # keep refs alive
    for seg, row in zip(cable, phi, strict=False):
        v = h.Vector(row.astype(np.float64))
        v.play(seg._ref_e_extracellular, t_vec, True)
        phi_vecs.append(v)

    rec_vecs = [h.Vector().record(seg._ref_v) for seg in cable]
    rec_t = h.Vector().record(h._ref_t)

    h.cvode_active(0)
    h.secondorder = 0
    h.dt = DT_MS
    h.finitialize(V_INIT_MV)
    h.continuerun(T_MAX_MS)

    t_arr = np.asarray(rec_t)
    v_arr = np.stack([np.asarray(v) for v in rec_vecs], axis=0)

    OUT_DIR.mkdir(exist_ok=True)
    np.savez(
        out_path, t=t_arr, v=v_arr,
        phi_e=phi, wave_uA=wave, seg_centers=centers,
        electrode_xyz=np.array(ELEC_XYZ_UM), sigma=SIGMA_S_M,
        pulse=np.array([PULSE.amp_uA, PULSE.width_ms, PULSE.delay_ms]),
        dt_ms=DT_MS, t_max_ms=T_MAX_MS, v_init_mv=V_INIT_MV,
        geometry=np.array([L_UM, NSEG, DIAM_UM, RA_OHM_CM, CM_UF_CM2]),
    )
    print(f"Saved {out_path}  v.shape={v_arr.shape}")


# Compare

def compare(jaxley_npz: Path, neuron_npz: Path, figure_out: Path) -> None:
    j = np.load(jaxley_npz)
    n = np.load(neuron_npz)

    metrics = common.compute_metrics(
        t_ref=n["t"], v_ref=n["v"],
        t_other=j["t"], v_other=j["v"],
    )
    common.print_metrics_summary(metrics, label="HH ECS")

    subtitle = (
        f"{int(n['geometry'][1])}-seg cable, "
        f"{PULSE.amp_uA:.0f} uA cathodic pulse {PULSE.width_ms:.1f} ms at electrode "
        f"{ELEC_XYZ_UM[1]:.0f} um above midpoint, sigma = {SIGMA_S_M} S/m; "
        f"RMSE median {np.median(metrics.rmse_per_seg):.3f} mV, "
        f"max {metrics.rmse_per_seg.max():.3f} mV"
    )
    common.plot_ecs_parity(
        metrics=metrics,
        phi_e=n["phi_e"],
        seg_positions=n["seg_centers"],
        seg_position_label="segment x (um)",
        pulse=PULSE,
        electrode_xyz_um=ELEC_XYZ_UM,
        sigma_S_m=SIGMA_S_M,
        dt_ms=DT_MS,
        out_path=figure_out,
        title_main="ECS parity on an HH cable: NEURON (CPU, f64) vs Jaxley (GPU/f32)",
        title_subtitle=subtitle,
    )


# Main

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--side", choices=("jaxley", "neuron", "compare"), required=True)
    p.add_argument("--figure-out", default=str(FIGURE_OUT))
    args = p.parse_args()

    jaxley_npz = OUT_DIR / "parity_hh_ecs_jaxley.npz"
    neuron_npz = OUT_DIR / "parity_hh_ecs_neuron.npz"

    if args.side == "jaxley":
        run_jaxley(jaxley_npz)
    elif args.side == "neuron":
        run_neuron(neuron_npz)
    else:
        if not jaxley_npz.exists() or not neuron_npz.exists():
            print(f"Missing {jaxley_npz} or {neuron_npz}; run --side jaxley and "
                  "--side neuron first.", file=sys.stderr)
            sys.exit(2)
        compare(jaxley_npz, neuron_npz, Path(args.figure_out))


if __name__ == "__main__":
    main()
