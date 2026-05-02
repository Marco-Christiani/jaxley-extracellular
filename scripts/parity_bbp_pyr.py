"""BBP L2/3 Pyr multi-compartment parity: Jaxley vs NEURON.

Runs the same step-current protocol through both simulators on the same BBP
cADpyr229 cell (Jaxley uses make_pyr_cell from the SWC; NEURON uses the BBP
HOC template with the .asc morphology). Saves v(t) at soma(0.5) for each.

This has TWO stages because NEURON runs in the dedicated `.#neuron` shell:

Stage 1 (NEURON side), writes results/parity_bbp_pyr_neuron.npz
    nix develop .#neuron
    cd reference/bbp/simulation && nrnivmodl mechanisms
    cd /path/to/jaxley-extracellular
    neuron-python -m scripts.parity_bbp_pyr --side neuron

Stage 2 (Jaxley side), writes results/parity_bbp_pyr_jaxley.npz
    python -m scripts.parity_bbp_pyr --side jaxley

Stage 3 (compare), prints comparison table
    python -m scripts.parity_bbp_pyr --side compare
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, cast

import numpy as np

from scripts.neuron_bbp_common import load_bbp_pyr_cell
from scripts.parity_common import (
    interp_xyz_on_polyline,
    interpolate_branch_voltage,
    provenance_fields,
    segment_center_xyz,
    waveform_metrics,
)

OUT_DIR = Path(__file__).resolve().parent.parent / "results"

# Protocol (matched)
DT = 0.025
V_INIT = -75.0
T_MAX = 300.0
CELSIUS = 34.0

I_DELAY = 50.0
I_DUR = 100.0
I_AMPS = (0.1, 0.5)  # sub and supra

JAXLEY_SITE_PROXY = "interpolated_soma_branch0_midpoint"
NEURON_SITE_PROXY = "section_midpoint_group0_x_0p5"


def run_neuron_all(amps: tuple[float, ...]) -> tuple[np.ndarray, list[np.ndarray]]:
    """Load the BBP template once, run each amp, return (t, [v_per_amp])."""
    from neuron import h  # type: ignore[import-not-found]
    cell = load_bbp_pyr_cell(h)  # pyright: ignore[reportUnknownVariableType]

    h.celsius = CELSIUS
    h.dt = DT

    traces: list[np.ndarray] = []
    t_arr: np.ndarray | None = None
    site_xyz = segment_center_xyz(cell.soma[0](0.5))  # pyright: ignore[reportUnknownVariableType]
    for amp in amps:
        iclamp = h.IClamp(cell.soma[0](0.5))  # pyright: ignore[reportUnknownVariableType]
        iclamp.delay = I_DELAY
        iclamp.dur = I_DUR
        iclamp.amp = amp

        v_vec = h.Vector().record(cell.soma[0](0.5)._ref_v)  # pyright: ignore[reportUnknownVariableType]
        t_vec = h.Vector().record(h._ref_t)  # pyright: ignore[reportUnknownVariableType]

        h.finitialize(V_INIT)
        h.continuerun(T_MAX)
        traces.append(np.array(v_vec))
        t_arr = np.array(t_vec)
    assert t_arr is not None
    return t_arr, traces, site_xyz


def run_jaxley(
    amp: float,
    ncomp: int,
    max_branch_len: float | None,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray | float | str]]:
    import jax
    import jaxley as jx

    from jaxley_extracellular.bbp.cell_factory import make_pyr_cell

    cell = make_pyr_cell(ncomp=ncomp, max_branch_len=max_branch_len)
    cell.set("v", V_INIT)
    cell.record("v")
    soma_branch = cell.soma.branch(0)  # pyright: ignore[reportOptionalMemberAccess]
    soma_branch.loc(0.5).stimulate(jx.step_current(I_DELAY, I_DUR, amp, DT, T_MAX))

    v_full = jx.integrate(cell, delta_t=DT, t_max=T_MAX)
    jax.block_until_ready(v_full)  # type: ignore[no-untyped-call]
    v_full_np = np.array(v_full)

    soma_poly_xyz = soma_branch.xyzr[0][:, :3].astype(float)
    soma_xyz = interp_xyz_on_polyline(soma_poly_xyz, 0.5)
    v_soma, support_gci, support_w = interpolate_branch_voltage(v_full_np, soma_branch, 0.5)

    meta: dict[str, np.ndarray | float | str] = {
        "site_xyz": soma_xyz,
        "site_frac": 0.5,
        "site_support_gci": support_gci,
        "site_support_weight": support_w,
        "site_proxy": JAXLEY_SITE_PROXY,
    }
    t = np.linspace(0.0, T_MAX, v_full_np.shape[1])
    return t, np.array(v_soma), meta


def summary(t: np.ndarray, v: np.ndarray) -> dict[str, float]:
    from scipy.signal import find_peaks
    peaks = cast(np.ndarray, find_peaks(v, height=-25.0, prominence=40.0)[0])
    pre = t < (I_DELAY - 5)
    during = (t > I_DELAY) & (t < I_DELAY + I_DUR)
    return {
        "v_rest": float(v[pre].mean()) if pre.any() else float("nan"),
        "v_stim_mean": float(v[during].mean()) if during.any() else float("nan"),
        "n_spikes": float(len(peaks)),
        "first_latency_ms": float(t[peaks[0]] - I_DELAY) if len(peaks) else float("nan"),
        "ap_peak_mV": float(v[peaks].max()) if len(peaks) else float("nan"),
    }


def cmd_side(side: str, ncomp: int, max_branch_len: float | None) -> None:
    OUT_DIR.mkdir(exist_ok=True)
    out = OUT_DIR / f"parity_bbp_pyr_{side}.npz"
    data: dict[str, Any] = {"amps": np.array(I_AMPS)}

    if side == "neuron":
        t, vs, site_xyz = run_neuron_all(I_AMPS)
        data["t"] = t
        data["site_xyz"] = site_xyz
        data["site_frac"] = 0.5
        data["site_proxy"] = NEURON_SITE_PROXY
        for amp, v in zip(I_AMPS, vs, strict=True):
            data[f"v_{amp}"] = v
    else:
        for amp in I_AMPS:
            print(f"  jaxley amp={amp} nA (ncomp={ncomp}, max_branch_len={max_branch_len}) ...")
            t, v, meta = run_jaxley(amp, ncomp, max_branch_len)
            if "t" not in data:
                data["t"] = t
                data["site_xyz"] = cast(np.ndarray, meta["site_xyz"])
                data["site_frac"] = float(meta["site_frac"])
                data["site_support_gci"] = cast(np.ndarray, meta["site_support_gci"])
                data["site_support_weight"] = cast(np.ndarray, meta["site_support_weight"])
                data["site_proxy"] = str(meta["site_proxy"])
            data[f"v_{amp}"] = v

    if side == "neuron":
        prov = provenance_fields(
            platform="cpu",
            hardware_label="cpu single-core (NEURON serial)",
            script_path=__file__,
        )
    else:
        import jax
        backend = jax.default_backend()
        prov = provenance_fields(
            platform=backend,
            hardware_label=f"{backend} single-device",
            script_path=__file__,
        )
    data.update(prov)
    np.savez(out, **data)
    print(f"Saved {out}")


def cmd_compare() -> None:
    n_path = OUT_DIR / "parity_bbp_pyr_neuron.npz"
    j_path = OUT_DIR / "parity_bbp_pyr_jaxley.npz"
    if not n_path.exists() or not j_path.exists():
        print("Missing: run --side neuron and --side jaxley first.")
        return
    n = np.load(n_path)
    j = np.load(j_path)
    t_n = n["t"]
    t_j = j["t"]

    if "site_xyz" in n.files and "site_xyz" in j.files:
        xyz_dist = float(np.linalg.norm(np.asarray(j["site_xyz"]) - np.asarray(n["site_xyz"])))
        j_proxy = str(j["site_proxy"]) if "site_proxy" in j.files else "unknown"
        n_proxy = str(n["site_proxy"]) if "site_proxy" in n.files else "unknown"
        print("site alignment:")
        print(f"  Jaxley proxy: {j_proxy}")
        print(f"  NEURON proxy: {n_proxy}")
        print(f"  xyz distance : {xyz_dist:.6f} um\n")

    print("\n=== BBP L2/3 Pyr multi-comp parity ===\n")
    for amp in I_AMPS:
        v_n = n[f"v_{amp}"]
        v_j = j[f"v_{amp}"]
        m_n = summary(t_n, v_n)
        m_j = summary(t_j, v_j)
        wm = waveform_metrics(t_n, v_n, t_j, v_j)

        print(f"--- amp = {amp} nA ---")
        print(f"{'metric':<22}{'NEURON':>12}{'Jaxley':>12}{'diff':>12}")
        print("-" * 58)
        for k in m_n:
            vn, vj = m_n[k], m_j[k]
            d = (vj - vn) if (not np.isnan(vn) and not np.isnan(vj)) else float("nan")
            print(f"{k:<22}{vn:>12.4f}{vj:>12.4f}{d:>+12.4f}")
        print("\n  waveform-level:")
        print(f"    RMSE           = {wm['rmse_mV']:.3f} mV")
        print(f"    MAE            = {wm['mae_mV']:.3f} mV")
        print(f"    max |diff|     = {wm['max_abs_mV']:.3f} mV")
        print(f"    Pearson r      = {wm['pearson_r']:.6f}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", choices=["neuron", "jaxley", "compare"], required=True)
    parser.add_argument("--ncomp", type=int, default=2)
    parser.add_argument("--max-branch-len", type=float, default=100.0)
    args = parser.parse_args()
    if args.side == "compare":
        cmd_compare()
    else:
        cmd_side(args.side, args.ncomp, args.max_branch_len)


if __name__ == "__main__":
    main()
