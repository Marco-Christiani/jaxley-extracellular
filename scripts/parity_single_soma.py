"""Single-soma parity test: Jaxley vs NEURON.

Builds a 1-compartment cylindrical soma with matched geometry in both
simulators, inserts the BBP Pyr (or PV) soma channel set at matched densities,
and compares v(t) under sub- and supra-threshold step currents.

This isolates the channel/integration implementation from morphology loading,
axial coupling, and BBP surface-area scaling.

Prerequisite: run ``scripts/neuron_single_soma.py`` first to produce
``results/neuron_single_soma_{pyr,pv}.npz``.

Run:
    python scripts/parity_single_soma.py
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import cast

import jax
import jaxley as jx
import numpy as np
from jaxley_mech.channels.hodgkin52 import Leak
from jaxley_mech.channels.l5pc import (
    SKE2,
    CaHVA,
    CaLVA,
    CaNernstReversal,
    CaPump,
    H,
    KPst,
    KTst,
    M,
    NapEt2,
    NaTs2T,
    SKv3_1,
)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# Must match scripts/neuron_single_soma.py
L = 20.0
RADIUS = 10.0     # um (= diam/2)
RA = 100.0
CM = 1.0

DT = 0.025
V_INIT = -75.0
T_MAX = 700.0

I_DELAY = 100.0
I_DUR = 500.0
I_SUB = 0.05
I_SUP = 0.5


PYR_SOMA = {
    "Leak_gLeak": 3e-5, "Leak_eLeak": -75.0,
    "H_gH": 0.000080, "eH": -45.0,
    "NaTs2T_gNaTs2T": 0.926705, "eNa": 50.0,
    "SKv3_1_gSKv3_1": 0.102517, "eK": -85.0,
    "SKE2_gSKE2": 0.099433,
    "CaHVA_gCaHVA": 0.000374,
    "CaLVA_gCaLVA": 0.000778,
    "CaPump_gamma": 0.000533, "CaPump_decay": 342.544232,
}

PV_SOMA = {
    "Leak_gLeak": 0.000091, "Leak_eLeak": -62.442793,
    "NaTs2T_gNaTs2T": 0.197999, "eNa": 50.0,
    "SKv3_1_gSKv3_1": 0.297559, "eK": -85.0,
    "NapEt2_gNapEt2": 0.000001,
    "M_gM": 0.000008,
    "KPst_gKPst": 0.156376,
    "KTst_gKTst": 0.092965,
    "SKE2_gSKE2": 0.019726,
    "CaHVA_gCaHVA": 0.000032,
    "CaLVA_gCaLVA": 0.001067,
    "CaPump_gamma": 0.000511, "CaPump_decay": 731.707637,
}


def make_cell(kind: str) -> jx.Cell:
    comp = jx.Compartment()
    branch = jx.Branch(comp, ncomp=1)
    cell = jx.Cell([branch], parents=[-1])

    cell.set("length", L)
    cell.set("radius", RADIUS)
    cell.set("axial_resistivity", RA)
    cell.set("capacitance", CM)

    cell.insert(Leak())
    # Ca chain order: CaNernstReversal -> CaHVA/CaLVA -> CaPump -> SKE2
    cell.insert(CaNernstReversal())
    cell.insert(CaHVA())
    cell.insert(CaLVA())
    cell.insert(CaPump())
    cell.insert(SKE2())
    cell.insert(NaTs2T())
    cell.insert(SKv3_1())

    if kind == "pyr":
        cell.insert(H())
        params = PYR_SOMA
    elif kind == "pv":
        cell.insert(NapEt2())
        cell.insert(M())
        cell.insert(KPst())
        cell.insert(KTst())
        params = PV_SOMA
    else:
        raise ValueError(kind)

    for key, val in params.items():
        cell.set(key, val)

    cell.init_states()
    return cell


def run_jaxley(kind: str, i_amp: float) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Build, warm up (JIT compile), then time a fresh integrate.

    Returns (t, v, compile_plus_run_s, cached_run_s).
    """
    cell = make_cell(kind)
    cell.set("v", V_INIT)
    cell.record("v")
    cell.stimulate(jx.step_current(I_DELAY, I_DUR, i_amp, DT, T_MAX))

    t0 = time.perf_counter()
    v = jx.integrate(cell, delta_t=DT, t_max=T_MAX)
    jax.block_until_ready(v)
    t_warm = time.perf_counter() - t0

    t0 = time.perf_counter()
    v = jx.integrate(cell, delta_t=DT, t_max=T_MAX)
    jax.block_until_ready(v)
    t_run = time.perf_counter() - t0

    t = np.linspace(0.0, T_MAX, v.shape[1])
    return t, np.array(v[0]), t_warm, t_run


def summary(t: np.ndarray, v: np.ndarray, i_delay: float) -> dict[str, float]:
    from scipy.signal import find_peaks  # type: ignore[import-untyped]
    peaks = cast(np.ndarray, find_peaks(v, height=-25.0, prominence=40.0)[0])
    pre = t < (i_delay - 10)
    return {
        "v_rest": float(v[pre].mean()) if pre.any() else float("nan"),
        "v_stim_last_100ms": float(v[(t > i_delay + 400) & (t < i_delay + 500)].mean()),
        "n_spikes": float(len(peaks)),
        "first_latency_ms": float(t[peaks[0]] - i_delay) if len(peaks) else float("nan"),
        "ap_peak_mV": float(v[peaks].max()) if len(peaks) else float("nan"),
        "v_min_post": float(v[t > i_delay + I_DUR].min()),
    }


def compare(kind: str) -> None:
    neuron_path = RESULTS_DIR / f"neuron_single_soma_{kind}.npz"
    if not neuron_path.exists():
        print(f"SKIP {kind}: {neuron_path} not found. Run scripts/neuron_single_soma.py first.")
        return

    n = np.load(neuron_path)
    t_n, v_n_sub, v_n_sup = n["t"], n["v_sub"], n["v_sup"]
    run_s_n_sub = float(n["run_s_sub"]) if "run_s_sub" in n.files else float("nan")
    run_s_n_sup = float(n["run_s_sup"]) if "run_s_sup" in n.files else float("nan")

    print(f"\n=== {kind.upper()} single-soma parity ===")

    print(f"\nRunning Jaxley {kind} subthreshold (I={I_SUB} nA)...")
    t_j, v_j_sub, warm_sub, run_sub = run_jaxley(kind, I_SUB)

    print(f"Running Jaxley {kind} suprathreshold (I={I_SUP} nA)...")
    _,   v_j_sup, warm_sup, run_sup = run_jaxley(kind, I_SUP)

    print(f"\n--- wall-clock (integrate {T_MAX} ms at dt={DT} ms) ---")
    print(f"{'stage':<14}{'NEURON':>12}{'Jaxley warm':>15}{'Jaxley cached':>16}")
    print("-" * 57)
    print(f"{'subthresh':<14}{run_s_n_sub*1e3:>10.1f} ms{warm_sub*1e3:>13.1f} ms{run_sub*1e3:>14.1f} ms")
    print(f"{'suprathresh':<14}{run_s_n_sup*1e3:>10.1f} ms{warm_sup*1e3:>13.1f} ms{run_sup*1e3:>14.1f} ms")

    for tag, v_n, v_j in [("subthresh", v_n_sub, v_j_sub),
                          ("suprathresh", v_n_sup, v_j_sup)]:
        m_n = summary(t_n, v_n, I_DELAY)
        m_j = summary(t_j, v_j, I_DELAY)

        print(f"\n--- {tag} ---")
        print(f"{'metric':<22}{'NEURON':>12}{'Jaxley':>12}{'diff':>12}")
        print("-" * 58)
        for key in m_n:
            vn, vj = m_n[key], m_j[key]
            diff = (vj - vn) if (not np.isnan(vn) and not np.isnan(vj)) else float("nan")
            print(f"{key:<22}{vn:>12.4f}{vj:>12.4f}{diff:>+12.4f}")

    out = RESULTS_DIR / f"parity_single_soma_{kind}.npz"
    np.savez(out,
             t_neuron=t_n, v_neuron_sub=v_n_sub, v_neuron_sup=v_n_sup,
             t_jaxley=t_j, v_jaxley_sub=v_j_sub, v_jaxley_sup=v_j_sup)
    print(f"\nTraces saved to {out}")


if __name__ == "__main__":
    compare("pyr")
    compare("pv")
