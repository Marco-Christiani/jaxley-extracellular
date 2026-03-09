#!/usr/bin/env python
"""Pilot sweep: strength-duration curves for monophasic and biphasic pulses.

Sweeps pulse width x polarity x waveform shape (mono/biphasic), finding
activation threshold via vectorised binary search at each condition.
Saves results as .npz for downstream analysis.

Usage:
    python scripts/pilot_sweep.py [--outdir results]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from jaxley_extracellular.extracellular.experiment import make_hh_cable_experiment


# -----------------------------------------------------------------------
# Sweep parameters
# -----------------------------------------------------------------------

PULSE_WIDTHS_MS = np.array([0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0])
ELECTRODE_DISTANCES_UM = np.array([50.0])  # can extend later
WAVEFORM_TYPES = ["monophasic_cathodic", "monophasic_anodic", "biphasic_cathodic_first"]

# Binary search
AMP_LO = 0.0
AMP_HI = 5000.0   # uA -- wide bracket to cover all conditions
N_ITER = 14        # precision: 5000/2^14 ~= 0.3 uA

# Model
NCOMP = 50
CABLE_LENGTH_UM = 1250.0
RADIUS_UM = 10.0
DT_MS = 0.025
T_MS = 5.0
SIGMA = 0.3
RECORD_COMP = 0


# -----------------------------------------------------------------------
# Waveform factories (must be vmap-compatible)
# -----------------------------------------------------------------------

def _make_mono_cathodic(amplitude, pw_steps, T):
    return jnp.zeros((T,)).at[:pw_steps].set(-amplitude)

def _make_mono_anodic(amplitude, pw_steps, T):
    return jnp.zeros((T,)).at[:pw_steps].set(amplitude)

def _make_biphasic_cathodic_first(amplitude, pw_steps, T):
    w = jnp.zeros((T,))
    w = w.at[:pw_steps].set(-amplitude)
    w = w.at[pw_steps:2 * pw_steps].set(amplitude)
    return w


WAVEFORM_FACTORIES = {
    "monophasic_cathodic": _make_mono_cathodic,
    "monophasic_anodic": _make_mono_anodic,
    "biphasic_cathodic_first": _make_biphasic_cathodic_first,
}


# -----------------------------------------------------------------------
# Main sweep
# -----------------------------------------------------------------------

def run_sweep(outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)

    total_configs = (
        len(PULSE_WIDTHS_MS) * len(WAVEFORM_TYPES) * len(ELECTRODE_DISTANCES_UM)
    )
    print(f"Pilot sweep: {total_configs} configs")
    print(f"  pulse widths: {PULSE_WIDTHS_MS} ms")
    print(f"  waveform types: {WAVEFORM_TYPES}")
    print(f"  electrode distances: {ELECTRODE_DISTANCES_UM} um")
    print(f"  binary search: {N_ITER} iterations, bracket [{AMP_LO}, {AMP_HI}] uA")
    print()

    T = int(T_MS / DT_MS)
    results = []

    for dist_um in ELECTRODE_DISTANCES_UM:
        exp = make_hh_cable_experiment(
            ncomp=NCOMP,
            cable_length_um=CABLE_LENGTH_UM,
            radius_um=RADIUS_UM,
            electrode_distance_um=float(dist_um),
            sigma=SIGMA,
            dt_ms=DT_MS,
            T_ms=T_MS,
        )

        for wtype in WAVEFORM_TYPES:
            factory = WAVEFORM_FACTORIES[wtype]

            for pw_ms in PULSE_WIDTHS_MS:
                pw_steps = int(pw_ms / DT_MS)

                def make_wf(amplitude, _pw=pw_steps, _T=T, _fac=factory):
                    return _fac(amplitude, _pw, _T)

                lo = jnp.array([AMP_LO])
                hi = jnp.array([AMP_HI])

                t0 = time.time()
                thr = exp.find_thresholds(
                    make_wf, lo, hi,
                    n_iter=N_ITER,
                    record_comp=RECORD_COMP,
                )
                elapsed = time.time() - t0
                thr_val = float(thr[0])
                charge_nC = thr_val * pw_ms

                row = {
                    "waveform_type": wtype,
                    "pulse_width_ms": float(pw_ms),
                    "electrode_distance_um": float(dist_um),
                    "threshold_uA": thr_val,
                    "charge_nC": charge_nC,
                    "time_s": elapsed,
                }
                results.append(row)
                print(
                    f"  {wtype:30s}  pw={pw_ms:.2f}ms  "
                    f"thr={thr_val:8.1f}uA  Q={charge_nC:8.1f}nC  "
                    f"({elapsed:.1f}s)"
                )
            print()

    # Save results
    out = {k: np.array([r[k] for r in results]) for k in results[0]}
    # String arrays need object dtype
    out["waveform_type"] = np.array([r["waveform_type"] for r in results])
    outpath = outdir / "pilot_sweep.npz"
    np.savez(outpath, **out)
    print(f"\nSaved {len(results)} results to {outpath}")

    # Print summary table
    print("\n" + "=" * 72)
    print("Strength-Duration Summary")
    print("=" * 72)
    header = f"{'Type':>30s}  {'pw(ms)':>7s}  {'I_thr(uA)':>10s}  {'Q(nC)':>8s}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['waveform_type']:>30s}  {r['pulse_width_ms']:>7.2f}  "
            f"{r['threshold_uA']:>10.1f}  {r['charge_nC']:>8.1f}"
        )

    total_time = sum(r["time_s"] for r in results)
    print(f"\nTotal wall time: {total_time:.0f}s ({total_time/60:.1f}min)")
    print(f"Throughput: {len(results)/total_time:.1f} configs/s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pilot ECS waveform sweep")
    parser.add_argument("--outdir", type=str, default="results")
    args = parser.parse_args()
    run_sweep(Path(args.outdir))
