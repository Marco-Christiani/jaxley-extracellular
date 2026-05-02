"""NEURON throughput: N sequential single-soma sims over N stimulus amplitudes.

Uses the same geometry + channel set as ``scripts/neuron_single_soma.py`` (Pyr
soma), varying only ``iclamp.amp``. Reports wall time per N.

Run from the dedicated NEURON shell:

    nix develop .#neuron
    cd reference/bbp/simulation && nrnivmodl mechanisms
    cd /path/to/jaxley-extracellular
    neuron-python -m scripts.throughput_neuron

Writes results/throughput_neuron.npz with keys n_values, wall_s, per_sim_s.
"""

import argparse
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from scripts.neuron_bbp_common import bbp_sim_dir

NEURON_DIR = bbp_sim_dir()
OUT_DIR = Path(__file__).resolve().parent.parent / "results"

# Match parity_single_soma.py (Pyr) exactly
L = 20.0
DIAM = 20.0
RA = 100.0
CM = 1.0

DT = 0.025
V_INIT = -75.0
T_MAX = 700.0
CELSIUS = 34.0

I_DELAY = 100.0
I_DUR = 500.0

ENA = 50.0
EK = -85.0

PYR_SOMA = {
    "pas":           [("g_pas", 3e-5), ("e_pas", -75.0)],
    "Ih":            [("gIhbar_Ih", 0.000080)],
    "NaTs2_t":       [("gNaTs2_tbar_NaTs2_t", 0.926705)],
    "SKv3_1":        [("gSKv3_1bar_SKv3_1", 0.102517)],
    "SK_E2":         [("gSK_E2bar_SK_E2", 0.099433)],
    "Ca_HVA":        [("gCa_HVAbar_Ca_HVA", 0.000374)],
    "Ca_LVAst":      [("gCa_LVAstbar_Ca_LVAst", 0.000778)],
    "CaDynamics_E2": [("gamma_CaDynamics_E2", 0.000533),
                      ("decay_CaDynamics_E2", 342.544232)],
}


def run_one(h: Any, amp: float) -> float:
    """Build a fresh section, run one step-current protocol, return wall seconds."""
    sec = h.Section(name="soma")
    sec.L = L
    sec.diam = DIAM
    sec.Ra = RA
    sec.cm = CM
    sec.nseg = 1

    for mech_name, params in PYR_SOMA.items():
        sec.insert(mech_name)
        for pname, pval in params:
            setattr(sec, pname, pval)

    sec.ena = ENA
    sec.ek = EK

    iclamp = h.IClamp(sec(0.5))
    iclamp.delay = I_DELAY
    iclamp.dur = I_DUR
    iclamp.amp = amp

    h.celsius = CELSIUS
    h.dt = DT
    h.finitialize(V_INIT)

    t0 = time.perf_counter()
    h.continuerun(T_MAX)
    return time.perf_counter() - t0


def bench(h: Any, n: int) -> tuple[float, float]:
    """Run N sims with amps uniformly in [0.05, 0.5] nA. Return (wall_s, per_sim_s)."""
    amps = np.linspace(0.05, 0.5, n)
    t0 = time.perf_counter()
    for amp in amps:
        run_one(h, float(amp))
    wall = time.perf_counter() - t0
    return wall, wall / n


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-values", type=int, nargs="+",
                        default=[1, 10, 100, 1000],
                        help="Batch sizes to benchmark.")
    args = parser.parse_args()

    os.chdir(str(NEURON_DIR))
    from neuron import h  # type: ignore[import-not-found]
    h.load_file("stdrun.hoc")

    OUT_DIR.mkdir(exist_ok=True)

    n_values: list[int] = []
    wall_s: list[float] = []
    per_sim_s: list[float] = []

    print(f"{'N':>8}{'wall (s)':>14}{'per-sim (ms)':>16}{'sims/s':>14}")
    print("-" * 52)
    for n in args.n_values:
        w, p = bench(h, n)
        n_values.append(n)
        wall_s.append(w)
        per_sim_s.append(p)
        print(f"{n:>8}{w:>14.3f}{p*1e3:>16.1f}{n/w:>14.2f}")

    out = OUT_DIR / "throughput_neuron.npz"
    np.savez(
        out,
        n_values=np.array(n_values),
        wall_s=np.array(wall_s),
        per_sim_s=np.array(per_sim_s),
        dt=DT, t_max=T_MAX, v_init=V_INIT, celsius=CELSIUS,
    )
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
