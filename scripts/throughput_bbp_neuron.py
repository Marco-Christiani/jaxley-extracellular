"""NEURON throughput on the BBP L2/3 Pyr cell (cADpyr229_L23_PC_5ecbf9b163).

Loads the same BBP cell template used by the vendored BBP reference simulation (biophysics,
morphology, channel mods), then runs a sequential loop of N step-current sims
varying amplitude. Same DT/T_MAX as scripts/throughput_bbp_jaxley.py for apples-
to-apples comparison.

Run from the dedicated NEURON shell:

    nix develop .#neuron
    cd reference/bbp/simulation && nrnivmodl mechanisms
    cd /path/to/jaxley-extracellular
    neuron-python -m scripts.throughput_bbp_neuron

Writes results/throughput_bbp_neuron.npz.
"""

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np

from scripts.neuron_bbp_common import load_bbp_pyr_cell
from scripts.parity_common import provenance_fields

OUT_DIR = Path(__file__).resolve().parent.parent / "results"

DT = 0.025
V_INIT = -75.0
T_MAX = 200.0
CELSIUS = 34.0

I_DELAY = 20.0
I_DUR = 100.0
def run_one(h: Any, cell: Any, amp: float) -> float:
    """Run one step-current protocol on an existing cell, return wall seconds.

    Uses a fresh IClamp per call; cell state is re-initialized via h.finitialize.
    """
    iclamp = h.IClamp(cell.soma[0](0.5))
    iclamp.delay = I_DELAY
    iclamp.dur = I_DUR
    iclamp.amp = amp

    h.celsius = CELSIUS
    h.dt = DT
    h.finitialize(V_INIT)
    t0 = time.perf_counter()
    h.continuerun(T_MAX)
    return time.perf_counter() - t0


def bench(h: Any, cell: Any, n: int) -> tuple[float, float]:
    amps = np.linspace(0.1, 1.0, n)
    t0 = time.perf_counter()
    for amp in amps:
        run_one(h, cell, float(amp))
    wall = time.perf_counter() - t0
    return wall, wall / n


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-values", type=int, nargs="+",
                        default=[1, 10, 100, 1000])
    args = parser.parse_args()

    from neuron import h  # type: ignore[import-not-found]

    cell = load_bbp_pyr_cell(h)

    n_soma = sum(1 for _ in cell.soma)
    n_apical = sum(1 for _ in cell.apical)
    n_basal = sum(1 for _ in cell.basal)
    n_axon = sum(1 for _ in cell.axon)
    total_secs = n_soma + n_apical + n_basal + n_axon
    total_segs = sum(sec.nseg for sec in list(cell.soma) + list(cell.apical)
                     + list(cell.basal) + list(cell.axon))
    print("BBP L2/3 Pyr cADpyr229_L23_PC_5ecbf9b163:")
    print(f"  sections: soma={n_soma}, apical={n_apical}, basal={n_basal}, "
          f"axon={n_axon}, total={total_secs}")
    print(f"  total segments (NEURON nseg): {total_segs}")
    print(f"  simulation: T_MAX={T_MAX} ms, DT={DT} ms ({int(T_MAX / DT)} steps)\n")

    OUT_DIR.mkdir(exist_ok=True)

    n_values: list[int] = []
    wall_s: list[float] = []
    per_sim_s: list[float] = []

    print(f"{'N':>8}{'wall (s)':>14}{'per-sim (ms)':>16}{'sims/s':>14}")
    print("-" * 52)
    for n in args.n_values:
        w, p = bench(h, cell, n)
        n_values.append(n)
        wall_s.append(w)
        per_sim_s.append(p)
        print(f"{n:>8}{w:>14.3f}{p*1e3:>16.1f}{n/w:>14.2f}")

    out = OUT_DIR / "throughput_bbp_neuron.npz"
    np.savez(
        out,
        n_values=np.array(n_values),
        wall_s=np.array(wall_s),
        per_sim_s=np.array(per_sim_s),
        total_segments=total_segs,
        total_sections=total_secs,
        dt=DT, t_max=T_MAX, v_init=V_INIT, celsius=CELSIUS,
        **provenance_fields(
            platform="cpu",
            hardware_label="cpu single-core (NEURON serial)",
            script_path=__file__,
        ),
    )
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
