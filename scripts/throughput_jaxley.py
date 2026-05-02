"""Jaxley throughput: N single-soma sims via jit+vmap over stimulus amplitude.

Same geometry + channel set as ``scripts/parity_single_soma.py`` (Pyr soma).
Builds one cell, then ``jax.jit(jax.vmap(...))`` the integrate over a vector of
amplitudes. Reports compile, first-run, and cached wall times per batch size.

Run:
    python scripts/throughput_jaxley.py
    python scripts/throughput_jaxley.py --n-values 1 10 100 1000 10000
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import cast

import jax
import jax.numpy as jnp
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
    NaTs2T,
    SKv3_1,
)

OUT_DIR = Path(__file__).resolve().parent.parent / "results"

# Geometry matched to scripts/parity_single_soma.py (Pyr)
L = 20.0
RADIUS = 10.0
RA = 100.0
CM = 1.0

DT = 0.025
V_INIT = -75.0
T_MAX = 700.0

I_DELAY = 100.0
I_DUR = 500.0

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


def make_pyr_cell() -> jx.Cell:
    comp = jx.Compartment()
    branch = jx.Branch(comp, ncomp=1)
    cell = jx.Cell([branch], parents=[-1])

    cell.set("length", L)
    cell.set("radius", RADIUS)
    cell.set("axial_resistivity", RA)
    cell.set("capacitance", CM)

    cell.insert(Leak())
    cell.insert(CaNernstReversal())
    cell.insert(CaHVA())
    cell.insert(CaLVA())
    cell.insert(CaPump())
    cell.insert(SKE2())
    cell.insert(NaTs2T())
    cell.insert(SKv3_1())
    cell.insert(H())

    for key, val in PYR_SOMA.items():
        cell.set(key, val)

    cell.set("v", V_INIT)
    cell.init_states()
    cell.record("v")
    return cell


def bench(n: int, cell: jx.Cell) -> dict[str, float]:
    """Benchmark one batch size. Returns compile_s, first_s, cached_s, sims_s."""
    amps = jnp.linspace(0.05, 0.5, n)

    def run_one(amp: jax.Array) -> jax.Array:
        stim = jx.step_current(I_DELAY, I_DUR, amp, DT, T_MAX)  # pyright: ignore[reportArgumentType]
        ds = cell.branch(0).comp(0).data_stimulate(stim, data_stimuli=None)
        v: jax.Array = jx.integrate(cell, delta_t=DT, t_max=T_MAX, data_stimuli=ds)
        return v

    run_batch = jax.jit(jax.vmap(run_one))

    t0 = time.perf_counter()
    v1 = cast(jax.Array, run_batch(amps))
    jax.block_until_ready(v1)  # type: ignore[no-untyped-call]
    t_first = time.perf_counter() - t0

    t0 = time.perf_counter()
    v2 = cast(jax.Array, run_batch(amps))
    jax.block_until_ready(v2)  # type: ignore[no-untyped-call]
    t_cached = time.perf_counter() - t0

    # Lower-bound compile by subtracting cached from first.
    t_compile = max(0.0, t_first - t_cached)

    return {
        "compile_s": t_compile,
        "first_s": t_first,
        "cached_s": t_cached,
        "per_sim_ms": t_cached * 1e3 / n,
        "sims_per_s": n / t_cached,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-values", type=int, nargs="+",
                        default=[1, 10, 100, 1000, 10000])
    args = parser.parse_args()

    print("jax", jax.__version__, "backend", jax.default_backend(), "devices", jax.devices())

    cell = make_pyr_cell()

    OUT_DIR.mkdir(exist_ok=True)

    n_values: list[int] = []
    compile_s: list[float] = []
    first_s: list[float] = []
    cached_s: list[float] = []

    print(f"\n{'N':>8}{'compile (s)':>14}{'1st run (s)':>14}{'cached (s)':>14}"
          f"{'per-sim (ms)':>16}{'sims/s':>14}")
    print("-" * 80)
    for n in args.n_values:
        r = bench(n, cell)
        n_values.append(n)
        compile_s.append(r["compile_s"])
        first_s.append(r["first_s"])
        cached_s.append(r["cached_s"])
        print(f"{n:>8}{r['compile_s']:>14.3f}{r['first_s']:>14.3f}"
              f"{r['cached_s']:>14.3f}{r['per_sim_ms']:>16.3f}{r['sims_per_s']:>14.1f}")

    out = OUT_DIR / "throughput_jaxley.npz"
    np.savez(
        out,
        n_values=np.array(n_values),
        compile_s=np.array(compile_s),
        first_s=np.array(first_s),
        cached_s=np.array(cached_s),
        dt=DT, t_max=T_MAX, v_init=V_INIT,
    )
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
