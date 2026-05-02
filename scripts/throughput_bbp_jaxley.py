"""Jaxley throughput on morphologically-accurate BBP L2/3 pyramidal cells.

Builds the full Pyr morphology (cADpyr229) via ``make_pyr_cell`` and times
``jax.jit(jax.vmap(...))`` over a vector of intracellular step-current amplitudes.
Each vmap lane = one independent BBP Pyr cell integration.

The amplitude vector is sharded across all visible JAX devices via the
modern ``jax.sharding`` API, so on a multi-chip TPU pod-slice the lanes
run in parallel across chips. On a single device this degrades to a
no-op ``(1,)`` mesh.

Run:
    python scripts/throughput_bbp_jaxley.py
    python scripts/throughput_bbp_jaxley.py --ncomp 2 --max-branch-len 100 --n-values 1 10 100 1000
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

from jaxley_extracellular.bbp.cell_factory import make_pyr_cell
from jaxley_extracellular.extracellular.sharding import (
    config_sharding,
    make_device_mesh,
    pad_to_devices,
    shard_batch,
)
from scripts.parity_common import provenance_fields

OUT_DIR = Path(__file__).resolve().parent.parent / "results"

DT = 0.025
V_INIT = -75.0
T_MAX = 200.0

I_DELAY = 20.0
I_DUR = 100.0


def bench(
    n: int,
    cell: jx.Cell,
    sharding: jax.sharding.NamedSharding,
    checkpoint_lengths: list[int] | None = None,
) -> dict[str, float]:
    amps_unsharded = jnp.linspace(0.1, 1.0, n)
    n_devices = jax.device_count()
    # Pad to a multiple of n_devices so each chip gets an equal slice. The
    # padded lanes do real (wasted) compute. We time on the original n only.
    amps_padded, n_pad = pad_to_devices(amps_unsharded, n_devices)
    amps_sharded = shard_batch(amps_unsharded, sharding) if n_devices > 1 else amps_unsharded
    n_padded = int(amps_padded.shape[0])

    soma_comp = cell.soma.branch(0).comp(0)  # pyright: ignore[reportOptionalMemberAccess]

    def run_one(amp: jax.Array) -> jax.Array:
        stim = jx.step_current(I_DELAY, I_DUR, amp, DT, T_MAX)  # pyright: ignore[reportArgumentType]
        ds = soma_comp.data_stimulate(stim, data_stimuli=None)
        v: jax.Array = jx.integrate(
            cell, delta_t=DT, t_max=T_MAX, data_stimuli=ds,
            checkpoint_lengths=checkpoint_lengths,
        )
        return v

    run_batch = jax.jit(jax.vmap(run_one))

    t0 = time.perf_counter()
    v1 = cast(jax.Array, run_batch(amps_sharded))
    jax.block_until_ready(v1)  # type: ignore[no-untyped-call]
    t_first = time.perf_counter() - t0

    t0 = time.perf_counter()
    v2 = cast(jax.Array, run_batch(amps_sharded))
    jax.block_until_ready(v2)  # type: ignore[no-untyped-call]
    t_cached = time.perf_counter() - t0

    t_compile = max(0.0, t_first - t_cached)
    return {
        "compile_s": t_compile,
        "first_s": t_first,
        "cached_s": t_cached,
        "per_sim_ms": t_cached * 1e3 / n,
        "sims_per_s": n / t_cached,
        "n_padded": float(n_padded),
        "n_pad_lanes": float(n_pad),
        "padded_sims_per_s": n_padded / t_cached,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ncomp", type=int, default=2)
    parser.add_argument("--max-branch-len", type=float, default=100.0)
    parser.add_argument("--n-values", type=int, nargs="+", default=[1, 10, 100, 1000])
    parser.add_argument(
        "--checkpoint-lengths", type=int, nargs="+", default=None,
        help="Hierarchical scan-checkpointing factors for jx.integrate (prod >= T).",
    )
    args = parser.parse_args()

    print("jax", jax.__version__, "backend", jax.default_backend(), "devices", jax.devices())

    mesh = make_device_mesh()
    sharding = config_sharding(mesh)
    n_devices = jax.device_count()
    print(f"Device mesh: {mesh.devices.shape} (n_devices={n_devices})")

    t0 = time.perf_counter()
    cell = make_pyr_cell(ncomp=args.ncomp, max_branch_len=args.max_branch_len)
    t_build = time.perf_counter() - t0
    ncomps = int(cell.nodes.shape[0])  # pyright: ignore[reportOptionalMemberAccess]
    cell.set("v", V_INIT)
    cell.soma.branch(0).comp(0).record("v")  # pyright: ignore[reportOptionalMemberAccess]
    print(f"\nBBP L2/3 Pyr cADpyr229: ncomp_arg={args.ncomp}, "
          f"max_branch_len={args.max_branch_len}, total_compartments={ncomps}, "
          f"build_s={t_build:.2f}")
    print(f"Simulation: T_MAX={T_MAX}ms, DT={DT}ms ({int(T_MAX / DT)} steps)\n")

    OUT_DIR.mkdir(exist_ok=True)

    n_values: list[int] = []
    compile_s: list[float] = []
    first_s: list[float] = []
    cached_s: list[float] = []
    n_padded_list: list[int] = []

    # B = batch size (cells per vmap dispatch); N is reserved for compartment
    # count to match paper notation.
    print(f"{'B':>6}{'B_pad':>8}{'compile (s)':>14}{'1st run (s)':>14}{'cached (s)':>14}"
          f"{'per-sim (ms)':>14}{'sims/s':>10}{'pad sims/s':>12}")
    print("-" * 88)
    for n in args.n_values:
        r = bench(n, cell, sharding, checkpoint_lengths=args.checkpoint_lengths)
        n_values.append(n)
        compile_s.append(r["compile_s"])
        first_s.append(r["first_s"])
        cached_s.append(r["cached_s"])
        n_padded_list.append(int(r["n_padded"]))
        print(f"{n:>6}{int(r['n_padded']):>8}{r['compile_s']:>14.3f}{r['first_s']:>14.3f}"
              f"{r['cached_s']:>14.3f}{r['per_sim_ms']:>14.3f}{r['sims_per_s']:>10.1f}"
              f"{r['padded_sims_per_s']:>12.1f}")

    backend = jax.default_backend()
    hardware_label = (
        f"{backend} {n_devices}-device" if n_devices > 1 else f"{backend} single-device"
    )
    out = OUT_DIR / "throughput_bbp_jaxley.npz"
    np.savez(
        out,
        n_values=np.array(n_values),
        n_padded=np.array(n_padded_list),
        n_devices=n_devices,
        compile_s=np.array(compile_s),
        first_s=np.array(first_s),
        cached_s=np.array(cached_s),
        ncomps=ncomps,
        dt=DT, t_max=T_MAX, v_init=V_INIT,
        **provenance_fields(
            platform=backend,
            hardware_label=hardware_label,
            script_path=__file__,
        ),
    )
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
