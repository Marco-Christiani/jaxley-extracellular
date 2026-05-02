"""Differentiability receipt: jax.grad through a full BBP Pyr sim.

Differentiates a somatic peak-depolarization scalar through the
morphologically-accurate BBP L2/3 Pyr cell's integration with respect
to a tunable biophysical parameter (somatic Na conductance).

Run:
    python scripts/diff_demo.py
"""

from __future__ import annotations

import time
from typing import cast

import jax
import jax.numpy as jnp
import jaxley as jx

from jaxley_extracellular.bbp.cell_factory import make_pyr_cell

DT = 0.025
V_INIT = -75.0
T_MAX = 60.0
I_DELAY = 10.0
I_DUR = 20.0
I_AMP = 0.4


def main() -> None:
    print("jax", jax.__version__, "backend", jax.default_backend(), "devices", jax.devices())

    cell = make_pyr_cell(ncomp=2, max_branch_len=100.0)
    cell.set("v", V_INIT)
    cell.soma.branch(0).comp(0).record("v")  # pyright: ignore[reportOptionalMemberAccess]
    soma_comp = cell.soma.branch(0).comp(0)  # pyright: ignore[reportOptionalMemberAccess]

    # Expose gNaTs2T in the soma as the only tunable parameter
    soma_comp.make_trainable("NaTs2T_gNaTs2T")
    params = cell.get_parameters()
    print(f"Trainable params: {[next(iter(p.keys())) for p in params]}")
    print(f"Starting gNaTs2T = {next(iter(params[0].values())).item():.4f} S/cm^2")

    stim = jx.step_current(I_DELAY, I_DUR, I_AMP, DT, T_MAX)
    ds = soma_comp.data_stimulate(stim, data_stimuli=None)

    def loss(trainable_params: list[dict[str, jax.Array]]) -> jax.Array:
        v = jx.integrate(
            cell,
            params=trainable_params,
            delta_t=DT,
            t_max=T_MAX,
            data_stimuli=ds,
        )
        # Peak depolarization at the soma (negative because we'll descend)
        return -jnp.max(v[0])

    loss_and_grad = jax.jit(jax.value_and_grad(loss))

    t0 = time.perf_counter()
    out = cast(tuple[jax.Array, list[dict[str, jax.Array]]], loss_and_grad(params))
    val, grad = out
    jax.block_until_ready(grad[0]["NaTs2T_gNaTs2T"])  # type: ignore[no-untyped-call]
    t_first = time.perf_counter() - t0

    t0 = time.perf_counter()
    out = cast(tuple[jax.Array, list[dict[str, jax.Array]]], loss_and_grad(params))
    val, grad = out
    jax.block_until_ready(grad[0]["NaTs2T_gNaTs2T"])  # type: ignore[no-untyped-call]
    t_cached = time.perf_counter() - t0

    peak_mV = -float(val)
    g = float(grad[0]["NaTs2T_gNaTs2T"].item())

    print()
    print(f"Peak soma voltage:           {peak_mV:+.2f} mV")
    print(f"d(peak_mV) / d(gNaTs2T):     {-g:+.2f} mV per (S/cm^2)")
    print(f"value+grad (compile + run):  {t_first:.2f} s")
    print(f"value+grad (cached):         {t_cached:.3f} s")
    print()
    print("OK: gradient flows end-to-end through BBP L2/3 Pyr integration.")


if __name__ == "__main__":
    main()
