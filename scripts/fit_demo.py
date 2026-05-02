"""Gradient-based parameter fit on a morphologically-accurate BBP Pyr cell.

Target: v(t) of BBP's default Pyr cell under a suprathreshold step.
Initial guess: somatic NaTs2T conductance perturbed by 15%.
Optimization: Adam steps minimizing MSE(v, v_target) via ``jax.value_and_grad``
through the full 700-compartment integration.

One parameter and a mild perturbation: the cost surface has a step
discontinuity at spike threshold (one fewer spike means a massive loss
jump), so we stay inside the basin where init and target produce the
same spike count. This is a gradient-flow receipt, not a benchmark of
the optimiser.

Run:
    python scripts/fit_demo.py
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
import jaxley as jx
import numpy as np

from jaxley_extracellular.bbp.cell_factory import make_pyr_cell

OUT_DIR = Path(__file__).resolve().parent.parent / "results"

DT = 0.025
V_INIT = -75.0
T_MAX = 80.0
I_DELAY = 10.0
I_DUR = 40.0
I_AMP = 0.5  # nA, suprathresh -> same spike count for init and target

N_STEPS = 20
LR = 0.02  # log-param-space learning rate

TRUE_NATS2T = 0.926705  # BBP PYR_SOMA default
# theta = log(g / TRUE). theta=0 <=> exact. Mild 0.85x perturbation stays in
# the same-spike-count basin as the target, so the loss surface is smooth.
INIT_THETA = float(np.log(0.85))


def build_cell() -> tuple[jx.Cell, Any]:
    cell = make_pyr_cell(ncomp=2, max_branch_len=100.0)
    cell.set("v", V_INIT)
    soma_comp = cell.soma.branch(0).comp(0)  # pyright: ignore[reportOptionalMemberAccess]
    soma_comp.record("v")
    soma_comp.make_trainable("NaTs2T_gNaTs2T")
    return cell, soma_comp


def simulate(cell: jx.Cell, params: list[dict[str, jax.Array]],
             soma_comp: Any) -> jax.Array:
    stim = jx.step_current(I_DELAY, I_DUR, I_AMP, DT, T_MAX)
    ds = soma_comp.data_stimulate(stim, data_stimuli=None)
    v: jax.Array = jx.integrate(cell, params=params, delta_t=DT, t_max=T_MAX,
                                data_stimuli=ds)
    return v


def main() -> None:
    print("jax", jax.__version__, "backend", jax.default_backend(), "devices", jax.devices())

    cell, soma_comp = build_cell()
    ncomps = int(cell.nodes.shape[0])  # pyright: ignore[reportOptionalMemberAccess]
    init_gNa = float(np.exp(INIT_THETA) * TRUE_NATS2T)
    print(f"\nBBP Pyr morphology: {ncomps} compartments")
    print(f"Target:   gNaTs2T = {TRUE_NATS2T:.4f} S/cm^2 (BBP value)")
    print(f"Initial:  gNaTs2T = {init_gNa:.4f} ({np.exp(INIT_THETA):.2f}x)")
    print("Optimizing in log space: theta = log(g / TRUE); theta=0 -> exact.")

    true_params = cell.get_parameters()
    print(f"\nGenerating target trace (stim {I_AMP} nA, {I_DUR} ms)...")
    t0 = time.perf_counter()
    v_target = simulate(cell, true_params, soma_comp)
    jax.block_until_ready(v_target)  # type: ignore[no-untyped-call]
    print(f"  target integration: {time.perf_counter() - t0:.2f}s")

    tpl_gNa = true_params[0]["NaTs2T_gNaTs2T"]

    def params_from_theta(theta: jax.Array) -> list[dict[str, jax.Array]]:
        gNa = jnp.full_like(tpl_gNa, TRUE_NATS2T * jnp.exp(theta[0]))
        return [{"NaTs2T_gNaTs2T": gNa}]

    def loss_fn(theta: jax.Array) -> jax.Array:
        p = params_from_theta(theta)
        v = simulate(cell, p, soma_comp)
        return jnp.mean((v[0] - v_target[0]) ** 2)

    loss_and_grad = jax.jit(jax.value_and_grad(loss_fn))

    theta = jnp.array([INIT_THETA])
    m = jnp.zeros_like(theta)
    v_state = jnp.zeros_like(theta)

    print(f"\nOptimizing ({N_STEPS} Adam steps, lr={LR})...")
    print(f"{'step':>5}{'loss (mV^2)':>14}{'gNaTs2T':>12}{'err %':>10}")
    print("-" * 42)

    history: list[tuple[int, float, float]] = []
    t0 = time.perf_counter()
    for step in range(1, N_STEPS + 1):
        result = cast(tuple[jax.Array, jax.Array], loss_and_grad(theta))
        loss, grads = result
        b1, b2, eps = 0.9, 0.999, 1e-8
        m = b1 * m + (1 - b1) * grads
        v_state = b2 * v_state + (1 - b2) * grads ** 2
        m_hat = m / (1 - b1 ** step)
        v_hat = v_state / (1 - b2 ** step)
        theta = theta - LR * m_hat / (jnp.sqrt(v_hat) + eps)
        gNa = float(TRUE_NATS2T * float(jnp.exp(theta[0])))
        pct_err = abs(gNa - TRUE_NATS2T) / TRUE_NATS2T * 100
        history.append((step, float(loss), gNa))
        print(f"{step:>5}{float(loss):>14.4f}{gNa:>12.4f}{pct_err:>9.2f}%")

    wall = time.perf_counter() - t0

    print(f"\nOptimization: {wall:.1f}s total, {wall / N_STEPS:.2f}s per step")
    final_gNa = float(TRUE_NATS2T * float(jnp.exp(theta[0])))
    print("\nFinal:")
    print(f"  gNaTs2T: true={TRUE_NATS2T:.4f}  init={init_gNa:.4f}  "
          f"fit={final_gNa:.4f}  err={(final_gNa - TRUE_NATS2T) / TRUE_NATS2T * 100:+.2f}%")

    OUT_DIR.mkdir(exist_ok=True)
    out = OUT_DIR / "fit_demo.npz"
    h = np.array(history)
    v_final = simulate(cell, params_from_theta(theta), soma_comp)
    jax.block_until_ready(v_final)  # type: ignore[no-untyped-call]
    np.savez(
        out,
        step=h[:, 0], loss=h[:, 1], gNaTs2T=h[:, 2],
        v_target=np.array(v_target[0]), v_final=np.array(v_final[0]),
        true_gNaTs2T=TRUE_NATS2T, init_gNaTs2T=init_gNa,
        wall_s=wall, n_steps=N_STEPS, ncomps=ncomps,
    )
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
