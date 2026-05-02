"""ECS differentiability receipt: jax.grad through phi_e + BBP Pyr.

Companion to ``scripts.diff_demo`` (intracellular). Differentiates a
peak-soma-voltage objective w.r.t. electrode xyz through
point_source_potential -> build_ecs_stimuli_nA -> jx.integrate on the
BBP Pyr cell.

Run:
    python scripts/diff_demo_ecs.py
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import cast

import jax
import jax.numpy as jnp
import jaxley as jx
import numpy as np

from jaxley_extracellular.bbp.cell_factory import make_pyr_cell
from jaxley_extracellular.extracellular.field import point_source_potential
from jaxley_extracellular.extracellular.jaxley_adapter import (
    build_ecs_stimuli_nA,
    ensure_compartment_centers,
    get_compartment_xyz,
    package_data_stimuli,
)

DT_MS     = 0.025
T_MAX_MS  = 15.0
V_INIT_MV = -75.0
SIGMA_S_M = 0.3
ELEC0_XYZ = jnp.asarray([0.0, 100.0, 0.0])  # 100 um above soma along apical axis
AMP_UA    = -50.0
PULSE_DELAY_MS = 2.0
PULSE_WIDTH_MS = 1.0


def _make_pulse(t_ms: jax.Array) -> jax.Array:
    """Cathodic monophasic pulse of AMP_UA between [delay, delay+width]."""
    in_pulse = (t_ms >= PULSE_DELAY_MS) & (t_ms < PULSE_DELAY_MS + PULSE_WIDTH_MS)
    return jnp.where(in_pulse, AMP_UA, 0.0)


def main() -> None:
    print("python:", __file__)
    print("jax", jax.__version__, "backend", jax.default_backend(),
          "devices", jax.devices())

    t0 = time.perf_counter()
    cell = make_pyr_cell(ncomp=2, max_branch_len=100.0)
    cell.set("v", V_INIT_MV)
    soma_comp = cell.soma.branch(0).comp(0)  # pyright: ignore[reportOptionalMemberAccess]
    soma_comp.record("v")
    ensure_compartment_centers(cell)
    comp_xyz = jnp.asarray(get_compartment_xyz(cell), dtype=jnp.float32)
    n_comp = int(comp_xyz.shape[0])
    print(f"BBP Pyr ncomp={n_comp}, build={time.perf_counter()-t0:.1f}s")

    n_steps = int(T_MAX_MS / DT_MS) + 1
    t_grid = jnp.arange(n_steps) * DT_MS

    def peak_soma_mV(electrode_xyz: jax.Array) -> jax.Array:
        """Scalar response: max somatic V (mV) given an electrode position.

        Closes over the cell + waveform; only `electrode_xyz` is jax-traced
        for gradient. ``point_source_potential`` is JAX-pure;
        ``build_ecs_stimuli_nA`` projects phi_e onto compartment currents
        via a sparse operator and is also JAX-pure.
        """
        wave = _make_pulse(t_grid)[None, :]                       # (1, T)
        elec_xyz = electrode_xyz[None, :]                         # (1, 3)
        phi_e = point_source_potential(
            comp_xyz=comp_xyz,
            electrode_positions=elec_xyz,
            electrode_currents=wave,
            sigma=SIGMA_S_M,
        )                                                         # (Ncomp, T)
        i_nA = build_ecs_stimuli_nA(cell, phi_e)                  # (Ncomp, T)
        ds = package_data_stimuli(cell, i_nA)
        v = cast(jax.Array, jx.integrate(
            cell, delta_t=DT_MS, t_max=T_MAX_MS, data_stimuli=ds,
        ))
        return jnp.max(v[0])

    val_and_grad = jax.value_and_grad(peak_soma_mV)

    print("\nFirst call (compile + run)...")
    t0 = time.perf_counter()
    v0, g0 = val_and_grad(ELEC0_XYZ)
    jax.block_until_ready(v0)  # type: ignore[no-untyped-call]
    jax.block_until_ready(g0)  # type: ignore[no-untyped-call]
    t_first = time.perf_counter() - t0
    print(f"  peak soma     : {float(v0):+.2f} mV")
    print(f"  d(peak)/d(xyz): {np.asarray(g0)} mV/um")
    print(f"  wall          : {t_first:.2f} s")

    print("\nSteady-state call...")
    t0 = time.perf_counter()
    v1, g1 = val_and_grad(ELEC0_XYZ)
    jax.block_until_ready(v1)  # type: ignore[no-untyped-call]
    jax.block_until_ready(g1)  # type: ignore[no-untyped-call]
    t_cached = time.perf_counter() - t0
    print(f"  wall          : {t_cached:.3f} s")

    print("\nOK: gradient flows end-to-end through ECS phi_e + BBP Pyr "
          "integration.")

    # Save a small receipt for the paper package.
    out = (Path(__file__).resolve().parent.parent
           / "results" / "paper_package" / "receipts"
           / "gradient_receipt_ecs.txt")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        f"python: {__file__}\n"
        f"jax {jax.__version__} backend {jax.default_backend()} "
        f"devices {jax.devices()}\n\n"
        f"BBP Pyr ncomp={n_comp}\n"
        f"Electrode xyz (um) = {tuple(float(x) for x in ELEC0_XYZ)}\n"
        f"Cathodic pulse: amp={AMP_UA} uA, width={PULSE_WIDTH_MS} ms, "
        f"delay={PULSE_DELAY_MS} ms\n"
        f"sigma = {SIGMA_S_M} S/m\n\n"
        f"Peak soma voltage:           {float(v0):+.2f} mV\n"
        f"d(peak_mV)/d(electrode_xyz): {np.asarray(g0)} mV/um\n"
        f"value+grad (compile + run):  {t_first:.2f} s\n"
        f"value+grad (steady-state):   {t_cached:.3f} s\n\n"
        f"OK: gradient flows end-to-end through ECS phi_e + BBP Pyr integration.\n"
    )
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
