"""Gradient tests for the extracellular stimulation pipeline.

Confirms that jax.grad flows through every differentiable parameter:

1. electrode_current waveform  -- through point_source_potential + phi_e_to_ecs_nA
2. electrode_pos               -- through point_source_potential
3. sigma                       -- through point_source_potential
4. Full end-to-end             -- amplitude scalar -> phi_e -> i_ecs -> jx.integrate
"""

from __future__ import annotations

from typing import cast

import jax
import jax.numpy as jnp
import jaxley as jx
import numpy as np
import pytest

from jaxley_extracellular.extracellular.discretization import build_voltage_operator_G
from jaxley_extracellular.extracellular.equivalent_current import phi_e_to_ecs_nA
from jaxley_extracellular.extracellular.field import point_source_potential
from jaxley_extracellular.extracellular.jaxley_adapter import (
    ensure_compartment_centers,
    get_compartment_xyz,
)
from jaxley_extracellular.extracellular.typing_helpers import ECSParameters

# ---------------------------------------------------------------------------
# Shared setup
# ---------------------------------------------------------------------------

DT = 0.025  # ms
T_MAX = 1.0  # ms -- intentionally short; we only need non-zero gradients
N_STEPS = int(T_MAX / DT)


def _make_branch(ncomp: int = 4) -> jx.Branch:
    comp = jx.Compartment()
    branch = jx.Branch(comp, ncomp=ncomp)
    branch.set("length", 100.0)
    branch.set("radius", 1.0)
    branch.set("axial_resistivity", 100.0)
    branch.set("capacitance", 1.0)
    branch.set("v", -65.0)
    branch.init_states()
    return branch


def _static_ecs_parts(branch: jx.Branch) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Pre-compute the parts of the ECS pipeline that don't need to be traced."""
    ensure_compartment_centers(branch)
    comp_xyz = jnp.asarray(get_compartment_xyz(branch))  # static constant in JAX

    branch.to_jax()
    params = cast(ECSParameters, branch.get_all_parameters(pstate=[]))
    G = build_voltage_operator_G(branch, params)  # static (Ncomp, Ncomp)
    idx = np.asarray(branch.base._internal_node_inds)
    cm = params["capacitance"][idx]  # static (Ncomp,)
    area = params["area"][idx]  # static (Ncomp,)
    return comp_xyz, G, cm, area


# ---------------------------------------------------------------------------
# 1. Gradient w.r.t. electrode current waveform
# ---------------------------------------------------------------------------


def test_grad_wrt_current_waveform() -> None:
    """jax.grad flows through point_source_potential + phi_e_to_ecs_nA w.r.t. I(t)."""
    branch = _make_branch(ncomp=4)
    comp_xyz, G, cm, area = _static_ecs_parts(branch)

    electrode_pos = jnp.array([50.0, 50.0, 0.0])  # static position
    sigma = 0.3
    T = 10

    def loss(electrode_current: jax.Array) -> jax.Array:
        phi_e = point_source_potential(comp_xyz, electrode_pos, electrode_current, sigma)
        i_ecs = phi_e_to_ecs_nA(phi_e, G, cm, area)
        # Use i_ecs[0] rather than i_ecs.sum(): the symmetric electrode position
        # makes sum(G @ phi_e) = 0 exactly (row sums of G are zero and the spatial
        # profile is symmetric), so the total sum carries no gradient signal.
        return i_ecs[0].sum()

    electrode_current = jnp.ones((T,))
    grad = cast(jax.Array, jax.grad(loss)(electrode_current))

    assert grad.shape == (T,), f"Expected ({T},), got {grad.shape}"
    assert jnp.all(jnp.isfinite(grad)), "Gradient contains non-finite values"
    assert jnp.any(grad != 0.0), "Gradient is unexpectedly all-zero"


# ---------------------------------------------------------------------------
# 2. Gradient w.r.t. electrode position
# ---------------------------------------------------------------------------


def test_grad_wrt_electrode_pos() -> None:
    """jax.grad flows through point_source_potential w.r.t. electrode_pos."""
    branch = _make_branch(ncomp=4)
    comp_xyz, _G, _cm, _area = _static_ecs_parts(branch)

    electrode_current = jnp.ones((5,))
    sigma = 0.3

    def loss(electrode_pos: jax.Array) -> jax.Array:
        phi_e = point_source_potential(comp_xyz, electrode_pos, electrode_current, sigma)
        return phi_e.sum()

    electrode_pos = jnp.array([50.0, 50.0, 0.0])
    grad = cast(jax.Array, jax.grad(loss)(electrode_pos))

    assert grad.shape == (3,), f"Expected (3,), got {grad.shape}"
    assert jnp.all(jnp.isfinite(grad)), "Gradient contains non-finite values"
    assert jnp.any(grad != 0.0), "Gradient is unexpectedly all-zero"


def test_grad_electrode_pos_points_away_from_cell() -> None:
    """Moving the electrode away should reduce phi_e -- check sign via grad."""
    branch = _make_branch(ncomp=4)
    comp_xyz, _G, _cm, _area = _static_ecs_parts(branch)

    electrode_current = jnp.ones((1,))
    sigma = 0.3

    def total_phi_e(electrode_pos: jax.Array) -> jax.Array:
        phi_e = point_source_potential(comp_xyz, electrode_pos, electrode_current, sigma)
        return phi_e.sum()

    # Electrode directly above the cable midpoint.
    # Moving it along +y increases distance to all compartments -> phi_e decreases.
    pos = jnp.array([50.0, 100.0, 0.0])
    grad = cast(jax.Array, jax.grad(total_phi_e)(pos))

    # d(phi_e)/d(y) should be negative (moving away reduces potential)
    assert float(grad[1]) < 0.0, f"Expected negative y-gradient, got {float(grad[1])}"


# ---------------------------------------------------------------------------
# 3. Gradient w.r.t. sigma (conductivity)
# ---------------------------------------------------------------------------


def test_grad_wrt_sigma() -> None:
    """jax.grad flows through point_source_potential w.r.t. sigma."""
    branch = _make_branch(ncomp=4)
    comp_xyz, _G, _cm, _area = _static_ecs_parts(branch)

    electrode_pos = jnp.array([50.0, 50.0, 0.0])
    electrode_current = jnp.ones((5,))

    def loss(sigma: jax.Array) -> jax.Array:
        phi_e = point_source_potential(comp_xyz, electrode_pos, electrode_current, sigma)
        return phi_e.sum()

    grad = cast(jax.Array, jax.grad(loss)(jnp.array(0.3)))

    assert jnp.isfinite(grad), "Gradient w.r.t. sigma is non-finite"
    # Higher sigma -> lower phi_e, so gradient should be negative.
    assert float(grad) < 0.0, f"Expected negative sigma-gradient, got {float(grad)}"


# ---------------------------------------------------------------------------
# 4. End-to-end gradient through jx.integrate
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_grad_through_integrate_wrt_amplitude() -> None:
    """jax.grad flows end-to-end through the full jx.integrate pipeline.

    Differentiates a scalar loss (sum of recorded voltages) w.r.t. a scalar
    amplitude that scales the electrode waveform.  This exercises the entire
    path:  amplitude -> phi_e -> i_ecs -> data_stimuli -> jx.integrate.
    """
    ncomp = 4
    branch = _make_branch(ncomp=ncomp)
    # Record all compartments.
    for i in range(ncomp):
        branch.comp(i).record(verbose=False)

    comp_xyz, G, cm, area = _static_ecs_parts(branch)

    electrode_pos = jnp.array([50.0, 50.0, 0.0])
    sigma = 0.3
    T = N_STEPS

    def loss(amplitude: jax.Array) -> jax.Array:
        waveform = amplitude * jnp.ones((T,))
        phi_e = point_source_potential(comp_xyz, electrode_pos, waveform, sigma)
        i_ecs = phi_e_to_ecs_nA(phi_e, G, cm, area)
        data_stimuli = branch.data_stimulate(i_ecs)
        v = jx.integrate(
            branch,
            delta_t=DT,
            t_max=T_MAX,
            data_stimuli=data_stimuli,
            solver="bwd_euler",
        )
        return jnp.asarray(v).sum()

    val, grad = jax.value_and_grad(loss)(jnp.array(1.0))

    assert jnp.isfinite(val), "Forward pass returned non-finite loss"
    assert jnp.isfinite(grad), f"Gradient is non-finite: {grad}"
    assert float(grad) != 0.0, "End-to-end gradient is unexpectedly zero"
