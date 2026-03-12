"""Validate that build_voltage_operator_G produces a Jaxley-consistent operator.

Tests
-----
1. Cable (jx.Branch, ncomp=4): G is tridiagonal, row-sums ~= 0, symmetric.
2. Cable row/col structure: diagonal <= 0, off-diagonal >= 0.
3. Spatially uniform phi_e => G @ phi_e ~= 0  (no net forcing).
4. Branched cell (jx.Cell, two branches): row-sums ~= 0, correct through-
   branchpoint cross-terms appear in G.
5. Single compartment (jx.Compartment): G is [[0]].
6. Operator equivalence: G @ v == Jaxley's _voltage_vectorfield (axial only).
7. Symmetry audit: symmetric for uniform params, asymmetric for non-uniform.
8. Analytical activating function: G @ phi_e matches d^2phi_e/dx^2.
"""

from __future__ import annotations

from typing import Any, cast

import jax
import jax.numpy as jnp
import jaxley as jx
import jaxley.solver_voltage as solver_voltage
import numpy as np
import pytest

from jaxley_extracellular.extracellular.discretization import build_voltage_operator_G
from jaxley_extracellular.extracellular.jaxley_adapter import (
    ensure_compartment_centers,
    get_compartment_xyz,
)
from jaxley_extracellular.extracellular.typing_helpers import ECSParameters

VoltageVectorfield = Any
_voltage_vectorfield: VoltageVectorfield = object.__getattribute__(
    solver_voltage, "_voltage_vectorfield"
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_branch(ncomp: int = 4) -> jx.Branch:
    comp = jx.Compartment()
    branch = jx.Branch(comp, ncomp=ncomp)
    branch.set("length", 100.0)  # um
    branch.set("radius", 1.0)  # um
    branch.set("axial_resistivity", 100.0)  # Ohm*cm
    branch.set("capacitance", 1.0)  # uF/cm^2
    branch.to_jax()
    return branch


def _make_cell_two_branches() -> jx.Cell:
    """Simple Y-cell: root branch + one child."""
    comp = jx.Compartment()
    b0 = jx.Branch(comp, ncomp=4)
    b1 = jx.Branch(comp, ncomp=3)
    cell = jx.Cell([b0, b1], parents=[-1, 0])
    cell.set("length", 100.0)
    cell.set("radius", 1.0)
    cell.set("axial_resistivity", 100.0)
    cell.set("capacitance", 1.0)
    cell.to_jax()
    return cell


def _build_G(module: jx.Module) -> np.ndarray:
    """Return G as a numpy array (already transferred from device)."""
    module.to_jax()
    params = cast(ECSParameters, module.get_all_parameters(pstate=[]))
    G = build_voltage_operator_G(module, params)
    return np.asarray(G)


# ---------------------------------------------------------------------------
# 1. Cable: basic structure
# ---------------------------------------------------------------------------


def test_cable_G_shape() -> None:
    branch = _make_branch(ncomp=4)
    G = _build_G(branch)
    assert G.shape == (4, 4), f"Expected (4,4), got {G.shape}"


def test_cable_G_row_sums_zero() -> None:
    """Row sums of G must be zero (current conservation / sealed ends)."""
    branch = _make_branch(ncomp=4)
    G = _build_G(branch)
    row_sums = G.sum(axis=1)
    np.testing.assert_allclose(row_sums, 0.0, atol=1e-10)


def test_cable_G_diagonal_negative() -> None:
    branch = _make_branch(ncomp=4)
    G = _build_G(branch)
    assert np.all(np.diag(G) <= 0.0), "Diagonal must be non-positive"
    # Interior compartments must have strictly negative diagonal
    assert np.all(np.diag(G)[1:-1] < 0.0)


def test_cable_G_offdiagonals_nonnegative() -> None:
    branch = _make_branch(ncomp=4)
    G = _build_G(branch)
    n = G.shape[0]
    mask = ~np.eye(n, dtype=bool)
    assert np.all(G[mask] >= 0.0), "Off-diagonal entries must be non-negative"


def test_cable_G_symmetric_for_uniform_params() -> None:
    """For a cable with identical compartments G should be symmetric."""
    branch = _make_branch(ncomp=5)
    G = _build_G(branch)
    np.testing.assert_allclose(G, G.T, atol=1e-10)


def test_cable_G_tridiagonal() -> None:
    """For a single branch there should be no entries outside the tridiagonal."""
    branch = _make_branch(ncomp=4)
    G = _build_G(branch)
    n = G.shape[0]
    for i in range(n):
        for j in range(n):
            if abs(i - j) > 1:
                assert G[i, j] == pytest.approx(0.0, abs=1e-12), (
                    f"Expected G[{i},{j}]=0, got {G[i, j]}"
                )


# ---------------------------------------------------------------------------
# 2. Uniform phi_e => zero forcing
# ---------------------------------------------------------------------------


def test_uniform_phi_e_zero_forcing() -> None:
    """A spatially uniform phi_e produces G @ phi_e = 0 (row sums = 0)."""
    branch = _make_branch(ncomp=6)
    G = _build_G(branch)
    Ncomp = G.shape[0]
    T = 50
    phi_uniform = np.ones((Ncomp, T))  # same value everywhere
    f_ecs = G @ phi_uniform
    np.testing.assert_allclose(f_ecs, 0.0, atol=1e-9)


def test_linear_phi_e_forcing_sign() -> None:
    """A linear phi_e gradient should produce opposite forcing at the two ends."""
    branch = _make_branch(ncomp=4)
    G = _build_G(branch)
    Ncomp = G.shape[0]
    phi_linear = np.linspace(0.0, 1.0, Ncomp)[:, np.newaxis]  # (Ncomp, 1)
    f = G @ phi_linear  # (Ncomp, 1)
    # Left end: positive forcing (phi_e is lower here -> current pushed in)
    assert float(f[0, 0]) > 0.0, "Left-end forcing should be positive for rising phi_e"
    # Right end: negative forcing
    assert float(f[-1, 0]) < 0.0, "Right-end forcing should be negative for rising phi_e"
    # Interior: forcing ~= 0 for linear gradient (uniform second derivative = 0)
    np.testing.assert_allclose(f[1:-1], 0.0, atol=1e-9)


# ---------------------------------------------------------------------------
# 3. Branched cell
# ---------------------------------------------------------------------------


def test_branched_cell_G_shape() -> None:
    cell = _make_cell_two_branches()
    G = _build_G(cell)
    # 4 comps in branch 0 + 3 comps in branch 1
    assert G.shape == (7, 7), f"Expected (7,7), got {G.shape}"


def test_branched_cell_G_row_sums_zero() -> None:
    cell = _make_cell_two_branches()
    G = _build_G(cell)
    row_sums = G.sum(axis=1)
    np.testing.assert_allclose(row_sums, 0.0, atol=1e-9)


def test_branched_cell_through_branchpoint_coupling() -> None:
    """Compartments that share a branchpoint should have non-zero cross-terms."""
    cell = _make_cell_two_branches()
    G = _build_G(cell)
    # Branch 0 has comps 0-3; last comp of branch 0 is index 3.
    # Branch 1 has comps 4-6; first comp of branch 1 is index 4.
    # These two share a branchpoint: G[3,4] and G[4,3] should be > 0.
    assert G[3, 4] > 0.0, "Expected cross-branchpoint coupling G[3,4] > 0"
    assert G[4, 3] > 0.0, "Expected cross-branchpoint coupling G[4,3] > 0"


def test_branched_cell_uniform_phi_zero_forcing() -> None:
    cell = _make_cell_two_branches()
    G = _build_G(cell)
    Ncomp = G.shape[0]
    phi_uniform = np.ones((Ncomp, 10))
    f_ecs = G @ phi_uniform
    np.testing.assert_allclose(f_ecs, 0.0, atol=1e-8)


# ---------------------------------------------------------------------------
# 4. Single compartment
# ---------------------------------------------------------------------------


def test_single_compartment_G() -> None:
    comp = jx.Compartment()
    comp.to_jax()
    G = _build_G(comp)
    assert G.shape == (1, 1)
    assert G[0, 0] == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# 5. Operator equivalence: G @ v vs Jaxley _voltage_vectorfield
# ---------------------------------------------------------------------------


def _get_solver_kwargs(module: jx.Module) -> dict[str, np.ndarray | int]:
    """Extract solver_kwargs needed for _voltage_vectorfield."""
    base = module.base
    edges = object.__getattribute__(base, "_comp_edges")
    n_nodes = cast(int, base._n_nodes)
    return dict(
        sinks=np.asarray(edges["sink"].to_list()),
        sources=np.asarray(edges["source"].to_list()),
        types=np.asarray(edges["type"].to_list()),
        n_nodes=n_nodes,
    )


def _compare_G_vs_vectorfield(
    module: jx.Module, v_comps: jax.Array, rtol: float = 1e-10, atol: float = 1e-10
) -> None:
    """Assert G @ v_comps == axial part of _voltage_vectorfield at compartments.

    Sets voltage_terms=0 and constant_terms=0 so _voltage_vectorfield returns
    only the axial contribution to dv/dt.  With JAX_ENABLE_X64=1 (set in
    conftest.py), both paths use float64, giving agreement to ~1e-13.
    """
    module.to_jax()
    params = cast(ECSParameters, module.get_all_parameters(pstate=[]))
    G = build_voltage_operator_G(module, params)
    kw = _get_solver_kwargs(module)
    idx = np.asarray(module.base._internal_node_inds)

    Gv = np.asarray(G @ v_comps)

    v_full = jnp.zeros(kw["n_nodes"])
    v_full = v_full.at[idx].set(v_comps)

    vf = _voltage_vectorfield(
        v_full,
        jnp.zeros(kw["n_nodes"]),
        jnp.zeros(kw["n_nodes"]),
        params["axial_conductances"]["v"],
        kw["sinks"],
        kw["sources"],
        kw["types"],
        kw["n_nodes"],
    )
    vf_comps = np.asarray(vf[idx])

    np.testing.assert_allclose(Gv, vf_comps, rtol=rtol, atol=atol)


def test_operator_equivalence_cable_random_v() -> None:
    """G @ v matches Jaxley's axial vectorfield for random voltages on a cable."""
    branch = _make_branch(ncomp=8)
    rng = np.random.default_rng(42)
    for _ in range(3):
        v = jnp.array(rng.standard_normal(8) * 20 - 65)
        _compare_G_vs_vectorfield(branch, v)


def test_operator_equivalence_cable_nonuniform_v() -> None:
    """G @ v matches Jaxley's axial vectorfield for structured voltage patterns."""
    branch = _make_branch(ncomp=8)
    n = 8
    # Sinusoidal
    v_sin = jnp.array(np.sin(np.linspace(0, 2 * np.pi, n)) * 30 - 65)
    _compare_G_vs_vectorfield(branch, v_sin)
    # Ramp
    v_ramp = jnp.linspace(-80.0, -40.0, n)
    _compare_G_vs_vectorfield(branch, v_ramp)
    # Spike-like: one compartment depolarised
    v_spike = jnp.full(n, -65.0).at[n // 2].set(30.0)
    _compare_G_vs_vectorfield(branch, v_spike)


def test_operator_equivalence_branched_random_v() -> None:
    """G @ v matches Jaxley's axial vectorfield for random v on a Y-cell."""
    cell = _make_cell_two_branches()
    ncomp = 7  # 4 + 3
    rng = np.random.default_rng(99)
    for _ in range(3):
        v = jnp.array(rng.standard_normal(ncomp) * 20 - 65)
        _compare_G_vs_vectorfield(cell, v)


def test_operator_equivalence_branched_nonuniform_v() -> None:
    """G @ v matches Jaxley's axial vectorfield for structured v on a Y-cell."""
    cell = _make_cell_two_branches()
    ncomp = 7
    v_sin = jnp.array(np.sin(np.linspace(0, 2 * np.pi, ncomp)) * 30 - 65)
    _compare_G_vs_vectorfield(cell, v_sin)
    v_ramp = jnp.linspace(-80.0, -40.0, ncomp)
    _compare_G_vs_vectorfield(cell, v_ramp)
    v_spike = jnp.full(ncomp, -65.0).at[ncomp // 2].set(30.0)
    _compare_G_vs_vectorfield(cell, v_spike)


# ---------------------------------------------------------------------------
# 6. Symmetry audit
# ---------------------------------------------------------------------------


def test_branched_cell_G_symmetric_for_uniform_params() -> None:
    """For a Y-cell with uniform params, G should be symmetric."""
    cell = _make_cell_two_branches()
    G = _build_G(cell)
    np.testing.assert_allclose(G, G.T, atol=1e-10)


def test_cable_G_asymmetric_for_nonuniform_radius() -> None:
    """With varying radius per compartment, G is NOT symmetric.

    This proves the symmetry claim is correctly limited to uniform params.
    """
    comp = jx.Compartment()
    branch = jx.Branch(comp, ncomp=5)
    branch.set("length", 100.0)
    branch.set("axial_resistivity", 100.0)
    branch.set("capacitance", 1.0)
    for i in range(5):
        branch.comp(i).set("radius", float(1.0 + i * 0.5))
    branch.to_jax()
    G = _build_G(branch)
    # G should NOT be symmetric
    max_asymmetry = np.max(np.abs(G - G.T))
    assert max_asymmetry > 0.1, (
        f"Expected asymmetric G for non-uniform radius, got max |G-G^T|={max_asymmetry}"
    )


# ---------------------------------------------------------------------------
# 7. Analytical activating function
# ---------------------------------------------------------------------------


def _make_uniform_cable(ncomp: int, total_length_um: float = 4000.0) -> jx.Branch:
    """Build a cable with coordinates consistent with compartment length.

    Sets xyzr so that coordinate spacing equals the per-compartment length,
    which is needed for the analytical activating function comparison.
    """
    comp_len = total_length_um / ncomp
    comp = jx.Compartment()
    branch = jx.Branch(comp, ncomp=ncomp)
    branch.set("length", comp_len)
    branch.set("radius", 1.0)
    branch.set("axial_resistivity", 100.0)
    branch.set("capacitance", 1.0)
    branch.xyzr[0] = np.array([[0, 0, 0, 1.0], [total_length_um, 0, 0, 1.0]])
    branch.compute_compartment_centers()
    branch.to_jax()
    return branch


def _analytical_d2phi_dx2(x: np.ndarray, x_e: float, y_e: float, sigma: float) -> np.ndarray:
    """Closed-form second spatial derivative of point-source phi_e [mV/um^2].

    phi_e [mV] = 1e3 / (4*pi*sigma * r)  for I = 1 uA
    d^2phi_e/dx^2 = C * (2*(x-x_e)^2 - y_e^2) / ((x-x_e)^2 + y_e^2)^(5/2)
    """
    C = 1e3 / (4 * np.pi * sigma)
    dx = x - x_e
    r2 = dx**2 + y_e**2
    return C * (2 * dx**2 - y_e**2) / r2**2.5


@pytest.mark.slow
def test_analytical_activating_function_interior() -> None:
    """G @ phi_e matches analytical d^2phi_e/dx^2 for interior compartments.

    For a uniform cable with axial conductance g and spacing dx:
        (G @ phi_e)[i] = g * (phi_e[i-1] - 2*phi_e[i] + phi_e[i+1])
                       ~= g * dx^2 * d^2phi_e/dx^2
    This should match to within O(dx^2) truncation error.
    """
    ncomp = 200
    total_length = 4000.0
    sigma = 0.3
    y_e = 1000.0

    branch = _make_uniform_cable(ncomp, total_length)
    params = cast(ECSParameters, branch.get_all_parameters(pstate=[]))
    G = np.asarray(build_voltage_operator_G(branch, params), dtype=np.float64)

    ensure_compartment_centers(branch)
    comp_xyz = np.asarray(get_compartment_xyz(branch), dtype=np.float64)
    x = comp_xyz[:, 0]
    dx = float(x[1] - x[0])
    x_e = total_length / 2.0

    # g_axial: off-diagonal entry for interior compartment
    g_ax = G[ncomp // 2, ncomp // 2 - 1]

    # Compute phi_e in float64 for I = 1 uA
    C = 1e3 / (4 * np.pi * sigma)
    distances = np.sqrt((x - x_e) ** 2 + y_e**2)
    phi_e = C / distances

    Gphi = G @ phi_e
    d2phi = _analytical_d2phi_dx2(x, x_e, y_e, sigma)
    analytical = g_ax * dx**2 * d2phi

    # Compare central 10% where truncation error is smallest
    lo = int(ncomp * 0.45)
    hi = int(ncomp * 0.55)
    np.testing.assert_allclose(
        Gphi[lo:hi],
        analytical[lo:hi],
        rtol=0.01,
        err_msg="G @ phi_e should match analytical activating function to ~1%",
    )


@pytest.mark.slow
def test_analytical_activating_function_convergence() -> None:
    """Truncation error between G @ phi_e and analytical decreases as O(dx^2).

    Halving dx (doubling ncomp) should reduce the relative error by ~4x,
    confirming second-order finite-difference convergence.
    """
    total_length = 4000.0
    sigma = 0.3
    y_e = 1000.0
    x_e = total_length / 2.0

    ncomps = [50, 100, 200, 400]
    errors: list[float] = []

    for ncomp in ncomps:
        branch = _make_uniform_cable(ncomp, total_length)
        params = cast(ECSParameters, branch.get_all_parameters(pstate=[]))
        G = np.asarray(build_voltage_operator_G(branch, params), dtype=np.float64)
        ensure_compartment_centers(branch)
        comp_xyz = np.asarray(get_compartment_xyz(branch), dtype=np.float64)
        x = comp_xyz[:, 0]
        dx = float(x[1] - x[0])
        g_ax = G[ncomp // 2, ncomp // 2 - 1]

        C = 1e3 / (4 * np.pi * sigma)
        distances = np.sqrt((x - x_e) ** 2 + y_e**2)
        phi_e = C / distances

        Gphi = G @ phi_e
        d2phi = _analytical_d2phi_dx2(x, x_e, y_e, sigma)
        analytical = g_ax * dx**2 * d2phi

        # Central 10% for clean convergence
        lo = int(ncomp * 0.45)
        hi = int(ncomp * 0.55)
        rel_err = float(
            np.max(np.abs(Gphi[lo:hi] - analytical[lo:hi]) / (np.abs(analytical[lo:hi]) + 1e-30))
        )
        errors.append(rel_err)

    # Consecutive doublings should give ~4x error reduction
    for i in range(1, len(errors)):
        ratio = errors[i - 1] / errors[i]
        assert 2.5 < ratio < 6.0, (
            f"Expected ~4x error reduction from ncomp={ncomps[i - 1]} to "
            f"{ncomps[i]}, got {ratio:.2f}x (errors: {errors[i - 1]:.6f} -> "
            f"{errors[i]:.6f})"
        )
