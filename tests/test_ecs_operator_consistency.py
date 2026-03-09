"""Validate that build_voltage_operator_G produces a Jaxley-consistent operator.

Tests
-----
1. Cable (jx.Branch, ncomp=4): G is tridiagonal, row-sums ~= 0, symmetric.
2. Cable row/col structure: diagonal <= 0, off-diagonal >= 0.
3. Spatially uniform phi_e => G @ phi_e ~= 0  (no net forcing).
4. Branched cell (jx.Cell, two branches): row-sums ~= 0, correct through-
   branchpoint cross-terms appear in G.
5. Single compartment (jx.Compartment): G is [[0]].
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp
import jaxley as jx
import pytest

from jaxley_extracellular.extracellular.discretization import build_voltage_operator_G


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_branch(ncomp: int = 4) -> jx.Branch:
    comp = jx.Compartment()
    branch = jx.Branch(comp, ncomp=ncomp)
    branch.set("length", 100.0)   # um
    branch.set("radius", 1.0)     # um
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
    params = module.get_all_parameters(pstate=[])
    G = build_voltage_operator_G(module, params)
    return np.asarray(G)


# ---------------------------------------------------------------------------
# 1. Cable: basic structure
# ---------------------------------------------------------------------------


def test_cable_G_shape():
    branch = _make_branch(ncomp=4)
    G = _build_G(branch)
    assert G.shape == (4, 4), f"Expected (4,4), got {G.shape}"


def test_cable_G_row_sums_zero():
    """Row sums of G must be zero (current conservation / sealed ends)."""
    branch = _make_branch(ncomp=4)
    G = _build_G(branch)
    row_sums = G.sum(axis=1)
    np.testing.assert_allclose(row_sums, 0.0, atol=1e-10)


def test_cable_G_diagonal_negative():
    branch = _make_branch(ncomp=4)
    G = _build_G(branch)
    assert np.all(np.diag(G) <= 0.0), "Diagonal must be non-positive"
    # Interior compartments must have strictly negative diagonal
    assert np.all(np.diag(G)[1:-1] < 0.0)


def test_cable_G_offdiagonals_nonnegative():
    branch = _make_branch(ncomp=4)
    G = _build_G(branch)
    n = G.shape[0]
    mask = ~np.eye(n, dtype=bool)
    assert np.all(G[mask] >= 0.0), "Off-diagonal entries must be non-negative"


def test_cable_G_symmetric_for_uniform_params():
    """For a cable with identical compartments G should be symmetric."""
    branch = _make_branch(ncomp=5)
    G = _build_G(branch)
    np.testing.assert_allclose(G, G.T, atol=1e-10)


def test_cable_G_tridiagonal():
    """For a single branch there should be no entries outside the tridiagonal."""
    branch = _make_branch(ncomp=4)
    G = _build_G(branch)
    n = G.shape[0]
    for i in range(n):
        for j in range(n):
            if abs(i - j) > 1:
                assert G[i, j] == pytest.approx(0.0, abs=1e-12), (
                    f"Expected G[{i},{j}]=0, got {G[i,j]}"
                )


# ---------------------------------------------------------------------------
# 2. Uniform phi_e => zero forcing
# ---------------------------------------------------------------------------


def test_uniform_phi_e_zero_forcing():
    """A spatially uniform phi_e produces G @ phi_e = 0 (row sums = 0)."""
    branch = _make_branch(ncomp=6)
    G = _build_G(branch)
    Ncomp = G.shape[0]
    T = 50
    phi_uniform = np.ones((Ncomp, T))  # same value everywhere
    f_ecs = G @ phi_uniform
    np.testing.assert_allclose(f_ecs, 0.0, atol=1e-9)


def test_linear_phi_e_forcing_sign():
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


def test_branched_cell_G_shape():
    cell = _make_cell_two_branches()
    G = _build_G(cell)
    # 4 comps in branch 0 + 3 comps in branch 1
    assert G.shape == (7, 7), f"Expected (7,7), got {G.shape}"


def test_branched_cell_G_row_sums_zero():
    cell = _make_cell_two_branches()
    G = _build_G(cell)
    row_sums = G.sum(axis=1)
    np.testing.assert_allclose(row_sums, 0.0, atol=1e-9)


def test_branched_cell_through_branchpoint_coupling():
    """Compartments that share a branchpoint should have non-zero cross-terms."""
    cell = _make_cell_two_branches()
    G = _build_G(cell)
    # Branch 0 has comps 0-3; last comp of branch 0 is index 3.
    # Branch 1 has comps 4-6; first comp of branch 1 is index 4.
    # These two share a branchpoint: G[3,4] and G[4,3] should be > 0.
    assert G[3, 4] > 0.0, "Expected cross-branchpoint coupling G[3,4] > 0"
    assert G[4, 3] > 0.0, "Expected cross-branchpoint coupling G[4,3] > 0"


def test_branched_cell_uniform_phi_zero_forcing():
    cell = _make_cell_two_branches()
    G = _build_G(cell)
    Ncomp = G.shape[0]
    phi_uniform = np.ones((Ncomp, 10))
    f_ecs = G @ phi_uniform
    np.testing.assert_allclose(f_ecs, 0.0, atol=1e-8)


# ---------------------------------------------------------------------------
# 4. Single compartment
# ---------------------------------------------------------------------------


def test_single_compartment_G():
    comp = jx.Compartment()
    comp.to_jax()
    G = _build_G(comp)
    assert G.shape == (1, 1)
    assert G[0, 0] == pytest.approx(0.0, abs=1e-12)
