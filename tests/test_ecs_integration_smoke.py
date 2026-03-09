"""End-to-end integration smoke tests for extracellular stimulation.

Tests
-----
1. Zero phi_e => voltage identical to baseline (no stimulus).
2. Uniform phi_e => voltage identical to baseline (row-sums-zero property).
3. Non-uniform phi_e changes voltage relative to baseline.
4. point_source_potential units sanity check.
5. Full pipeline (field + adapter + jx.integrate) on a cable.
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
import jaxley as jx
import pytest

from jaxley_extracellular.extracellular.field import point_source_potential
from jaxley_extracellular.extracellular.jaxley_adapter import (
    build_ecs_stimuli_nA,
    ensure_compartment_centers,
    get_compartment_xyz,
    package_data_stimuli,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

DT = 0.025   # ms
T_MAX = 2.0  # ms -- short run to keep tests fast
N_STEPS = int(T_MAX / DT)


def _make_recorded_branch(ncomp: int = 4) -> jx.Branch:
    comp = jx.Compartment()
    branch = jx.Branch(comp, ncomp=ncomp)
    branch.set("length", 100.0)
    branch.set("radius", 1.0)
    branch.set("axial_resistivity", 100.0)
    branch.set("capacitance", 1.0)
    branch.set("v", -65.0)
    branch.init_states()
    # Record all compartments so jx.integrate returns voltage traces.
    for i in range(ncomp):
        branch.comp(i).record(verbose=False)
    return branch


def _baseline_voltage(branch: jx.Branch) -> np.ndarray:
    """Integrate with no stimulus and return voltage traces."""
    v = jx.integrate(branch, delta_t=DT, t_max=T_MAX, solver="bwd_euler")
    return np.asarray(v)


def _integrate_with_ecs(branch: jx.Branch, phi_e_mV: jnp.ndarray) -> np.ndarray:
    """Compute ECS stimuli from phi_e and run jx.integrate."""
    i_ecs = build_ecs_stimuli_nA(branch, phi_e_mV)
    data_stimuli = package_data_stimuli(branch, i_ecs)
    v = jx.integrate(
        branch, delta_t=DT, t_max=T_MAX, data_stimuli=data_stimuli, solver="bwd_euler"
    )
    return np.asarray(v)


# ---------------------------------------------------------------------------
# 1. Zero phi_e => baseline match
# ---------------------------------------------------------------------------


def test_zero_phi_e_matches_baseline():
    """With phi_e = 0, integrate should produce identical voltage to baseline."""
    branch = _make_recorded_branch(ncomp=4)
    baseline = _baseline_voltage(branch)

    Ncomp = 4
    phi_e_zero = jnp.zeros((Ncomp, N_STEPS))
    branch.set("v", -65.0)
    branch.init_states()
    ecs_v = _integrate_with_ecs(branch, phi_e_zero)

    np.testing.assert_allclose(ecs_v, baseline, atol=1e-6)


# ---------------------------------------------------------------------------
# 2. Uniform phi_e => baseline match
# ---------------------------------------------------------------------------


def test_uniform_phi_e_matches_baseline():
    """Spatially uniform phi_e has G @ phi_e = 0, so voltage must equal baseline."""
    branch = _make_recorded_branch(ncomp=4)
    baseline = _baseline_voltage(branch)

    Ncomp = 4
    # Use a time-varying but spatially uniform phi_e (any waveform works).
    waveform = jnp.sin(jnp.linspace(0.0, 2 * jnp.pi, N_STEPS))  # (T,)
    phi_e_uniform = jnp.ones((Ncomp, 1)) * waveform[jnp.newaxis, :]  # (Ncomp, T)

    branch.set("v", -65.0)
    branch.init_states()
    ecs_v = _integrate_with_ecs(branch, phi_e_uniform)

    np.testing.assert_allclose(ecs_v, baseline, atol=1e-5)


# ---------------------------------------------------------------------------
# 3. Non-uniform phi_e changes voltage
# ---------------------------------------------------------------------------


def test_nonuniform_phi_e_changes_voltage():
    """A spatial phi_e gradient must produce a measurable voltage change."""
    branch = _make_recorded_branch(ncomp=4)
    baseline = _baseline_voltage(branch)

    Ncomp = 4
    # Linearly increasing phi_e: gradient drives current.
    gradient = np.linspace(-1.0, 1.0, Ncomp)  # mV spatial profile
    phi_e = jnp.asarray(gradient[:, np.newaxis]) * jnp.ones((1, N_STEPS))

    branch.set("v", -65.0)
    branch.init_states()
    ecs_v = _integrate_with_ecs(branch, phi_e)

    # Voltage traces must differ from baseline.
    max_diff = np.abs(ecs_v - baseline).max()
    assert max_diff > 1e-6, f"Expected voltage change but max_diff={max_diff}"


# ---------------------------------------------------------------------------
# 4. point_source_potential sanity check
# ---------------------------------------------------------------------------


def test_point_source_potential_units():
    """Verify phi_e magnitude against hand-calculated value.

    Setup: I=1 uA, sigma=0.3 S/m, r=100 um
    Expected: phi_e = 1e3 / (4 pi * 0.3 * 100) ~= 2.653 mV
    """
    comp_xyz = np.array([[100.0, 0.0, 0.0]])  # 100 um away
    electrode_pos = np.array([0.0, 0.0, 0.0])
    I = jnp.ones((1,))  # 1 uA, T=1 step
    sigma = 0.3  # S/m

    phi_e = point_source_potential(comp_xyz, electrode_pos, I, sigma)
    expected = 1e3 / (4.0 * np.pi * 0.3 * 100.0)

    assert phi_e.shape == (1, 1)
    np.testing.assert_allclose(float(phi_e[0, 0]), expected, rtol=1e-6)


def test_point_source_potential_shape():
    Ncomp = 5
    T = 20
    comp_xyz = np.random.default_rng(0).uniform(size=(Ncomp, 3)) * 200.0
    electrode_pos = np.array([500.0, 0.0, 0.0])  # far away
    I = jnp.ones((T,))
    phi_e = point_source_potential(comp_xyz, electrode_pos, I, sigma=0.3)
    assert phi_e.shape == (Ncomp, T)


def test_point_source_potential_farther_smaller():
    """Farther compartments must see smaller phi_e."""
    comp_xyz = np.array([[50.0, 0.0, 0.0], [200.0, 0.0, 0.0]])
    electrode_pos = np.array([0.0, 0.0, 0.0])
    I = jnp.ones((1,))
    phi_e = point_source_potential(comp_xyz, electrode_pos, I, sigma=0.3)
    assert float(phi_e[0, 0]) > float(phi_e[1, 0])


# ---------------------------------------------------------------------------
# 5. Full pipeline on a cable
# ---------------------------------------------------------------------------


def test_full_pipeline_cable():
    """Point-source electrode -> phi_e -> i_ecs -> jx.integrate -- smoke test."""
    ncomp = 4
    branch = _make_recorded_branch(ncomp=ncomp)

    # Populate compartment centres.
    ensure_compartment_centers(branch)
    comp_xyz = get_compartment_xyz(branch)
    assert comp_xyz.shape == (ncomp, 3), f"Expected ({ncomp},3), got {comp_xyz.shape}"

    # Electrode sitting above the midpoint of the cable.
    electrode_pos = np.array([50.0, 50.0, 0.0])  # um
    T = N_STEPS
    waveform = jnp.zeros((T,)).at[10:50].set(1.0)  # rectangular 1 uA pulse

    phi_e = point_source_potential(comp_xyz, electrode_pos, waveform, sigma=0.3)
    assert phi_e.shape == (ncomp, T)

    i_ecs = build_ecs_stimuli_nA(branch, phi_e)
    assert i_ecs.shape == (ncomp, T)

    data_stimuli = package_data_stimuli(branch, i_ecs)

    branch.set("v", -65.0)
    branch.init_states()
    v = jx.integrate(
        branch, delta_t=DT, t_max=T_MAX, data_stimuli=data_stimuli, solver="bwd_euler"
    )
    v = jax.device_get(v)

    # Basic sanity: voltage stays in a physiological range during short run.
    assert v.shape[0] == ncomp, "Expected one trace per compartment"
    assert float(v.min()) > -90.0, f"Voltage too low: {float(v.min())}"
    assert float(v.max()) < 50.0, f"Voltage too high: {float(v.max())}"


# ---------------------------------------------------------------------------
# 6. ensure_compartment_centers idempotent
# ---------------------------------------------------------------------------


def test_ensure_compartment_centers_idempotent():
    """Calling ensure_compartment_centers twice should not raise or corrupt data."""
    comp = jx.Compartment()
    ensure_compartment_centers(comp)
    xyz1 = get_compartment_xyz(comp)
    ensure_compartment_centers(comp)
    xyz2 = get_compartment_xyz(comp)
    np.testing.assert_array_equal(xyz1, xyz2)
