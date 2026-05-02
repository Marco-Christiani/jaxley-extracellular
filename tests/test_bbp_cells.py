"""Assembly and smoke tests for BBP L2/3 cell models."""

import jax.numpy as jnp
import jaxley as jx
import pytest

from jaxley_extracellular.bbp.cell_factory import PV_SWC, PYR_SWC, make_pv_cell, make_pyr_cell
from jaxley_extracellular.bbp.channel_params import (
    PV_SOMA,
    PYR_AXON,
    PYR_SOMA,
)

# Skip the whole module if bundled SWC files are missing (not yet converted).
pytestmark = pytest.mark.skipif(
    not PYR_SWC.exists() or not PV_SWC.exists(),
    reason="Bundled .swc files not found under src/jaxley_extracellular/bbp/data",
)


# ---------------------------------------------------------------------------
# Pyramidal cell
# ---------------------------------------------------------------------------


class TestPyrCell:
    @pytest.fixture(scope="class")
    def cell(self) -> jx.Cell:
        return make_pyr_cell(ncomp=2)

    def test_has_section_groups(self, cell: jx.Cell):
        """Cell should have soma, axon, apical, basal groups."""
        for group in ("soma", "axon", "apical", "basal"):
            view = getattr(cell, group, None)
            assert view is not None, f"Missing group: {group}"

    def test_soma_conductances(self, cell: jx.Cell):
        """Spot-check that soma NaTs2T conductance matches biophysics.hoc."""
        nodes = cell.soma.nodes
        g_na = nodes["NaTs2T_gNaTs2T"].values
        expected = PYR_SOMA["NaTs2T_gNaTs2T"]
        assert jnp.allclose(g_na, expected, rtol=1e-4), (
            f"NaTs2T_gNaTs2T: got {g_na[0]}, expected {expected}"
        )

    def test_axon_conductances(self, cell: jx.Cell):
        """Spot-check axon NaTaT conductance."""
        nodes = cell.axon.nodes
        g_na = nodes["NaTaT_gNaTaT"].values
        expected = PYR_AXON["NaTaT_gNaTaT"]
        assert jnp.allclose(g_na, expected, rtol=1e-4)

    def test_reversal_potentials(self, cell: jx.Cell):
        """eK should be -85, eNa should be 50 in soma."""
        nodes = cell.soma.nodes
        assert jnp.allclose(nodes["eK"].values, -85.0)
        assert jnp.allclose(nodes["eNa"].values, 50.0)

    def test_integrate_no_nan(self, cell: jx.Cell):
        """10 ms integration should produce finite voltages."""
        cell.record()
        v = jx.integrate(cell, delta_t=0.025, t_max=10.0)
        assert jnp.all(jnp.isfinite(v)), "NaN or Inf in voltage trace"

    def test_resting_potential(self, cell: jx.Cell):
        """Resting potential should be in physiological range."""
        cell.record()
        v = jx.integrate(cell, delta_t=0.025, t_max=50.0)
        v_rest = v[:, -1]  # last time point
        assert jnp.all(v_rest > -90.0), f"v_rest too low: {v_rest.min()}"
        assert jnp.all(v_rest < -40.0), f"v_rest too high: {v_rest.max()}"

    def test_spike_with_current_injection(self, cell: jx.Cell):
        """Somatic current injection should evoke a spike."""
        cell.delete_recordings()
        cell.delete_stimuli()
        cell.soma.record("v")
        cell.soma.stimulate(jx.step_current(0.5, 1.5, 2.0, 0.025, 5.0))
        v = jx.integrate(cell, delta_t=0.025, t_max=5.0)
        v_max = float(v.max())
        assert v_max > -20.0, f"No spike detected, v_max={v_max}"


# ---------------------------------------------------------------------------
# PV interneuron
# ---------------------------------------------------------------------------


class TestPVCell:
    @pytest.fixture(scope="class")
    def cell(self) -> jx.Cell:
        return make_pv_cell(ncomp=2)

    def test_has_section_groups(self, cell: jx.Cell):
        for group in ("soma", "axon", "apical", "basal"):
            view = getattr(cell, group, None)
            assert view is not None, f"Missing group: {group}"

    def test_soma_conductances(self, cell: jx.Cell):
        nodes = cell.soma.nodes
        g_na = nodes["NaTs2T_gNaTs2T"].values
        expected = PV_SOMA["NaTs2T_gNaTs2T"]
        assert jnp.allclose(g_na, expected, rtol=1e-4)

    def test_reversal_potentials(self, cell: jx.Cell):
        nodes = cell.soma.nodes
        assert jnp.allclose(nodes["eK"].values, -85.0)
        assert jnp.allclose(nodes["eNa"].values, 50.0)

    def test_integrate_no_nan(self, cell: jx.Cell):
        cell.record()
        v = jx.integrate(cell, delta_t=0.025, t_max=10.0)
        assert jnp.all(jnp.isfinite(v)), "NaN or Inf in voltage trace"

    def test_resting_potential(self, cell: jx.Cell):
        cell.record()
        v = jx.integrate(cell, delta_t=0.025, t_max=50.0)
        v_rest = v[:, -1]
        assert jnp.all(v_rest > -90.0), f"v_rest too low: {v_rest.min()}"
        assert jnp.all(v_rest < -40.0), f"v_rest too high: {v_rest.max()}"

    def test_spike_with_current_injection(self, cell: jx.Cell):
        cell.delete_recordings()
        cell.delete_stimuli()
        cell.soma.record("v")
        cell.soma.stimulate(jx.step_current(0.5, 1.5, 2.0, 0.025, 5.0))
        v = jx.integrate(cell, delta_t=0.025, t_max=5.0)
        v_max = float(v.max())
        assert v_max > -20.0, f"No spike detected, v_max={v_max}"
