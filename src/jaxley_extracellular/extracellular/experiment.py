"""High-level experiment runner for ECS waveform sweeps."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Any, cast

import jax
import jax.numpy as jnp
import jaxley as jx
import jaxley.channels as ch
import numpy as np
from jax.experimental.sparse import BCOO
from jaxtyping import Array

from jaxley_extracellular.extracellular.discretization import build_voltage_operator_G
from jaxley_extracellular.extracellular.equivalent_current import phi_e_to_ecs_nA
from jaxley_extracellular.extracellular.field import point_source_potential
from jaxley_extracellular.extracellular.jaxley_adapter import (
    ensure_compartment_centers,
    get_compartment_xyz,
)
from jaxley_extracellular.extracellular.response import (
    extract_response_features,
)
from jaxley_extracellular.extracellular.typing_helpers import ECSParameters


class ECSExperiment:
    """Pre-computed static parts of an extracellular stimulation experiment.

    Attributes
    ----------
    module : jx.Module
        Jaxley module (Branch/Cell/Network), already ``.to_jax()``-ed.
    comp_xyz : Array, (Ncomp, 3)
    electrode_positions : Array, (N_elec, 3)
    G : Array, (Ncomp, Ncomp)
    cm, area : Array, (Ncomp,)
    dt_ms, T_ms : float
    T : int
        Number of timesteps.
    """

    def __init__(
        self,
        module: Any,
        electrode_positions: Array,
        sigma: float = 0.3,
        dt_ms: float = 0.025,
        T_ms: float = 5.0,
    ):
        """Set up an ECS experiment.

        Parameters
        ----------
        module : jx.Module
            Jaxley module with channels inserted and states initialised.
            Will be ``.to_jax()``-ed internally.
        electrode_positions : Array, shape ``(N_elec, 3)``
            Electrode xyz positions in um.
        sigma : float
            Extracellular conductivity in S/m (default 0.3, typical cortex).
        dt_ms : float
            Simulation timestep in ms.
        T_ms : float
            Total simulation duration in ms.
        """
        ensure_compartment_centers(module)
        self.comp_xyz = jnp.asarray(get_compartment_xyz(module))
        self.electrode_positions = jnp.asarray(electrode_positions)
        self.sigma = sigma
        self.dt_ms = dt_ms
        self.T_ms = T_ms
        self.T = int(T_ms / dt_ms)

        module.to_jax()
        params: ECSParameters = module.get_all_parameters(pstate=[])
        # G is structurally sparse (tree morphology, O(Ncomp) nonzeros).
        # BCOO keeps `G @ phi_e` cheap under vmap on TPU. Dense G triggers
        # an XLA broadcast to (B, Ncomp, Ncomp) that OOMs at Ncomp~17500.
        self.G = BCOO.fromdense(build_voltage_operator_G(module, params))
        idx = np.asarray(module.base._internal_node_inds)
        self.cm = params["capacitance"][idx]
        self.area = params["area"][idx]

        self.module = module

    # Single simulation (traced)

    def simulate_waveform(
        self,
        waveforms: Array,
        checkpoint_lengths: list[int] | None = None,
    ) -> Array:
        """Run one ECS simulation and return voltage traces.

        Parameters
        ----------
        waveforms : Array, shape ``(N_elec, T)``
            Per-electrode current waveforms in uA.
        checkpoint_lengths : list[int] or None
            Forwarded to ``jx.integrate``. Pass a factorisation of T whose
            product is >= the number of timesteps to enable hierarchical
            scan-rematerialisation; reduces HBM at the cost of recompute.
            ``None`` (default) = no checkpointing.

        Returns
        -------
        v : Array, shape ``(Ncomp, T+1)``
            Voltage at every compartment over time.
        """
        phi_e = point_source_potential(  # (Ncomp, T) [mV]
            self.comp_xyz,           # (Ncomp, 3) [um]
            self.electrode_positions,  # (N_elec, 3) [um]
            waveforms,               # (N_elec, T) [uA]
            self.sigma,              # [S/m]
        )
        i_ecs = phi_e_to_ecs_nA(phi_e, self.G, self.cm, self.area)  # (Ncomp, T) [nA]
        data_stimuli = self.module.data_stimulate(i_ecs)
        v = jx.integrate(
            self.module,
            delta_t=self.dt_ms,
            t_max=self.T_ms,
            data_stimuli=data_stimuli,
            solver="bwd_euler",
            checkpoint_lengths=checkpoint_lengths,
        )
        # mypy treats jx.integrate as Any (untyped third-party API)
        return cast(Array, v)  # pyright: ignore[reportUnnecessaryCast]

    def simulate_and_extract(
        self,
        waveforms: Array,
        record_comp: int = 0,
        threshold_mV: float = 0.0,
        checkpoint_lengths: list[int] | None = None,
    ) -> dict[str, Array]:
        """Run simulation and extract response features at one compartment.

        Parameters
        ----------
        waveforms : Array, shape ``(N_elec, T)``
        record_comp : int
            Which compartment to extract features from.
        threshold_mV : float
        checkpoint_lengths : list[int] or None
            See :meth:`simulate_waveform`.

        Returns
        -------
        dict with keys: spiked, latency_ms, vmax, vmin
        """
        v = self.simulate_waveform(waveforms, checkpoint_lengths=checkpoint_lengths)
        return extract_response_features(
            v[record_comp],
            self.dt_ms,
            threshold_mV,
        )

    # Batched sweep (vmap)

    def run_sweep(
        self,
        waveforms: Array,
        record_comp: int = 0,
        threshold_mV: float = 0.0,
    ) -> dict[str, Array]:
        """Run a batch of waveforms via ``jit(vmap(...))``.

        Parameters
        ----------
        waveforms : Array, shape ``(B, N_elec, T)``
        record_comp : int
        threshold_mV : float

        Returns
        -------
        dict with keys mapping to Arrays of shape ``(B,)``.
        """

        run_one = partial(
            self.simulate_and_extract,
            record_comp=record_comp,
            threshold_mV=threshold_mV,
        )
        run_batch = jax.jit(jax.vmap(run_one))
        # pyright cannot infer vmapped dict outputs precisely (runtime shape is validated by tests)
        return cast(dict[str, Array], run_batch(waveforms))

    # Threshold search (vectorised binary search)

    def find_thresholds(
        self,
        make_waveform_fn: Callable[[Array], Array],
        amp_lo: Array,
        amp_hi: Array,
        n_iter: int = 10,
        record_comp: int = 0,
        threshold_mV: float = 0.0,
    ) -> Array:
        """Vectorised binary search for activation threshold amplitude.

        Parameters
        ----------
        make_waveform_fn : callable
            ``(amplitude: float) -> Array of shape (N_elec, T)``.  Must be
            jit/vmap-compatible.  Receives a scalar amplitude and returns
            the full per-electrode waveforms.
        amp_lo, amp_hi : Array, shape ``(N,)``
            Initial lower and upper brackets per configuration.
            ``amp_lo`` should be sub-threshold, ``amp_hi`` supra-threshold.
        n_iter : int
            Number of bisection iterations (precision ~= range / 2**n_iter).
        record_comp : int
        threshold_mV : float

        Returns
        -------
        thresholds : Array, shape ``(N,)``
            Estimated threshold amplitude for each configuration.
        """
        lo = jnp.asarray(amp_lo, dtype=jnp.float32)
        hi = jnp.asarray(amp_hi, dtype=jnp.float32)

        def _is_spiked_for_amplitude(amp: Array) -> Array:
            w: Array = make_waveform_fn(amp)
            feats = self.simulate_and_extract(w, record_comp, threshold_mV)
            return feats["spiked"]

        test_amplitude = jax.jit(jax.vmap(_is_spiked_for_amplitude))

        for _ in range(n_iter):
            mid = (lo + hi) / 2.0
            # pyright widens vmapped return types
            spiked: Array = cast(Array, test_amplitude(mid))
            lo = jnp.where(spiked, lo, mid)
            hi = jnp.where(spiked, mid, hi)

        return (lo + hi) / 2.0


# Convenience: standard HH cable experiment


def make_hh_cable_experiment(
    ncomp: int = 50,
    cable_length_um: float = 1250.0,
    radius_um: float = 10.0,
    axial_resistivity: float = 100.0,
    electrode_positions: np.ndarray | None = None,
    electrode_distance_um: float = 50.0,
    sigma: float = 0.3,
    dt_ms: float = 0.025,
    T_ms: float = 5.0,
) -> ECSExperiment:
    """Build a uniform Hodgkin-Huxley cable wrapped as an ECS experiment.

    Parameters
    ----------
    ncomp : int, optional
        Number of compartments along the cable (default 50).
    cable_length_um : float, optional
        Total cable length in um (default 1250).
    radius_um : float, optional
        Fibre radius in um (default 10).
    axial_resistivity : float, optional
        Intracellular axial resistivity in Ohm-cm (default 100).
    electrode_positions : numpy.ndarray, shape ``(N_elec, 3)``, or None, optional
        Explicit electrode positions in um. If ``None`` (default), a
        single electrode is placed perpendicular above the first
        compartment at ``electrode_distance_um``.
    electrode_distance_um : float, optional
        Default placement distance from the first compartment, used
        only when ``electrode_positions`` is ``None`` (default 50).
    sigma : float, optional
        Extracellular conductivity in S/m (default 0.3, typical
        cortical grey matter).
    dt_ms : float, optional
        Simulation timestep in ms (default 0.025).
    T_ms : float, optional
        Total simulation duration in ms (default 5).

    Returns
    -------
    ECSExperiment
        Experiment object ready for ``simulate_waveform`` /
        ``simulate_and_extract`` / ``find_thresholds`` calls.
    """
    comp = jx.Compartment()
    branch = jx.Branch(comp, ncomp=ncomp)
    # NB: this is per-compartment, not total cable length
    branch.set("length", cable_length_um / ncomp)
    branch.set("radius", radius_um)
    branch.set("axial_resistivity", axial_resistivity)
    branch.set("capacitance", 1.0)
    branch.set("v", -65.0)
    branch.insert(ch.HH())
    branch.init_states()
    for i in range(ncomp):
        branch.comp(i).record(verbose=False)

    # uses compute_xyz() which reads nodes["length"] and sets xyzr to match,
    #  then interpolates compartment centers
    ensure_compartment_centers(branch)
    comp_xyz = get_compartment_xyz(branch)

    if electrode_positions is None:
        electrode_positions = np.array([[
            float(comp_xyz[0, 0]),
            electrode_distance_um,
            0.0,
        ]])
    positions_arr = jnp.array(electrode_positions)

    return ECSExperiment(
        branch,
        positions_arr,
        sigma,
        dt_ms,
        T_ms,
    )


def _first_global_comp_index(view: Any) -> int:
    """Return the first global compartment index for a Jaxley view."""
    global_idx = view.nodes["global_comp_index"].to_numpy()
    if global_idx.size == 0:
        raise ValueError("recording view has no compartments")
    return int(global_idx[0])


def make_bbp_pyr_experiment(
    ncomp: int = 2,
    max_branch_len: float | None = 100.0,
    electrode_positions: np.ndarray | None = None,
    electrode_distance_um: float = 100.0,
    record_site: str = "soma",
    sigma: float = 0.3,
    dt_ms: float = 0.025,
    T_ms: float = 5.0,
    v_init_mv: float = -75.0,
) -> tuple[ECSExperiment, int]:
    """Build a BBP L2/3 pyramidal cell ECS experiment.

    Parameters
    ----------
    ncomp : int, optional
        Compartments per SWC branch (default 2).
    max_branch_len : float or None, optional
        Branch-subdivision length in um. See
        :func:`jaxley_extracellular.bbp.make_pyr_cell` (default 100).
    electrode_positions : numpy.ndarray, shape ``(N_elec, 3)``, or None, optional
        Explicit electrode positions in um. If ``None`` (default), a
        single electrode is placed at ``electrode_distance_um`` above
        the soma centre along the +y axis.
    electrode_distance_um : float, optional
        Default electrode displacement from the soma centre, used only
        when ``electrode_positions`` is ``None`` (default 100).
    record_site : str, optional
        Named BBP group exposed by the Jaxley cell, e.g.\ ``"soma"``,
        ``"apical"``, ``"basal"``, ``"axon"`` (default ``"soma"``). The
        first compartment of branch 0 in that group is used as the
        canonical readout site.
    sigma : float, optional
        Extracellular conductivity in S/m (default 0.3).
    dt_ms : float, optional
        Simulation timestep in ms (default 0.025).
    T_ms : float, optional
        Total simulation duration in ms (default 5).
    v_init_mv : float, optional
        Initial transmembrane voltage in mV (default ``-75``).

    Returns
    -------
    experiment : ECSExperiment
        Configured experiment object.
    record_comp : int
        Global compartment index of the canonical readout site, for
        passing to ``ECSExperiment.simulate_and_extract``.

    Raises
    ------
    ValueError
        If ``record_site`` does not name a group on the Pyr cell.
    """
    from jaxley_extracellular.bbp.cell_factory import make_pyr_cell

    cell = make_pyr_cell(ncomp=ncomp, max_branch_len=max_branch_len)
    cell.set("v", v_init_mv)
    cell.record("v", verbose=False)

    soma_xyz = cell.soma.nodes[["x", "y", "z"]].mean().to_numpy()  # pyright: ignore[reportOptionalMemberAccess]
    if electrode_positions is None:
        electrode_positions = np.array([[
            float(soma_xyz[0]),
            float(soma_xyz[1]) + electrode_distance_um,
            float(soma_xyz[2]),
        ]])
    positions_arr = jnp.array(electrode_positions)

    if not hasattr(cell, record_site):
        raise ValueError(f"unknown BBP record site {record_site!r}")
    record_view = getattr(cell, record_site).branch(0).comp(0)
    record_comp = _first_global_comp_index(record_view)

    exp = ECSExperiment(
        cell,
        positions_arr,
        sigma,
        dt_ms,
        T_ms,
    )
    return exp, record_comp
