"""High-level experiment runner for ECS waveform sweeps.

Provides:
- ``setup_hh_cable``: build a long HH cable with ECS infrastructure
- ``simulate_waveform``: single waveform -> voltage trace -> features
- ``run_sweep``: vmap over a batch of waveforms

All functions are designed to be JIT/vmap-compatible for the traced
parts (waveform -> features).  Model construction and G computation
happen outside the traced region.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Any, cast

import jax
import jax.numpy as jnp
import jaxley as jx
import jaxley.channels as ch
import numpy as np
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

# ---------------------------------------------------------------------------
# Model setup (not traced -- runs once)
# ---------------------------------------------------------------------------


class ECSExperiment:
    """Pre-computed static parts of an extracellular stimulation experiment.

    Attributes
    ----------
    module : jx.Module
        Jaxley module (Branch/Cell/Network), already ``.to_jax()``-ed.
    comp_xyz : Array, (Ncomp, 3)
    G : Array, (Ncomp, Ncomp)
    cm, area : Array, (Ncomp,)
    dt_ms, T_ms : float
    T : int
        Number of timesteps.
    """

    def __init__(
        self,
        module: Any,
        electrode_pos: Array,
        sigma: float = 0.3,
        dt_ms: float = 0.025,
        T_ms: float = 5.0,
    ):
        ensure_compartment_centers(module)
        self.comp_xyz = jnp.asarray(get_compartment_xyz(module))
        self.electrode_pos = jnp.asarray(electrode_pos)
        self.sigma = sigma
        self.dt_ms = dt_ms
        self.T_ms = T_ms
        self.T = int(T_ms / dt_ms)

        module.to_jax()
        params: ECSParameters = module.get_all_parameters(pstate=[])
        self.G = build_voltage_operator_G(module, params)
        idx = np.asarray(module.base._internal_node_inds)
        self.cm = params["capacitance"][idx]
        self.area = params["area"][idx]

        self.module = module

    # ------------------------------------------------------------------
    # Single simulation (traced)
    # ------------------------------------------------------------------

    def simulate_waveform(self, waveform: Array) -> Array:
        """Run one ECS simulation and return voltage traces.

        Parameters
        ----------
        waveform : Array, shape ``(T,)``
            Electrode current in uA.

        Returns
        -------
        v : Array, shape ``(Ncomp, T+1)``
            Voltage at every compartment over time.
        """
        phi_e = point_source_potential(
            self.comp_xyz,
            self.electrode_pos,
            waveform,
            self.sigma,
        )
        i_ecs = phi_e_to_ecs_nA(phi_e, self.G, self.cm, self.area)
        data_stimuli = self.module.data_stimulate(i_ecs)
        v = jx.integrate(
            self.module,
            delta_t=self.dt_ms,
            t_max=self.T_ms,
            data_stimuli=data_stimuli,
            solver="bwd_euler",
        )
        # mypy treats jx.integrate as Any (untyped third-party API), so cast at boundary.
        return cast(Array, v)  # pyright: ignore[reportUnnecessaryCast]

    def simulate_and_extract(
        self,
        waveform: Array,
        record_comp: int = 0,
        threshold_mV: float = 0.0,
    ) -> dict[str, Array]:
        """Run simulation and extract response features at one compartment.

        Parameters
        ----------
        waveform : Array, shape ``(T,)``
        record_comp : int
            Which compartment to extract features from.
        threshold_mV : float

        Returns
        -------
        dict with keys: spiked, latency_ms, vmax, vmin
        """
        v = self.simulate_waveform(waveform)
        return extract_response_features(
            v[record_comp],
            self.dt_ms,
            threshold_mV,
        )

    # ------------------------------------------------------------------
    # Batched sweep (vmap)
    # ------------------------------------------------------------------

    def run_sweep(
        self,
        waveforms: Array,
        record_comp: int = 0,
        threshold_mV: float = 0.0,
    ) -> dict[str, Array]:
        """Run a batch of waveforms via ``jit(vmap(...))``.

        Parameters
        ----------
        waveforms : Array, shape ``(B, T)``
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
        # pyright cannot infer vmapped dict outputs precisely; runtime shape is validated by tests.
        return cast(dict[str, Array], run_batch(waveforms))

    # ------------------------------------------------------------------
    # Threshold search (vectorised binary search)
    # ------------------------------------------------------------------

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
            ``(amplitude: float) -> Array of shape (T,)``.  Must be
            jit/vmap-compatible.  Receives a scalar amplitude and returns
            the full waveform.
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
            # pyright widens vmapped return types; cast keeps the binary-search arrays typed.
            spiked: Array = cast(Array, test_amplitude(mid))
            lo = jnp.where(spiked, lo, mid)
            hi = jnp.where(spiked, mid, hi)

        return (lo + hi) / 2.0


# ---------------------------------------------------------------------------
# Convenience: standard HH cable experiment
# ---------------------------------------------------------------------------


def make_hh_cable_experiment(
    ncomp: int = 50,
    cable_length_um: float = 1250.0,
    radius_um: float = 10.0,
    axial_resistivity: float = 100.0,
    electrode_pos: tuple[float, float, float] | None = None,
    electrode_distance_um: float = 50.0,
    sigma: float = 0.3,
    dt_ms: float = 0.025,
    T_ms: float = 5.0,
) -> ECSExperiment:
    """Build a standard HH cable with electrode above one end.

    Parameters
    ----------
    ncomp : int
    cable_length_um : float
    radius_um : float
    axial_resistivity : float
    electrode_pos : tuple or None
        If None, electrode placed perpendicular above the first compartment.
    electrode_distance_um : float
        Only used when *electrode_pos* is None.
    sigma : float
    dt_ms : float
    T_ms : float

    Returns
    -------
    ECSExperiment
    """
    comp = jx.Compartment()
    branch = jx.Branch(comp, ncomp=ncomp)
    branch.set("length", cable_length_um)
    branch.set("radius", radius_um)
    branch.set("axial_resistivity", axial_resistivity)
    branch.set("capacitance", 1.0)
    branch.set("v", -65.0)
    branch.insert(ch.HH())
    branch.init_states()
    for i in range(ncomp):
        branch.comp(i).record(verbose=False)

    ensure_compartment_centers(branch)
    comp_xyz = get_compartment_xyz(branch)

    if electrode_pos is None:
        electrode_pos = (
            float(comp_xyz[0, 0]),
            electrode_distance_um,
            0.0,
        )
    electrode_pos_arr = jnp.array(electrode_pos)

    return ECSExperiment(
        branch,
        electrode_pos_arr,
        sigma,
        dt_ms,
        T_ms,
    )
