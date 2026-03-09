"""Glue layer between Jaxley modules and the extracellular stimulation pipeline.

Coordinate preparation
----------------------
Jaxley does not guarantee that `module.nodes[["x","y","z"]]` is populated.
`ensure_compartment_centers(module)` checks and calls `compute_xyz()` /
`compute_compartment_centers()` as needed.

High-level pipeline
-------------------
    i_nA = build_ecs_stimuli_nA(module, phi_e_mV)

Stimulus injection helpers
--------------------------
    data_stimuli = package_data_stimuli(module, i_nA)
    # pass data_stimuli into jx.integrate(..., data_stimuli=data_stimuli)

Design constraints (Phase 1)
-----------------------------
- Top-level module only (no views): `module.base is module`.
- Fixed morphology: G is computed once and reused for all T timesteps.
- No Jaxley source modifications.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
from jax import Array

from jaxley_extracellular.extracellular.discretization import build_voltage_operator_G
from jaxley_extracellular.extracellular.equivalent_current import phi_e_to_ecs_nA
from jaxley_extracellular.extracellular.typing_helpers import (
    DataStimuli,
    ECSParameters,
)

# ---------------------------------------------------------------------------
# Coordinate preparation
# ---------------------------------------------------------------------------


def ensure_compartment_centers(module: Any) -> None:
    """Populate ``module.nodes[["x","y","z"]]`` if absent or NaN.

    Calls ``compute_xyz()`` if the raw ``xyzr`` traces contain NaN, then
    calls ``compute_compartment_centers()`` to interpolate midpoints.

    Args:
        module: A Jaxley module (top-level or view).  Mutates in-place.
    """
    nodes = module.base.nodes
    cols_present = all(c in nodes.columns for c in ("x", "y", "z"))
    has_valid = cols_present and not nodes[["x", "y", "z"]].isna().any(axis=None)

    if not has_valid:
        # xyzr is a list of (n_traced_pts, 4) arrays; column 3 is radius.
        raw_xyz = module.xyzr[0][:, :3]
        if np.isnan(raw_xyz).any():
            module.compute_xyz()
        module.compute_compartment_centers()


def get_compartment_xyz(module: Any) -> np.ndarray:
    """Return (Ncomp, 3) compartment-centre coordinates in um.

    Raises if coordinates have not yet been populated; call
    ``ensure_compartment_centers`` first.

    Args:
        module: Top-level Jaxley module.

    Returns:
        numpy array (Ncomp, 3), ordered by ``_internal_node_inds``.
    """
    base = module.base
    idx = np.asarray(base._internal_node_inds)
    nodes = base.nodes
    missing = any(c not in nodes.columns for c in ("x", "y", "z"))
    if missing or nodes[["x", "y", "z"]].isna().any(axis=None):
        raise RuntimeError(
            "Compartment coordinates are not populated.  "
            "Call ensure_compartment_centers(module) first."
        )
    # pandas-stubs cannot infer DataFrame->ndarray shape here, but runtime is a float array.
    return cast(np.ndarray, nodes.loc[idx, ["x", "y", "z"]].to_numpy(dtype=float))


# ---------------------------------------------------------------------------
# Full ECS pipeline
# ---------------------------------------------------------------------------


def build_ecs_stimuli_nA(module: Any, phi_e_mV: Array) -> Array:
    """Full pipeline: phi_e [mV] -> i_ecs [nA] for every compartment and timestep.

    Steps:
        1. ``module.to_jax()`` + ``get_all_parameters`` to obtain G, cm, area.
        2. Build voltage operator G via ``build_voltage_operator_G``.
        3. ``phi_e_to_ecs_nA`` to get the equivalent injected current.

    Args:
        module:     Top-level Jaxley module (``module.base is module``).
                    *After* ``ensure_compartment_centers`` has been called if
                    coordinates are needed upstream.
        phi_e_mV:  (Ncomp, T) extracellular potential at compartment centres
                   in mV.  Ncomp must equal ``len(module.base._internal_node_inds)``.

    Returns:
        i_ecs_nA: (Ncomp, T) equivalent stimulus current in nA.
    """
    module.to_jax()
    params: ECSParameters = module.get_all_parameters(pstate=[])

    G: Array = build_voltage_operator_G(module, params)  # (Ncomp, Ncomp)

    idx = np.asarray(module.base._internal_node_inds)
    cm: Array = params["capacitance"][idx]  # (Ncomp,) uF/cm^2
    area: Array = params["area"][idx]  # (Ncomp,) um^2

    return phi_e_to_ecs_nA(phi_e_mV, G, cm, area)  # (Ncomp, T) nA


# ---------------------------------------------------------------------------
# data_stimulate packaging
# ---------------------------------------------------------------------------


def package_data_stimuli(module: Any, i_nA: Array) -> DataStimuli:
    """Wrap ``i_nA`` into the ``data_stimuli`` tuple expected by ``jx.integrate``.

    Equivalent to calling ``module.data_stimulate(i_nA, data_stimuli=None)``.

    Args:
        module: Top-level Jaxley module.
        i_nA:   (Ncomp, T) stimulus current in nA.

    Returns:
        data_stimuli tuple for passing into ``jx.integrate``.
    """
    # Jaxley returns a heterogeneous tuple consumed by jx.integrate as-is.
    return cast(DataStimuli, module.data_stimulate(i_nA, data_stimuli=None))
