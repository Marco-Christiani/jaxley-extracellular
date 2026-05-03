"""Glue layer between Jaxley modules and the extracellular stimulation pipeline."""

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

# Coordinate preparation


def ensure_compartment_centers(module: Any) -> None:
    """Populate ``module.nodes[["x","y","z"]]`` if absent or NaN.

    Calls ``compute_xyz()`` if the raw ``xyzr`` traces contain NaN,
    then calls ``compute_compartment_centers()`` to interpolate
    midpoints.

    Parameters
    ----------
    module : jx.Module
        A Jaxley module (top-level or view). Mutated in place.
    """
    nodes = module.base.nodes
    cols_present = all(c in nodes.columns for c in ("x", "y", "z"))
    has_valid = cols_present and not nodes[["x", "y", "z"]].isna().any(axis=None)

    if not has_valid:
        # xyzr is a list of (n_traced_pts, 4) arrays and column 3 is radius
        raw_xyz = module.xyzr[0][:, :3]
        if np.isnan(raw_xyz).any():
            module.compute_xyz()
        module.compute_compartment_centers()


def get_compartment_xyz(module: Any) -> np.ndarray:
    """Return compartment-centre coordinates in um.

    Parameters
    ----------
    module : jx.Module
        Top-level Jaxley module. Call
        :func:`ensure_compartment_centers` first if coordinates have
        not yet been populated.

    Returns
    -------
    numpy.ndarray, shape ``(Ncomp, 3)``
        Compartment-centre coordinates in um, ordered by
        ``module.base._internal_node_inds``.

    Raises
    ------
    RuntimeError
        If coordinates are not yet populated on ``module.base.nodes``.
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


# Full ECS pipeline


def build_ecs_stimuli_nA(module: Any, phi_e_mV: Array) -> Array:
    r"""Convert per-compartment :math:`\boldsymbol{\Phi}` to current in nA.

    Wraps the three-step pipeline:

    1. Call ``module.to_jax()`` and ``get_all_parameters`` to obtain
       ``G``, ``cm``, and ``area``.
    2. Build the voltage operator ``G`` via
       :func:`build_voltage_operator_G`.
    3. Apply :func:`phi_e_to_ecs_nA` to obtain the equivalent injected
       current.

    Algebraically, this helper computes the public-API current encoding

    .. math::

        \mathbf{I}_{\mathrm{ecs}}
        =
        \left(\mathbf{c} \odot \frac{\mathbf{A}}{10^5}\right)
        \odot (G\boldsymbol{\Phi}).

    Parameters
    ----------
    module : jx.Module
        Top-level Jaxley module (``module.base is module``). Call
        :func:`ensure_compartment_centers` first if upstream code
        depends on populated coordinates.
    phi_e_mV : Array, shape ``(Ncomp, T)``
        Extracellular potential at compartment centres, in mV. ``Ncomp``
        must equal ``len(module.base._internal_node_inds)``.

    Returns
    -------
    Array, shape ``(Ncomp, T)``
        Equivalent stimulus current in nA.
    """
    module.to_jax()
    params: ECSParameters = module.get_all_parameters(pstate=[])

    G: Array = build_voltage_operator_G(module, params)  # (Ncomp, Ncomp)

    idx = np.asarray(module.base._internal_node_inds)
    cm: Array = params["capacitance"][idx]  # (Ncomp,) uF/cm^2
    area: Array = params["area"][idx]  # (Ncomp,) um^2

    return phi_e_to_ecs_nA(phi_e_mV, G, cm, area)  # (Ncomp, T) nA


# data_stimulate packaging


def package_data_stimuli(module: Any, i_nA: Array) -> DataStimuli:
    """Wrap ``i_nA`` into the ``data_stimuli`` tuple for ``jx.integrate``.

    Equivalent to calling
    ``module.data_stimulate(i_nA, data_stimuli=None)``.

    Parameters
    ----------
    module : jx.Module
        Top-level Jaxley module.
    i_nA : Array, shape ``(Ncomp, T)``
        Per-compartment stimulus current in nA.

    Returns
    -------
    DataStimuli
        Tuple in the form expected by ``jx.integrate``'s
        ``data_stimuli`` argument.
    """
    # Jaxley returns a heterogeneous tuple consumed by jx.integrate as-is.
    return cast(DataStimuli, module.data_stimulate(i_nA, data_stimuli=None))
