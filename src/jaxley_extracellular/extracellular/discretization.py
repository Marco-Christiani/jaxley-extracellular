"""Build the Jaxley-consistent voltage diffusion operator G.

G entries are in 1/ms, matching Jaxley's cable ODE:

    dv/dt [mV/ms] = G [1/ms] @ v [mV] + membrane_terms
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np
from jax import Array

from jaxley_extracellular.extracellular.typing_helpers import ECSParameters


def build_voltage_operator_G(module: Any, params: ECSParameters) -> Array:
    """Return the dense voltage diffusion operator G [1/ms], shape (Ncomp, Ncomp).

    Delegates to Jaxley's own ``_compute_transition_matrix`` and strips
    branchpoint pseudo-nodes, exactly as Jaxley does in
    ``build_exp_euler_transition_matrix``.

    Args:
        module: A top-level Jaxley module (Compartment, Branch, Cell, Network)
                *after* calling ``module.to_jax()``.  Must NOT be a view
                (i.e. ``module.base is module``).
        params: Output of ``module.get_all_parameters(pstate=[])``.

    Returns:
        G: jax.Array of shape (Ncomp, Ncomp), where
           Ncomp = len(module.base._internal_node_inds).
    """
    axial_conds_v: Array = params["axial_conductances"]["v"]  # (C,)
    base = module.base
    n_nodes: int = int(base._n_nodes)
    idx: np.ndarray = np.asarray(base._internal_node_inds)  # (Ncomp,)

    # Single compartment has no edges and Jaxley's indexer chokes on empty
    #  float arrays.
    if len(axial_conds_v) == 0:
        return jnp.zeros((len(idx), len(idx)))

    vals, rows, cols = base._compute_transition_matrix(axial_conds_v)

    # Assemble dense matrix and strip branchpoint pseudo-node rows/cols
    G_full: Array = jnp.zeros((n_nodes, n_nodes)).at[(rows, cols)].add(vals)
    return G_full[jnp.ix_(idx, idx)]  # (Ncomp, Ncomp)
