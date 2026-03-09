"""Build the Jaxley-consistent voltage diffusion operator G.

This module replicates Jaxley's internal `_compute_transition_matrix` /
`build_exp_euler_transition_matrix` logic to reconstruct **G itself** (not
the matrix exponential) from a top-level Jaxley module.

Key public function
-------------------
    build_voltage_operator_G(module, params) -> jax.Array  shape (Ncomp, Ncomp)

Units
-----
    G entries are in 1/ms  (equivalently mS/cm^2 divided by uF/cm^2, = kHz).
    This is exactly the coefficient used in Jaxley's cable ODE:
        dv/dt [mV/ms] = G [1/ms] @ v [mV]  +  membrane_terms

Private internals accessed
--------------------------
    module.base._comp_edges       -- pandas DataFrame of directed edges
    module.base._n_nodes          -- int, total nodes including branchpoints
    module.base._internal_node_inds -- int array, indices of real compartments
    params["axial_conductances"]["v"] -- per-edge conductances, shape (C,)

All of these are set during normal Jaxley module initialisation and are
stable across Jaxley 0.8.x.  We do NOT use `_exp_euler_solve_indexer`
directly; instead we replicate the same indexer logic from `_comp_edges`.
"""

from __future__ import annotations

from typing import Any, cast

import jax.numpy as jnp
import numpy as np
from jax import Array
from numpy.typing import NDArray

from jaxley_extracellular.extracellular.typing_helpers import ECSParameters

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_voltage_operator_G(module: Any, params: ECSParameters) -> Array:
    """Return the dense voltage diffusion operator G [1/ms], shape (Ncomp, Ncomp).

    Args:
        module: A top-level Jaxley module (Compartment, Branch, Cell, Network)
                *after* calling `module.to_jax()`.  Must NOT be a view
                (i.e. `module.base is module`).
        params: Output of `module.get_all_parameters(pstate=[])`.

    Returns:
        G: jax.Array of shape (Ncomp, Ncomp), where
           Ncomp = len(module.base._internal_node_inds).
           Branchpoint pseudo-nodes are eliminated via Gaussian substitution,
           exactly as Jaxley does internally.
    """
    axial_conds_v: Array = params["axial_conductances"]["v"]  # (C,)
    base = module.base
    comp_edges = base._comp_edges
    n_nodes: int = int(base._n_nodes)
    idx: np.ndarray = np.asarray(base._internal_node_inds)  # (Ncomp,)

    vals, rows, cols = _build_G_coo(axial_conds_v, comp_edges, n_nodes)

    G_full: Array = jnp.zeros((n_nodes, n_nodes)).at[(rows, cols)].add(vals)
    # Strip branchpoint rows/cols -- identical to Jaxley's build_dense
    return G_full[jnp.ix_(idx, idx)]  # (Ncomp, Ncomp)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_G_coo(
    axial_conductances_v: Array, comp_edges: object, n_nodes: int
) -> tuple[Array, Array, Array]:
    """COO representation of G, mirroring Jaxley's _compute_transition_matrix.

    Returns (vals, rows, cols) where rows[k] is the row index and cols[k] is
    the column index for vals[k].  Shape of each: (n_nodes + num_offdiag + num_bp,).

    The three contributions assembled here match Jaxley's indexer sections:
        1. Diagonal        -- negative row-drain (types 0, 1, 2 sinks)
        2. Off-diagonal    -- comp-to-comp axial coupling (type 0)
        3. Branchpoint     -- through-branchpoint coupling (types 1-4 combined)
    """
    IntArray = NDArray[np.int_]

    # Jaxley exposes comp_edges as an untyped pandas DataFrame; keep one explicit dynamic boundary.
    comp_edges_df = cast(Any, comp_edges)

    # --- 1. Off-diagonals: comp-to-comp (type 0) ----------------------------
    df_offdiag = comp_edges_df.loc[comp_edges_df["type"] == 0]
    offdiag_inds: IntArray = cast(IntArray, df_offdiag.index.to_numpy(dtype=int))
    sink_offdiags: IntArray = cast(IntArray, df_offdiag["sink"].to_numpy(dtype=int))
    source_offdiags: IntArray = cast(IntArray, df_offdiag["source"].to_numpy(dtype=int))
    g_offdiags: Array = axial_conductances_v[offdiag_inds]

    # --- 2. Diagonal: negative sum of draining conductances (types 0,1,2) ---
    df_diag = comp_edges_df.loc[comp_edges_df["type"].isin([0, 1, 2])]
    diag_inds: IntArray = cast(IntArray, df_diag.index.to_numpy(dtype=int))
    diag_sinks: IntArray = cast(IntArray, df_diag["sink"].to_numpy(dtype=int))
    g_diags: Array = jnp.zeros(n_nodes).at[diag_sinks].add(-axial_conductances_v[diag_inds])

    # --- 3. Branchpoint cross-terms -----------------------------------------
    df_to_bp = comp_edges_df.loc[comp_edges_df["type"].isin([3, 4])].copy()
    df_bp_to_comp = comp_edges_df.loc[comp_edges_df["type"].isin([1, 2])].copy()

    sink_bp: IntArray
    source_bp: IntArray
    g_branchpoint: Array

    if len(df_to_bp) > 0:
        # Group comp->bp edges by destination branchpoint so we can normalise.
        df_to_bp["_group"] = df_to_bp.groupby("sink").ngroup()
        comp_to_bp_inds: IntArray = cast(IntArray, df_to_bp.index.to_numpy(dtype=int))
        comp_to_bp_group_sinks: IntArray = cast(IntArray, df_to_bp["_group"].to_numpy(dtype=int))

        bp_to_comp_inds: IntArray = cast(IntArray, df_bp_to_comp.index.to_numpy(dtype=int))

        # Outer join to enumerate all (bp->comp, comp->bp) pairs sharing a BP.
        # Index both DataFrames by the branchpoint node index so the join
        # matches on the branchpoint.
        df_b2c = df_bp_to_comp.set_index("source", drop=False).copy()
        df_b2c["_pos"] = np.arange(len(df_bp_to_comp))
        df_c2b = df_to_bp.set_index("sink", drop=False).copy()
        df_c2b["_pos"] = np.arange(len(df_to_bp))

        bp_conns = df_b2c.join(df_c2b, how="outer", lsuffix="_b2c", rsuffix="_c2b")[
            ["sink_b2c", "source_c2b", "_pos_b2c", "_pos_c2b"]
        ]

        b2c_expanded: IntArray = cast(IntArray, bp_conns["_pos_b2c"].to_numpy(dtype=int))
        c2b_expanded: IntArray = cast(IntArray, bp_conns["_pos_c2b"].to_numpy(dtype=int))
        sink_bp = cast(IntArray, bp_conns["sink_b2c"].to_numpy(dtype=int))
        source_bp = cast(IntArray, bp_conns["source_c2b"].to_numpy(dtype=int))

        # Normaliser: sum of comp->bp conductances per branchpoint group.
        g_comp_to_bp: Array = axial_conductances_v[comp_to_bp_inds]
        n_groups = int(comp_to_bp_group_sinks.max()) + 1
        normalizers: Array = jnp.zeros(n_groups).at[comp_to_bp_group_sinks].add(g_comp_to_bp)
        g_norm: Array = g_comp_to_bp / normalizers[comp_to_bp_group_sinks]

        g_bp_to_comp: Array = axial_conductances_v[bp_to_comp_inds]
        g_branchpoint = g_bp_to_comp[b2c_expanded] * g_norm[c2b_expanded]
    else:
        g_branchpoint = jnp.zeros(0)
        sink_bp = np.zeros(0, dtype=int)
        source_bp = np.zeros(0, dtype=int)

    # --- Assemble COO --------------------------------------------------------
    diag_nodes = np.arange(n_nodes)
    rows: Array = jnp.concatenate(
        [jnp.asarray(diag_nodes), jnp.asarray(sink_offdiags), jnp.asarray(sink_bp)]
    )
    cols: Array = jnp.concatenate(
        [jnp.asarray(diag_nodes), jnp.asarray(source_offdiags), jnp.asarray(source_bp)]
    )
    vals: Array = jnp.concatenate([g_diags, g_offdiags, g_branchpoint])
    return vals, rows, cols
