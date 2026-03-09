"""Point-source electrode field model.

Computes the extracellular potential phi_e at compartment centres due to a
monopolar point-source electrode in a homogeneous, isotropic, infinite medium
using the classical formula:

    phi_e(r, t) = I(t) / (4 pi sigma |r - r_elec|)

Units used throughout
---------------------
    Electrode position / compartment centres : um
    Electrode current                         : uA
    Extracellular conductivity (sigma)        : S/m
    Returned phi_e                            : mV

Unit derivation
---------------
    phi_e [V] = I [A] / (4 pi sigma [S/m] r [m])
    With I in uA (*1e-6) and r in um (*1e-6):
        phi_e [V] = I_uA*1e-6 / (4 pi sigma r_um*1e-6)
                  = I_uA / (4 pi sigma r_um)
    Multiply by 1e3 to convert V -> mV:
        phi_e [mV] = I_uA * 1e3 / (4 pi sigma r_um)

Public function
---------------
    point_source_potential(comp_xyz, electrode_pos, electrode_current, sigma,
                           min_distance_um=1.0) -> jax.Array  (Ncomp, T) mV
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from jax import Array

# electrode_pos may be a plain numpy array (static) or a JAX array (traced,
# e.g. when differentiating w.r.t. electrode placement).
_ArrayLike = np.ndarray | Array
_ScalarLike = float | Array


def point_source_potential(
    comp_xyz: _ArrayLike,
    electrode_pos: _ArrayLike,
    electrode_current: Array,
    sigma: _ScalarLike,
    min_distance_um: float = 1.0,
) -> Array:
    """Compute phi_e at compartment centres from a single point-source electrode.

    All distance/potential arithmetic is performed in JAX so that
    ``electrode_pos``, ``electrode_current``, and ``sigma`` can be JAX-traced
    for gradient-based optimisation.

    Args:
        comp_xyz: (Ncomp, 3) compartment-centre coordinates in um.  Typically a
                  static numpy array from ``get_compartment_xyz``; will be
                  promoted to a JAX constant automatically.
        electrode_pos: (3,) electrode position in um.  Pass a ``jnp.array`` to
                       differentiate w.r.t. electrode placement.
        electrode_current: (T,) electrode current samples in uA.  Pass a JAX
                           array to differentiate w.r.t. the waveform.
        sigma: Extracellular conductivity in S/m.  Typical brain tissue ~0.3 S/m.
        min_distance_um: Minimum distance floor in um to prevent division by zero
                         when a compartment centre coincides with the electrode
                         (default 1 um).

    Returns:
        phi_e: jax.Array of shape (Ncomp, T) in mV.
    """
    # Promote both spatial arrays to JAX so the whole computation is traceable.
    comp_xyz_j: Array = jnp.asarray(comp_xyz)  # (Ncomp, 3) -- static constant
    electrode_pos_j: Array = jnp.asarray(electrode_pos)  # (3,) -- may be traced

    if comp_xyz_j.ndim != 2 or comp_xyz_j.shape[1] != 3:
        raise ValueError(f"comp_xyz must be (Ncomp, 3), got {comp_xyz_j.shape}")
    if electrode_pos_j.ndim != 1 or electrode_pos_j.shape[0] != 3:
        raise ValueError(f"electrode_pos must be (3,), got {electrode_pos_j.shape}")

    # Euclidean distance from each compartment centre to the electrode [um]
    diff: Array = comp_xyz_j - electrode_pos_j[jnp.newaxis, :]  # (Ncomp, 3)
    distances: Array = jnp.sqrt((diff**2).sum(axis=-1))  # (Ncomp,)
    distances = jnp.maximum(distances, min_distance_um)

    # Spatial transfer factor [mV/uA]: phi_e [mV] = prefactor * I [uA]
    # Derivation: phi_e [mV] = I_uA * 1e3 / (4 pi sigma [S/m] * r [um])
    prefactor: Array = 1e3 / (4.0 * jnp.pi * sigma * distances)  # (Ncomp,)

    # Broadcast over time: (Ncomp, 1) * (1, T) -> (Ncomp, T)
    return prefactor[:, jnp.newaxis] * jnp.asarray(electrode_current)[jnp.newaxis, :]
