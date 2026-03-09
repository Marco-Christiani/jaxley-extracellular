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

import numpy as np
from jax import Array
import jax.numpy as jnp


def point_source_potential(
    comp_xyz: np.ndarray,
    electrode_pos: np.ndarray,
    electrode_current: Array,
    sigma: float,
    min_distance_um: float = 1.0,
) -> Array:
    """Compute phi_e at compartment centres from a single point-source electrode.

    Args:
        comp_xyz: (Ncomp, 3) array of compartment-centre coordinates in um.
        electrode_pos: (3,) array with electrode position in um.
        electrode_current: (T,) JAX array of electrode current samples in uA.
                           The time axis must match the simulation time vector.
        sigma: Extracellular conductivity in S/m.  Typical brain tissue: ~0.3 S/m.
        min_distance_um: Floor on the compartment-to-electrode distance in um,
                         preventing division by zero for electrodes very close
                         to or coinciding with a compartment centre (default 1 um).

    Returns:
        phi_e: jax.Array of shape (Ncomp, T) in mV.
    """
    if comp_xyz.ndim != 2 or comp_xyz.shape[1] != 3:
        raise ValueError(f"comp_xyz must be (Ncomp, 3), got {comp_xyz.shape}")
    electrode_pos = np.asarray(electrode_pos, dtype=float)
    if electrode_pos.shape != (3,):
        raise ValueError(f"electrode_pos must be (3,), got {electrode_pos.shape}")

    # Euclidean distance from each compartment centre to the electrode  [um]
    diff = comp_xyz - electrode_pos[np.newaxis, :]  # (Ncomp, 3)
    distances = np.sqrt((diff**2).sum(axis=1))  # (Ncomp,)
    distances = np.maximum(distances, min_distance_um)  # guard against zero

    # Spatial transfer factor: phi_e [mV] = prefactor [mV/uA] * I [uA]
    # prefactor = 1e3 / (4 pi sigma [S/m] * r [um])
    prefactor = 1e3 / (4.0 * np.pi * sigma * distances)  # (Ncomp,) mV/uA

    # Broadcast: (Ncomp, 1) * (1, T) -> (Ncomp, T)
    return jnp.asarray(prefactor[:, np.newaxis]) * jnp.asarray(electrode_current)[
        np.newaxis, :
    ]
