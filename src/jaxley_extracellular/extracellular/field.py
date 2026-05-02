"""Point-source electrode field model (multi-electrode).

    phi_e [mV] = sum_i  I_i [uA] * 1e3 / (4 pi sigma [S/m] * r_i [um])

Superposition of point sources in a homogeneous, isotropic medium.
Units: positions in um, current in uA, sigma in S/m, phi_e in mV.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from jax import Array

# Electrode arrays may be plain numpy (static) or JAX arrays (traced,
# e.g. when differentiating w.r.t. electrode placement).
_ArrayLike = np.ndarray | Array
_ScalarLike = float | Array


def point_source_potential(
    comp_xyz: _ArrayLike,
    electrode_positions: _ArrayLike,
    electrode_currents: Array,
    sigma: _ScalarLike,
    min_distance_um: float = 1.0,
) -> Array:
    """Compute phi_e at compartment centres from point-source electrodes.

    Multiple electrodes are handled via superposition: the total potential
    is the sum of independent point-source contributions.

    All arithmetic is performed in JAX, so ``electrode_positions``,
    ``electrode_currents``, and ``sigma`` can be JAX-traced for
    gradient-based optimisation.

    Parameters
    ----------
    comp_xyz : array_like, shape ``(Ncomp, 3)``
        Compartment-centre coordinates in um.
    electrode_positions : array_like, shape ``(N_elec, 3)``
        Electrode positions in um.
    electrode_currents : Array, shape ``(N_elec, T)``
        Electrode current waveforms in uA.
    sigma : float or Array
        Extracellular conductivity in S/m. Typical brain tissue is
        ~0.3 S/m.
    min_distance_um : float, optional
        Minimum distance floor in um to prevent division by zero when a
        compartment centre coincides with an electrode (default 1 um).

    Returns
    -------
    Array, shape ``(Ncomp, T)``
        Extracellular potential at each compartment centre, in mV,
        summed over all electrodes.

    Raises
    ------
    ValueError
        If ``comp_xyz``, ``electrode_positions``, or
        ``electrode_currents`` have the wrong shape.
    """
    comp_xyz_j: Array = jnp.asarray(comp_xyz)  # (Ncomp, 3)
    positions_j: Array = jnp.asarray(electrode_positions)  # (N_elec, 3)
    currents_j: Array = jnp.asarray(electrode_currents)  # (N_elec, T)

    if comp_xyz_j.ndim != 2 or comp_xyz_j.shape[1] != 3:
        raise ValueError(f"comp_xyz must be (Ncomp, 3), got {comp_xyz_j.shape}")
    if positions_j.ndim != 2 or positions_j.shape[1] != 3:
        raise ValueError(
            f"electrode_positions must be (N_elec, 3), got {positions_j.shape}"
        )
    if currents_j.ndim != 2:
        raise ValueError(
            f"electrode_currents must be (N_elec, T), got {currents_j.shape}"
        )

    # Euclidean distance from each compartment centre to each electrode [um]
    # broadcast: (N_elec, 1, 3) - (1, Ncomp, 3) -> (N_elec, Ncomp, 3)
    diff: Array = positions_j[:, jnp.newaxis, :] - comp_xyz_j[jnp.newaxis, :, :]
    distances: Array = jnp.sqrt((diff**2).sum(axis=-1))  # (N_elec, Ncomp) [um]
    distances = jnp.maximum(distances, min_distance_um)

    # Spatial transfer factor [mV/uA]:  phi_e [mV] = prefactor * I [uA]
    # Derivation: phi_e [V] = I [A] / (4 pi sigma [S/m] * r [m])
    #             phi_e [mV] = I [uA] * 1e3 / (4 pi sigma [S/m] * r [um])
    prefactor: Array = 1e3 / (4.0 * jnp.pi * sigma * distances)  # (N_elec, Ncomp)

    # Per-electrode phi_e, then superposition (linear sum):
    # broadcast: (N_elec, Ncomp, 1) * (N_elec, 1, T) -> (N_elec, Ncomp, T)
    # sum over electrodes (axis=0) -> (Ncomp, T) [mV]
    phi_e: Array = (
        prefactor[:, :, jnp.newaxis] * currents_j[:, jnp.newaxis, :]
    ).sum(axis=0)
    return phi_e
