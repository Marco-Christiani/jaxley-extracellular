"""Convert extracellular potential phi_e into Jaxley-compatible stimulus current.

Formulation
-----------
The cable equation with extracellular field acts on the axial driving potential
u = v + phi_e.  Jaxley's axial operator G acts on v alone, so the
extracellular field induces an extra forcing term:

    dv/dt = membrane_terms(v, ...) + G (v + phi_e)
           = membrane_terms(v, ...) + G v  +  G phi_e

The induced forcing is:

    f_ecs          = G @ phi_e          [mV/ms]   (Ncomp, T)
    i_ecs_density  = cm * f_ecs         [uA/cm^2]  (Ncomp, T)
    i_ecs_nA       = i_ecs_density * area / 1e5   [nA]       (Ncomp, T)

The last conversion is the exact inverse of Jaxley's
`convert_point_process_to_distributed` (jaxley/utils/cell_utils.py):

    nA / um^2  *  1e5  =  uA/cm^2
    (because 1 um^2 = 1e-8 cm^2,  1 nA = 1e-3 uA,  so 1 nA/um^2 = 1e5 uA/cm^2)

Public function
---------------
    phi_e_to_ecs_nA(phi_e_mV, G, cm, area_um2) -> jax.Array  (Ncomp, T) nA
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array


def phi_e_to_ecs_nA(
    phi_e_mV: Array,
    G: Array,
    cm: Array,
    area_um2: Array,
) -> Array:
    """Convert phi_e to equivalent Jaxley stimulus current in nA.

    Args:
        phi_e_mV:  (Ncomp, T) extracellular potential at compartment centres [mV].
        G:         (Ncomp, Ncomp) voltage diffusion operator [1/ms].
        cm:        (Ncomp,) membrane capacitance per compartment [uF/cm^2].
        area_um2:  (Ncomp,) membrane surface area per compartment [um^2].

    Returns:
        i_ecs_nA: (Ncomp, T) equivalent injected current [nA], ready to pass
                  into `module.data_stimulate(i_ecs_nA)`.
    """
    # f_ecs [mV/ms]: induced rate-of-change from extracellular gradient
    f_ecs: Array = G @ phi_e_mV  # (Ncomp, T)

    # i_density [uA/cm^2]: multiply by capacitance to match Jaxley's ODE units
    i_density: Array = cm[:, jnp.newaxis] * f_ecs  # (Ncomp, T)

    # i_nA [nA]: invert Jaxley's convert_point_process_to_distributed
    # i_density [uA/cm^2] = i_nA [nA] / area [um^2] * 1e5
    # => i_nA = i_density * area / 1e5
    i_ecs_nA: Array = i_density * area_um2[:, jnp.newaxis] / 1e5  # (Ncomp, T)
    return i_ecs_nA
