"""Convert extracellular potential phi_e into Jaxley-compatible stimulus current.

f_ecs [mV/ms]  = G @ phi_e
i_ecs [nA]     = cm * f_ecs * area / 1e5
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

    Parameters
    ----------
    phi_e_mV : Array, shape ``(Ncomp, T)``
        Extracellular potential at compartment centres, in mV.
    G : Array, shape ``(Ncomp, Ncomp)``
        Voltage diffusion operator, in 1/ms. Expected to be a sparse
        ``BCOO`` matrix; dense G triggers a ``(B, Ncomp, Ncomp)`` XLA
        broadcast under ``vmap`` that exhausts HBM at large Ncomp.
    cm : Array, shape ``(Ncomp,)``
        Specific membrane capacitance per compartment, in uF/cm^2.
    area_um2 : Array, shape ``(Ncomp,)``
        Membrane surface area per compartment, in um^2.

    Returns
    -------
    Array, shape ``(Ncomp, T)``
        Equivalent injected current in nA, ready to pass into
        ``module.data_stimulate(i_ecs_nA)``.
    """
    # G is expected to be a sparse BCOO matrix. sparse @ dense avoids the
    # (B, Ncomp, Ncomp) broadcast that dense G triggers on TPU under vmap.
    f_ecs: Array = G @ phi_e_mV

    # i_density [uA/cm^2]: multiply by capacitance to match Jaxley's ODE units
    i_density: Array = cm[:, jnp.newaxis] * f_ecs  # (Ncomp, T)

    # i_nA [nA]: invert Jaxley's convert_point_process_to_distributed
    # i_density [uA/cm^2] = i_nA [nA] / area [um^2] * 1e5
    # => i_nA = i_density * area / 1e5
    i_ecs_nA: Array = i_density * area_um2[:, jnp.newaxis] / 1e5  # (Ncomp, T)
    return i_ecs_nA
