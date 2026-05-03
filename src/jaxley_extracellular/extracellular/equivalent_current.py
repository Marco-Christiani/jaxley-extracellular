r"""Convert extracellular potential into Jaxley-compatible stimulus current.

The activating function is the voltage-rate term

.. math::

    \mathbf{f} = G\boldsymbol{\Phi},

with units mV/ms. Jaxley's public stimulation API accepts current in nA,
so the package encodes :math:`\mathbf{f}` as

.. math::

    \mathbf{I}_{\mathrm{ecs}}
    =
    \left(\mathbf{c} \odot \frac{\mathbf{A}}{10^5}\right)
    \odot (G\boldsymbol{\Phi}).

The capacitance and area factors cancel inside Jaxley's current-density
conversion; they are an API encoding detail, not an added biophysical term.
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
    r"""Convert :math:`\boldsymbol{\Phi}` to equivalent current in nA.

    For each compartment :math:`j` and timestep :math:`t`, this computes

    .. math::

        I_{\mathrm{ecs},j}(t)
        =
        c_j [G\boldsymbol{\phi}(t)]_j \frac{A_j}{10^5}.

    The result is shaped like ``phi_e_mV`` and can be passed through
    Jaxley's standard ``data_stimuli`` path.

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
