"""Factory functions for BBP L2/3 pyramidal and PV interneuron cells.

Channel insertion order matters for the calcium signalling chain:
``CaNernstReversal -> CaHVA/CaLVA -> CaPump -> SKE2``. Inserting out
of order produces uninitialised state references at integration time.

For the Pyr cell, the apical ``gIhbar_Ih`` is not uniform: BBP's
biophysics specifies a path-distance-dependent gradient. We apply it
under a Euclidean-distance approximation in
:func:`_apply_pyr_apical_ih_gradient`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jaxley as jx
import numpy as np
from jaxley_mech.channels.hodgkin52 import Leak
from jaxley_mech.channels.l5pc import (
    SKE2,
    CaHVA,
    CaLVA,
    CaNernstReversal,
    CaPump,
    H,
    KPst,
    KTst,
    M,
    NapEt2,
    NaTaT,
    NaTs2T,
    SKv3_1,
)

from jaxley_extracellular.bbp.channel_params import (
    PV_ALL,
    PV_APICAL,
    PV_AXON,
    PV_BASAL,
    PV_SOMA,
    PYR_ALL,
    PYR_APICAL,
    PYR_AXON,
    PYR_BASAL,
    PYR_SOMA,
)

# Bundled morphologies converted from the vendored BBP reference simulation:
#   reference/bbp/simulation/cells/L23_PC_cADpyr229_1/morphology/
#       dend-C170897A-P3_axon-C260897C-P4_-_Clone_4.asc
#   reference/bbp/simulation/cells/L23_LBC_cNAC187_1/morphology/
#       C050398B-I4_-_Clone_3.asc
# Converted with: morph_tool.convert(sanitize=True, single_point_soma=False)
# Files named after the cell-type directory (L23_PC_cADpyr229_1 -> L23_PC_cADpyr229, etc.)
_DATA_DIR = Path(__file__).resolve().parent / "data"
PYR_SWC = _DATA_DIR / "L23_PC_cADpyr229.swc"
PV_SWC = _DATA_DIR / "L23_LBC_cNAC187.swc"


def _set_params(view: Any, params: dict[str, Any]) -> None:
    for key, val in params.items():
        view.set(key, val)


def _insert_calcium_chain(view: Any) -> None:
    """Insert the Ca signalling chain on a section view.

    Insertion order is ``CaNernstReversal -> CaHVA -> CaLVA -> CaPump
    -> SKE2``. Out-of-order insertion produces uninitialised state
    references at integration time.

    Parameters
    ----------
    view : jx.View
        A Jaxley section view (e.g.\ ``cell.soma``, ``cell.axon``).
    """
    view.insert(CaNernstReversal())
    view.insert(CaHVA())
    view.insert(CaLVA())
    view.insert(CaPump())
    view.insert(SKE2())


def _apply_pyr_apical_ih_gradient(cell: jx.Cell) -> None:
    """Apply BBP's distance-dependent apical ``gIhbar_Ih`` (cADpyr229).

    From ``biophysics.hoc``::

        distribute(apical, "gIhbar_Ih",
                   "(-0.869600 + 2.087000*exp((d-0)*0.003100))*0.000080")

    where ``d`` is NEURON's path distance from the soma in microns. We
    approximate ``d`` with the Euclidean distance from the soma centre,
    which is close for roughly-vertical apical trees. Compared to a
    uniform value, the approximation closes the resting-potential
    offset against NEURON from $-0.62$ mV to $-0.38$ mV. A path-distance
    traversal would close it further.

    Parameters
    ----------
    cell : jx.Cell
        The BBP Pyr cell. Mutated in place: per-compartment apical
        ``H_gH`` is overwritten with the gradient value.
    """
    soma_xyz = cell.soma.nodes[["x", "y", "z"]].mean().to_numpy()  # pyright: ignore[reportOptionalMemberAccess]
    apical = cell.apical.nodes  # pyright: ignore[reportOptionalMemberAccess]
    coords = apical[["x", "y", "z"]].to_numpy()
    dist_um = np.sqrt(np.sum((coords - soma_xyz) ** 2, axis=1))
    g_per_comp = (-0.8696 + 2.087 * np.exp(dist_um * 0.0031)) * 8e-5

    branches = apical["local_branch_index"].to_numpy()
    comps = apical["local_comp_index"].to_numpy()
    for br, co, g in zip(branches, comps, g_per_comp, strict=True):
        cell.apical.branch(int(br)).comp(int(co)).set("H_gH", float(g))  # pyright: ignore[reportOptionalMemberAccess]


def make_pyr_cell(
    ncomp: int = 4,
    swc_path: str | Path | None = None,
    max_branch_len: float | None = 50.0,
) -> jx.Cell:
    """Build an L2/3 pyramidal cell (cADpyr229) with BBP biophysics.

    Loads the bundled ``L23_PC_cADpyr229`` SWC morphology, inserts the
    BBP channel set per section group (soma, axon, apical, basal),
    applies BBP's biophysical parameters, and overwrites apical
    ``H_gH`` with the path-distance gradient (under a Euclidean
    approximation).

    Parameters
    ----------
    ncomp : int, optional
        Number of Jaxley compartments per SWC branch (default 4).
    swc_path : str or Path, optional
        Override the bundled morphology path. Defaults to
        ``L23_PC_cADpyr229.swc`` shipped under
        ``jaxley_extracellular.bbp.data``.
    max_branch_len : float or None, optional
        If a float (default 50 um), split long SWC branches so no
        compartment exceeds roughly ``max_branch_len / ncomp`` microns.
        The default approximates what NEURON's ``geom_nseg()`` produces
        for this cell and is needed for spike-timing parity. Pass
        ``None`` to keep the raw SWC-branch layout.

    Returns
    -------
    jx.Cell
        The fully-configured Pyr cell with channels inserted, parameters
        set, and states initialised.
    """
    cell = jx.read_swc(
        str(swc_path or PYR_SWC),
        ncomp=ncomp,
        assign_groups=True,
        max_branch_len=max_branch_len,
    )

    # Phase 1: insert channels (views go stale after insert, re-fetch for set)
    cell.insert(Leak())

    _insert_calcium_chain(cell.soma)
    cell.soma.insert(NaTs2T())
    cell.soma.insert(SKv3_1())
    cell.soma.insert(H())

    _insert_calcium_chain(cell.axon)
    cell.axon.insert(NaTaT())
    cell.axon.insert(NapEt2())
    cell.axon.insert(KPst())
    cell.axon.insert(KTst())
    cell.axon.insert(SKv3_1())

    cell.apical.insert(NaTs2T())
    cell.apical.insert(SKv3_1())
    cell.apical.insert(M())
    cell.apical.insert(H())

    cell.basal.insert(H())

    # Phase 2: set parameters on fresh views
    _set_params(cell, PYR_ALL)
    _set_params(cell.soma, PYR_SOMA)
    _set_params(cell.axon, PYR_AXON)
    _set_params(cell.apical, PYR_APICAL)
    _set_params(cell.basal, PYR_BASAL)

    # Phase 3: override apical gIhbar_Ih with BBP's distance-dependent gradient
    _apply_pyr_apical_ih_gradient(cell)

    cell.init_states()
    return cell


def make_pv_cell(
    ncomp: int = 4,
    swc_path: str | Path | None = None,
    max_branch_len: float | None = 50.0,
) -> jx.Cell:
    """Build an L2/3 PV basket cell (cNAC187) with BBP biophysics.

    Loads the bundled ``L23_LBC_cNAC187`` SWC morphology, inserts the
    BBP PV channel set per section group, and applies BBP's biophysical
    parameters.

    .. note::

       The full-morphology PV cell is unstable in Jaxley's default
       integrator (NaN at ~30 ms even unstimulated), traceable to BBP's
       ``g_pas = 1e-6`` on apical and basal. Single-compartment PV
       parity passes; full-morphology runtime claims should be made
       only on the Pyr cell.

    Parameters
    ----------
    ncomp : int, optional
        Number of Jaxley compartments per SWC branch (default 4).
    swc_path : str or Path, optional
        Override the bundled morphology path. Defaults to
        ``L23_LBC_cNAC187.swc`` shipped under
        ``jaxley_extracellular.bbp.data``.
    max_branch_len : float or None, optional
        See :func:`make_pyr_cell`.

    Returns
    -------
    jx.Cell
        The fully-configured PV cell with channels inserted, parameters
        set, and states initialised.
    """
    cell = jx.read_swc(
        str(swc_path or PV_SWC),
        ncomp=ncomp,
        assign_groups=True,
        max_branch_len=max_branch_len,
    )

    # Phase 1: insert channels
    cell.insert(Leak())

    _insert_calcium_chain(cell.soma)
    cell.soma.insert(NaTs2T())
    cell.soma.insert(SKv3_1())
    cell.soma.insert(NapEt2())
    cell.soma.insert(M())
    cell.soma.insert(KPst())
    cell.soma.insert(KTst())

    _insert_calcium_chain(cell.axon)
    cell.axon.insert(NaTaT())
    cell.axon.insert(NapEt2())
    cell.axon.insert(M())
    cell.axon.insert(KPst())
    cell.axon.insert(KTst())
    cell.axon.insert(SKv3_1())

    cell.apical.insert(NaTs2T())
    cell.apical.insert(SKv3_1())
    cell.apical.insert(NapEt2())
    cell.apical.insert(M())
    cell.apical.insert(KPst())
    cell.apical.insert(KTst())
    cell.apical.insert(H())

    cell.basal.insert(NaTs2T())
    cell.basal.insert(SKv3_1())
    cell.basal.insert(NapEt2())
    cell.basal.insert(M())
    cell.basal.insert(KPst())
    cell.basal.insert(KTst())
    cell.basal.insert(H())

    # Phase 2: set parameters on fresh views
    _set_params(cell, PV_ALL)
    _set_params(cell.soma, PV_SOMA)
    _set_params(cell.axon, PV_AXON)
    _set_params(cell.apical, PV_APICAL)
    _set_params(cell.basal, PV_BASAL)

    cell.init_states()
    return cell
