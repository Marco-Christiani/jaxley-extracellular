"""BBP (Blue Brain Project) L2/3 neuron models for Jaxley.

Provides factory functions for L2/3 pyramidal (Pyr) and PV interneuron (LBC)
cells using channel implementations from ``jaxley_mech.channels.l5pc``.

Channel parameters are extracted from the BBP biophysics.hoc files for:
- L23_PC_cADpyr229 (cell_id=6)
- L23_LBC_cNAC187 (cell_id=36)
"""

from jaxley_extracellular.bbp.cell_factory import make_pv_cell, make_pyr_cell

__all__ = [
    "make_pv_cell",
    "make_pyr_cell",
]
