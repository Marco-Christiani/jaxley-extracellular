"""Shared NEURON helpers for BBP-backed scripts."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

DEFAULT_BBP_SIM_DIR = Path(__file__).resolve().parent.parent / "reference" / "bbp" / "simulation"


def bbp_sim_dir() -> Path:
    """Return the vendored BBP simulation root."""
    return Path(os.environ.get("BBP_SIM_DIR", str(DEFAULT_BBP_SIM_DIR))).expanduser().resolve()


def bbp_mech_so(sim_dir: Path | None = None) -> Path:
    """Return the compiled NEURON mechanism library path."""
    if "BBP_MECH_SO" in os.environ:
        return Path(os.environ["BBP_MECH_SO"]).expanduser().resolve()
    base = bbp_sim_dir() if sim_dir is None else sim_dir
    return (base / "x86_64" / "libnrnmech.so").resolve()


def bbp_cell_dir(cell_name: str, sim_dir: Path | None = None) -> Path:
    """Return the directory for a BBP cell template under the vendored tree."""
    base = bbp_sim_dir() if sim_dir is None else sim_dir
    return (base / "cells" / cell_name).resolve()


BBP_SIM_DIR = bbp_sim_dir()
PYR_CELL_DIR = bbp_cell_dir("L23_PC_cADpyr229_1", BBP_SIM_DIR)
BBP_MECH_SO = bbp_mech_so(BBP_SIM_DIR)


def load_bbp_pyr_cell(h: Any) -> Any:
    """Load the BBP Pyr template and instantiate one cell without synapses."""
    h.nrn_load_dll(str(BBP_MECH_SO))
    os.chdir(str(PYR_CELL_DIR))
    h.load_file("stdrun.hoc")
    h.load_file("import3d.hoc")
    h.load_file("morphology.hoc")
    h.load_file("biophysics.hoc")
    h.load_file("template.hoc")
    return h.cADpyr229_L23_PC_5ecbf9b163(0)


def pyr_cell_dir() -> Path:
    """Expose the resolved BBP Pyr cell directory for logging."""
    return PYR_CELL_DIR
