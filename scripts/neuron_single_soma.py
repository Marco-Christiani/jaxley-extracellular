"""Generate NEURON reference traces for the single-soma parity test.

Builds a 1-compartment cylindrical soma (no morphology, no BBP scaling),
inserts the Pyr or PV channel set, and records v(t) under two step currents.

Run from the dedicated NEURON shell:

    nix develop .#neuron
    cd reference/bbp/simulation && nrnivmodl mechanisms
    cd /path/to/jaxley-extracellular
    neuron-python -m scripts.neuron_single_soma

Writes results/neuron_single_soma_{pyr,pv}.npz with keys t, v_sub, v_sup.
"""

import os
import time
from pathlib import Path

import numpy as np

from scripts.neuron_bbp_common import bbp_sim_dir

NEURON_DIR = bbp_sim_dir()
OUT_DIR = Path(__file__).resolve().parent.parent / "results"

# Geometry (matched exactly between NEURON and Jaxley)
L = 20.0          # um
DIAM = 20.0       # um
RA = 100.0        # Ohm*cm
CM = 1.0          # uF/cm^2

# Simulation
DT = 0.025
V_INIT = -75.0
T_MAX = 700.0
CELSIUS = 34.0    # matches jaxley_mech l5pc Q10 target

# Stimuli
I_DELAY = 100.0
I_DUR = 500.0
I_SUB = 0.05      # nA subthreshold
I_SUP = 0.5       # nA suprathreshold

# Reversals from BBP biophysics.hoc (Ih.ehcn is a mod-internal RANGE, = -45 mV)
ENA = 50.0
EK = -85.0

# NEURON SUFFIX -> list of (RANGE_name, value).  Extracted from biophysics.hoc.
PYR_SOMA = {
    "pas":           [("g_pas", 3e-5), ("e_pas", -75.0)],
    "Ih":            [("gIhbar_Ih", 0.000080)],
    "NaTs2_t":       [("gNaTs2_tbar_NaTs2_t", 0.926705)],
    "SKv3_1":        [("gSKv3_1bar_SKv3_1", 0.102517)],
    "SK_E2":         [("gSK_E2bar_SK_E2", 0.099433)],
    "Ca_HVA":        [("gCa_HVAbar_Ca_HVA", 0.000374)],
    "Ca_LVAst":      [("gCa_LVAstbar_Ca_LVAst", 0.000778)],
    "CaDynamics_E2": [("gamma_CaDynamics_E2", 0.000533),
                      ("decay_CaDynamics_E2", 342.544232)],
}

PV_SOMA = {
    "pas":           [("g_pas", 0.000091), ("e_pas", -62.442793)],
    "NaTs2_t":       [("gNaTs2_tbar_NaTs2_t", 0.197999)],
    "SKv3_1":        [("gSKv3_1bar_SKv3_1", 0.297559)],
    "Nap_Et2":       [("gNap_Et2bar_Nap_Et2", 0.000001)],
    "Im":            [("gImbar_Im", 0.000008)],
    "K_Pst":         [("gK_Pstbar_K_Pst", 0.156376)],
    "K_Tst":         [("gK_Tstbar_K_Tst", 0.092965)],
    "SK_E2":         [("gSK_E2bar_SK_E2", 0.019726)],
    "Ca_HVA":        [("gCa_HVAbar_Ca_HVA", 0.000032)],
    "Ca_LVAst":      [("gCa_LVAstbar_Ca_LVAst", 0.001067)],
    "CaDynamics_E2": [("gamma_CaDynamics_E2", 0.000511),
                      ("decay_CaDynamics_E2", 731.707637)],
}


def run_protocol(mechs: dict, i_amp: float, section_name: str) -> tuple[np.ndarray, np.ndarray, float]:
    """Create a fresh section, run a step-current protocol, return (t, v, run_seconds)."""
    from neuron import h
    from neuron.units import ms, mV

    h.load_file("stdrun.hoc")

    sec = h.Section(name=section_name)
    sec.L = L
    sec.diam = DIAM
    sec.Ra = RA
    sec.cm = CM
    sec.nseg = 1

    for mech_name, params in mechs.items():
        sec.insert(mech_name)
        for pname, pval in params:
            setattr(sec, pname, pval)

    # Ion reversals set once after insertion
    has_na = any(m in mechs for m in ("NaTs2_t", "Nap_Et2", "NaTa_t"))
    has_k = any(m in mechs for m in ("SKv3_1", "SK_E2", "K_Pst", "K_Tst", "Im"))
    if has_na:
        sec.ena = ENA
    if has_k:
        sec.ek = EK

    iclamp = h.IClamp(sec(0.5))
    iclamp.delay = I_DELAY
    iclamp.dur = I_DUR
    iclamp.amp = i_amp

    v_vec = h.Vector().record(sec(0.5)._ref_v)
    t_vec = h.Vector().record(h._ref_t)

    h.celsius = CELSIUS
    h.dt = DT
    h.finitialize(V_INIT * mV)
    t0 = time.perf_counter()
    h.continuerun(T_MAX * ms)
    run_s = time.perf_counter() - t0

    return np.array(t_vec), np.array(v_vec), run_s


def run(label: str, mechs: dict) -> None:
    t, v_sub, t_sub = run_protocol(mechs, I_SUB, f"soma_{label}_sub")
    _, v_sup, t_sup = run_protocol(mechs, I_SUP, f"soma_{label}_sup")

    OUT_DIR.mkdir(exist_ok=True)
    out = OUT_DIR / f"neuron_single_soma_{label}.npz"
    np.savez(
        out, t=t, v_sub=v_sub, v_sup=v_sup,
        run_s_sub=t_sub, run_s_sup=t_sup,
        L=L, diam=DIAM, Ra=RA, cm=CM, v_init=V_INIT, dt=DT, celsius=CELSIUS,
        i_sub=I_SUB, i_sup=I_SUP, i_delay=I_DELAY, i_dur=I_DUR,
    )
    n_spk = int((np.diff((v_sup > -25).astype(int)) > 0).sum())
    print(f"Saved {out}  (suprathresh spikes: {n_spk}, sub={t_sub*1e3:.1f} ms, sup={t_sup*1e3:.1f} ms)")


if __name__ == "__main__":
    os.chdir(str(NEURON_DIR))
    run("pyr", PYR_SOMA)
    run("pv", PV_SOMA)
