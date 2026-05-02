"""Conductance and passive parameters from BBP biophysics.hoc files.

Values extracted verbatim from the ``distribute()`` calls in:
- ``L23_PC_cADpyr229_1/biophysics.hoc``
- ``L23_LBC_cNAC187_1/biophysics.hoc``

Naming convention: ``{Channel}_{param}`` matches jaxley_mech parameter names.
All conductances in S/cm^2, gamma dimensionless, decay in ms.

Note on PV "Ca" channel: The NEURON ``Ca`` mechanism is identical in kinetics
to ``Ca_HVA`` (same Reuveni et al. 1993 reference, same equations), just with
a different SUFFIX. We use ``CaHVA`` from jaxley_mech for both.
"""

from typing import Any

# L23 Pyramidal cell (cADpyr229)

PYR_ALL: dict[str, Any] = {
    "Leak_gLeak": 3e-5,
    "Leak_eLeak": -75.0,
    "axial_resistivity": 100.0,
}

PYR_SOMA: dict[str, Any] = {
    "NaTs2T_gNaTs2T": 0.926705,
    "SKv3_1_gSKv3_1": 0.102517,
    "SKE2_gSKE2": 0.099433,
    "CaHVA_gCaHVA": 0.000374,
    "CaLVA_gCaLVA": 0.000778,
    "H_gH": 0.000080,
    "CaPump_gamma": 0.000533,
    "CaPump_decay": 342.544232,
    "eNa": 50.0,
    "eK": -85.0,
    "capacitance": 1.0,
}

PYR_AXON: dict[str, Any] = {
    "NaTaT_gNaTaT": 3.429725,
    "NapEt2_gNapEt2": 0.009803,
    "KPst_gKPst": 0.959296,
    "KTst_gKTst": 0.001035,
    "SKv3_1_gSKv3_1": 0.094971,
    "SKE2_gSKE2": 0.008085,
    "CaHVA_gCaHVA": 0.000306,
    "CaLVA_gCaLVA": 0.000050,
    "CaPump_gamma": 0.016713,
    "CaPump_decay": 384.114655,
    "eNa": 50.0,
    "eK": -85.0,
    "capacitance": 1.0,
}

PYR_APICAL: dict[str, Any] = {
    "NaTs2T_gNaTs2T": 0.012009,
    "SKv3_1_gSKv3_1": 0.000513,
    "M_gM": 0.000740,
    # Ih: uniform baseline, overwritten per-compartment by the
    # distance-dependent gradient in cell_factory._apply_pyr_apical_ih_gradient.
    "H_gH": 0.000080,
    "eNa": 50.0,
    "eK": -85.0,
    "capacitance": 2.0,
}

PYR_BASAL: dict[str, Any] = {
    # Ih: uniform value 0.000080
    "H_gH": 0.000080,
    "capacitance": 2.0,
}


# L23 PV interneuron / LBC (cNAC187)

PV_ALL: dict[str, Any] = {
    "axial_resistivity": 100.0,
    "capacitance": 1.0,
}

PV_SOMA: dict[str, Any] = {
    "NaTs2T_gNaTs2T": 0.197999,
    "SKv3_1_gSKv3_1": 0.297559,
    "NapEt2_gNapEt2": 0.000001,
    "M_gM": 0.000008,
    "KPst_gKPst": 0.156376,
    "KTst_gKTst": 0.092965,
    "SKE2_gSKE2": 0.019726,
    "CaHVA_gCaHVA": 0.000032,
    "CaLVA_gCaLVA": 0.001067,
    "CaPump_gamma": 0.000511,
    "CaPump_decay": 731.707637,
    "Leak_gLeak": 0.000091,
    "Leak_eLeak": -62.442793,
    "eNa": 50.0,
    "eK": -85.0,
}

PV_AXON: dict[str, Any] = {
    "NaTaT_gNaTaT": 3.959764,
    "NapEt2_gNapEt2": 0.000000,
    "M_gM": 0.000999,
    "KPst_gKPst": 0.004729,
    "KTst_gKTst": 0.098908,
    "SKv3_1_gSKv3_1": 0.317363,
    "SKE2_gSKE2": 0.003442,
    "CaHVA_gCaHVA": 0.000003,
    "CaLVA_gCaLVA": 0.000015,
    "CaPump_gamma": 0.010353,
    "CaPump_decay": 64.277990,
    "Leak_gLeak": 0.000094,
    "Leak_eLeak": -60.216510,
    "eNa": 50.0,
    "eK": -85.0,
}

PV_APICAL: dict[str, Any] = {
    "NaTs2T_gNaTs2T": 0.000010,
    "SKv3_1_gSKv3_1": 0.004399,
    "NapEt2_gNapEt2": 0.000001,
    "M_gM": 0.000008,
    # K_Pst inserted but no distribute call -> NEURON PARAMETER default
    "KPst_gKPst": 0.00001,
    "KTst_gKTst": 0.009500,
    # Ih: distance-dependent, uniform base value 0.000052
    "H_gH": 0.000052,
    "Leak_gLeak": 0.000001,
    "Leak_eLeak": -79.315740,
    "eNa": 50.0,
    "eK": -85.0,
}

PV_BASAL: dict[str, Any] = {
    "NaTs2T_gNaTs2T": 0.000010,
    "SKv3_1_gSKv3_1": 0.004399,
    "NapEt2_gNapEt2": 0.000001,
    "M_gM": 0.000008,
    # K_Pst inserted but no distribute call -> NEURON PARAMETER default
    "KPst_gKPst": 0.00001,
    "KTst_gKTst": 0.009500,
    # Ih: distance-dependent, uniform base value 0.000052
    "H_gH": 0.000052,
    "Leak_gLeak": 0.000001,
    "Leak_eLeak": -79.315740,
    "eNa": 50.0,
    "eK": -85.0,
}
