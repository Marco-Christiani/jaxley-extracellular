"""Channel import smoke-test and Ca chain wiring check."""

import jax.numpy as jnp
import pytest
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


@pytest.mark.parametrize(
    "cls",
    [NaTaT, NaTs2T, NapEt2, KPst, KTst, SKE2, SKv3_1, M, CaHVA, CaLVA, CaPump, CaNernstReversal, H],
    ids=lambda c: c.__name__,
)
def test_channel_instantiates(cls):
    assert cls().channel_params is not None


def test_eca_nernst():
    """CaNernstReversal computes physiologically reasonable eCa."""
    ch = CaNernstReversal()
    states = {"CaCon_i": jnp.array([5e-5]), "CaCon_e": jnp.array([2.0])}
    new_states = ch.update_states(states, 0.025, jnp.array([-65.0]), {})
    eca = new_states["eCa"].item()
    # Nernst: (RT/2F)*ln(2/5e-5) ~ 128 mV
    assert 100 < eca < 180, f"eCa={eca:.1f} mV outside expected range"
