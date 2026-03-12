"""Tests for waveform generation and response feature extraction."""

from functools import partial

import jax
import jax.numpy as jnp
import pytest

from jaxley_extracellular.extracellular.response import (
    detect_spike,
    extract_response_features,
    extract_response_features_batch,
    spike_latency_ms,
    spike_latency_steps,
)
from jaxley_extracellular.extracellular.waveforms import (
    make_biphasic_grid,
    make_biphasic_pulse,
    make_monophasic_pulse,
)

# ===================================================================
# Waveform tests
# ===================================================================


class TestMonophasicPulse:
    def test_shape(self) -> None:
        w = make_monophasic_pulse(100.0, 0.5, 0.025, 5.0)
        assert w.shape == (200,)

    @pytest.mark.parametrize(
        ("cathodic", "expected_sign"),
        [(True, -1.0), (False, 1.0)],
        ids=["cathodic", "anodic"],
    )
    def test_polarity(self, cathodic: bool, expected_sign: float) -> None:
        w = make_monophasic_pulse(100.0, 0.5, 0.025, 5.0, cathodic=cathodic)
        extreme = float(w.min()) if cathodic else float(w.max())
        assert extreme == pytest.approx(expected_sign * 100.0)
        zero_side = float(w.max()) if cathodic else float(w.min())
        assert zero_side == 0.0

    def test_pulse_width(self) -> None:
        w = make_monophasic_pulse(1.0, 1.0, 0.025, 5.0)
        n_active = int((w != 0).sum())
        assert n_active == int(1.0 / 0.025)  # 40 samples

    def test_delay(self) -> None:
        w = make_monophasic_pulse(1.0, 0.5, 0.025, 5.0, delay_ms=1.0)
        onset = int(1.0 / 0.025)  # 40
        assert float(w[onset - 1]) == 0.0
        assert float(w[onset]) != 0.0

    def test_zero_outside_pulse(self) -> None:
        w = make_monophasic_pulse(100.0, 0.5, 0.025, 5.0)
        pw_samples = int(0.5 / 0.025)
        assert float(jnp.abs(w[pw_samples:]).sum()) == 0.0


class TestBiphasicPulse:
    def test_charge_balanced(self) -> None:
        w = make_biphasic_pulse(100.0, 0.5, 0.025, 5.0)
        assert float(jnp.abs(w.sum())) == pytest.approx(0.0, abs=1e-4)

    @pytest.mark.parametrize(
        ("cathodic_first", "first_positive"),
        [(True, False), (False, True)],
        ids=["cathodic_first", "anodic_first"],
    )
    def test_leading_phase(self, cathodic_first: bool, first_positive: bool) -> None:
        w = make_biphasic_pulse(100.0, 0.5, 0.025, 5.0, cathodic_first=cathodic_first)
        first_nonzero = w[w != 0][0]
        if first_positive:
            assert float(first_nonzero) > 0
        else:
            assert float(first_nonzero) < 0

    def test_interphase_gap(self) -> None:
        w = make_biphasic_pulse(100.0, 0.5, 0.025, 5.0, interphase_ms=0.5)
        pw = int(0.5 / 0.025)  # 20
        ip = int(0.5 / 0.025)  # 20
        # Gap should be zero
        gap = w[pw : pw + ip]
        assert float(jnp.abs(gap).sum()) == 0.0
        # Second phase should start after gap
        assert float(w[pw + ip]) != 0.0

    def test_shape(self) -> None:
        w = make_biphasic_pulse(100.0, 0.5, 0.025, 5.0)
        assert w.shape == (200,)


class TestBiphasicGrid:
    def test_grid_shape(self) -> None:
        amps = jnp.array([50.0, 100.0, 200.0])
        pws = jnp.array([0.25, 0.5, 1.0])
        waveforms, grid_amps, grid_pws = make_biphasic_grid(
            amps,
            pws,
            0.025,
            5.0,
        )
        assert waveforms.shape == (9, 200)  # 3 x 3 grid
        assert grid_amps.shape == (9,)
        assert grid_pws.shape == (9,)

    def test_all_charge_balanced(self) -> None:
        amps = jnp.array([50.0, 100.0])
        pws = jnp.array([0.25, 0.5])
        waveforms, _, _ = make_biphasic_grid(amps, pws, 0.025, 5.0)
        charges = jnp.abs(waveforms.sum(axis=1))
        assert float(charges.max()) == pytest.approx(0.0, abs=1e-4)


# ===================================================================
# Response feature tests
# ===================================================================


class TestDetectSpike:
    def test_spike_detected(self) -> None:
        v = jnp.array([-65.0, -60.0, -40.0, 10.0, -30.0])
        assert bool(detect_spike(v, threshold_mV=0.0))

    def test_subthreshold(self) -> None:
        v = jnp.array([-65.0, -60.0, -55.0, -60.0])
        assert not bool(detect_spike(v, threshold_mV=0.0))

    def test_custom_threshold(self) -> None:
        v = jnp.array([-65.0, -50.0, -45.0])
        assert bool(detect_spike(v, threshold_mV=-50.0))


class TestSpikeLatency:
    def test_latency_correct(self) -> None:
        v = jnp.array([-65.0, -60.0, -40.0, 10.0, -30.0])
        idx = spike_latency_steps(v, threshold_mV=0.0)
        assert int(idx) == 3  # first crossing at index 3

    def test_no_spike_sentinel(self) -> None:
        v = jnp.array([-65.0, -60.0, -55.0])
        idx = spike_latency_steps(v, threshold_mV=0.0)
        assert int(idx) == len(v)  # sentinel = T

    def test_latency_ms(self) -> None:
        v = jnp.array([-65.0, -60.0, -40.0, 10.0, -30.0])
        lat = spike_latency_ms(v, dt_ms=0.025, threshold_mV=0.0)
        assert float(lat) == pytest.approx(3 * 0.025)

    def test_vmap_compatible(self) -> None:
        """Latency can be vmapped over a batch of traces."""
        v_batch = jnp.array(
            [
                [-65.0, -60.0, 10.0, -30.0],  # spikes at idx 2
                [-65.0, -60.0, -55.0, -60.0],  # no spike
            ]
        )
        latency_fn = partial(spike_latency_steps, threshold_mV=0.0)
        latencies = jax.vmap(latency_fn)(v_batch)
        assert int(latencies[0]) == 2
        assert int(latencies[1]) == 4  # sentinel = T


class TestExtractResponseFeatures:
    def test_spiking_trace(self) -> None:
        v = jnp.array([-65.0, -60.0, 10.0, -30.0, -65.0])
        feats = extract_response_features(v, dt_ms=0.025)
        assert bool(feats["spiked"])
        assert float(feats["latency_ms"]) == pytest.approx(2 * 0.025)
        assert float(feats["vmax"]) == pytest.approx(10.0)
        assert float(feats["vmin"]) == pytest.approx(-65.0)

    def test_subthreshold_trace(self) -> None:
        v = jnp.array([-65.0, -60.0, -55.0, -60.0, -65.0])
        feats = extract_response_features(v, dt_ms=0.025)
        assert not bool(feats["spiked"])
        assert float(feats["latency_ms"]) == pytest.approx(5 * 0.025)


class TestBatchFeatures:
    def test_batch_shape(self) -> None:
        v_batch = jnp.array(
            [
                [-65.0, -60.0, 10.0, -30.0],
                [-65.0, -60.0, -55.0, -60.0],
                [-65.0, -50.0, 30.0, 5.0],
            ]
        )
        feats = extract_response_features_batch(v_batch, dt_ms=0.025)
        assert feats["spiked"].shape == (3,)
        assert feats["latency_ms"].shape == (3,)
        assert feats["vmax"].shape == (3,)

    def test_batch_values(self) -> None:
        v_batch = jnp.array(
            [
                [-65.0, -60.0, 10.0, -30.0],
                [-65.0, -60.0, -55.0, -60.0],
            ]
        )
        feats = extract_response_features_batch(v_batch, dt_ms=0.025)
        assert bool(feats["spiked"][0])
        assert not bool(feats["spiked"][1])
