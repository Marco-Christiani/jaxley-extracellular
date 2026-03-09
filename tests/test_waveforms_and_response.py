"""Tests for waveform generation and response feature extraction."""

import jax
import jax.numpy as jnp
import pytest

from jaxley_extracellular.extracellular.waveforms import (
    make_monophasic_pulse,
    make_biphasic_pulse,
    make_biphasic_grid,
)
from jaxley_extracellular.extracellular.response import (
    detect_spike,
    spike_latency_steps,
    spike_latency_ms,
    extract_response_features,
    extract_response_features_batch,
)


# ===================================================================
# Waveform tests
# ===================================================================

class TestMonophasicPulse:
    def test_shape(self):
        w = make_monophasic_pulse(100.0, 0.5, 0.025, 5.0)
        assert w.shape == (200,)

    def test_cathodic_negative(self):
        w = make_monophasic_pulse(100.0, 0.5, 0.025, 5.0, cathodic=True)
        assert float(w.min()) == pytest.approx(-100.0)
        assert float(w.max()) == 0.0

    def test_anodic_positive(self):
        w = make_monophasic_pulse(100.0, 0.5, 0.025, 5.0, cathodic=False)
        assert float(w.max()) == pytest.approx(100.0)
        assert float(w.min()) == 0.0

    def test_pulse_width(self):
        w = make_monophasic_pulse(1.0, 1.0, 0.025, 5.0)
        n_active = int((w != 0).sum())
        assert n_active == int(1.0 / 0.025)  # 40 samples

    def test_delay(self):
        w = make_monophasic_pulse(1.0, 0.5, 0.025, 5.0, delay_ms=1.0)
        onset = int(1.0 / 0.025)  # 40
        assert float(w[onset - 1]) == 0.0
        assert float(w[onset]) != 0.0

    def test_zero_outside_pulse(self):
        w = make_monophasic_pulse(100.0, 0.5, 0.025, 5.0)
        pw_samples = int(0.5 / 0.025)
        assert float(jnp.abs(w[pw_samples:]).sum()) == 0.0


class TestBiphasicPulse:
    def test_charge_balanced(self):
        w = make_biphasic_pulse(100.0, 0.5, 0.025, 5.0)
        assert float(jnp.abs(w.sum())) == pytest.approx(0.0, abs=1e-4)

    def test_cathodic_first(self):
        w = make_biphasic_pulse(100.0, 0.5, 0.025, 5.0, cathodic_first=True)
        # First nonzero sample should be negative
        first_nonzero = w[w != 0][0]
        assert float(first_nonzero) < 0

    def test_anodic_first(self):
        w = make_biphasic_pulse(100.0, 0.5, 0.025, 5.0, cathodic_first=False)
        first_nonzero = w[w != 0][0]
        assert float(first_nonzero) > 0

    def test_interphase_gap(self):
        w = make_biphasic_pulse(100.0, 0.5, 0.025, 5.0, interphase_ms=0.5)
        pw = int(0.5 / 0.025)  # 20
        ip = int(0.5 / 0.025)  # 20
        # Gap should be zero
        gap = w[pw : pw + ip]
        assert float(jnp.abs(gap).sum()) == 0.0
        # Second phase should start after gap
        assert float(w[pw + ip]) != 0.0

    def test_shape(self):
        w = make_biphasic_pulse(100.0, 0.5, 0.025, 5.0)
        assert w.shape == (200,)


class TestBiphasicGrid:
    def test_grid_shape(self):
        amps = jnp.array([50.0, 100.0, 200.0])
        pws = jnp.array([0.25, 0.5, 1.0])
        waveforms, grid_amps, grid_pws = make_biphasic_grid(
            amps, pws, 0.025, 5.0,
        )
        assert waveforms.shape == (9, 200)  # 3 x 3 grid
        assert grid_amps.shape == (9,)
        assert grid_pws.shape == (9,)

    def test_all_charge_balanced(self):
        amps = jnp.array([50.0, 100.0])
        pws = jnp.array([0.25, 0.5])
        waveforms, _, _ = make_biphasic_grid(amps, pws, 0.025, 5.0)
        charges = jnp.abs(waveforms.sum(axis=1))
        assert float(charges.max()) == pytest.approx(0.0, abs=1e-4)


# ===================================================================
# Response feature tests
# ===================================================================

class TestDetectSpike:
    def test_spike_detected(self):
        v = jnp.array([-65.0, -60.0, -40.0, 10.0, -30.0])
        assert bool(detect_spike(v, threshold_mV=0.0))

    def test_subthreshold(self):
        v = jnp.array([-65.0, -60.0, -55.0, -60.0])
        assert not bool(detect_spike(v, threshold_mV=0.0))

    def test_custom_threshold(self):
        v = jnp.array([-65.0, -50.0, -45.0])
        assert bool(detect_spike(v, threshold_mV=-50.0))


class TestSpikeLatency:
    def test_latency_correct(self):
        v = jnp.array([-65.0, -60.0, -40.0, 10.0, -30.0])
        idx = spike_latency_steps(v, threshold_mV=0.0)
        assert int(idx) == 3  # first crossing at index 3

    def test_no_spike_sentinel(self):
        v = jnp.array([-65.0, -60.0, -55.0])
        idx = spike_latency_steps(v, threshold_mV=0.0)
        assert int(idx) == len(v)  # sentinel = T

    def test_latency_ms(self):
        v = jnp.array([-65.0, -60.0, -40.0, 10.0, -30.0])
        lat = spike_latency_ms(v, dt_ms=0.025, threshold_mV=0.0)
        assert float(lat) == pytest.approx(3 * 0.025)

    def test_vmap_compatible(self):
        """Latency can be vmapped over a batch of traces."""
        v_batch = jnp.array([
            [-65.0, -60.0, 10.0, -30.0],   # spikes at idx 2
            [-65.0, -60.0, -55.0, -60.0],  # no spike
        ])
        latencies = jax.vmap(
            lambda v: spike_latency_steps(v, threshold_mV=0.0)
        )(v_batch)
        assert int(latencies[0]) == 2
        assert int(latencies[1]) == 4  # sentinel = T


class TestExtractResponseFeatures:
    def test_spiking_trace(self):
        v = jnp.array([-65.0, -60.0, 10.0, -30.0, -65.0])
        feats = extract_response_features(v, dt_ms=0.025)
        assert bool(feats["spiked"])
        assert float(feats["latency_ms"]) == pytest.approx(2 * 0.025)
        assert float(feats["vmax"]) == pytest.approx(10.0)
        assert float(feats["vmin"]) == pytest.approx(-65.0)

    def test_subthreshold_trace(self):
        v = jnp.array([-65.0, -60.0, -55.0, -60.0, -65.0])
        feats = extract_response_features(v, dt_ms=0.025)
        assert not bool(feats["spiked"])
        assert float(feats["latency_ms"]) == pytest.approx(5 * 0.025)


class TestBatchFeatures:
    def test_batch_shape(self):
        v_batch = jnp.array([
            [-65.0, -60.0, 10.0, -30.0],
            [-65.0, -60.0, -55.0, -60.0],
            [-65.0, -50.0, 30.0, 5.0],
        ])
        feats = extract_response_features_batch(v_batch, dt_ms=0.025)
        assert feats["spiked"].shape == (3,)
        assert feats["latency_ms"].shape == (3,)
        assert feats["vmax"].shape == (3,)

    def test_batch_values(self):
        v_batch = jnp.array([
            [-65.0, -60.0, 10.0, -30.0],
            [-65.0, -60.0, -55.0, -60.0],
        ])
        feats = extract_response_features_batch(v_batch, dt_ms=0.025)
        assert bool(feats["spiked"][0])
        assert not bool(feats["spiked"][1])
