"""Tests for waveform generation and response feature extraction."""

from functools import partial

import jax
import jax.numpy as jnp
import pytest

from jaxley_extracellular.extracellular.response import (
    detect_spike,
    extract_response_features,
    extract_response_features_batch,
    firing_rate_hz,
    mean_isi_ms,
    spike_count,
    spike_latency_ms,
    spike_latency_steps,
)
from jaxley_extracellular.extracellular.waveforms import (
    make_biphasic_grid,
    make_biphasic_pulse,
    make_monophasic_pulse,
    make_pulse_train,
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


class TestPulseTrain:
    def test_shape(self) -> None:
        w = make_pulse_train(100.0, 0.5, 0.025, 10.0, frequency_hz=200.0)
        assert w.shape == (400,)  # 10ms / 0.025ms

    def test_freq_zero_gives_single_monophasic(self) -> None:
        """freq=0 produces exactly one pulse, matching make_monophasic_pulse."""
        w_train = make_pulse_train(100.0, 0.5, 0.025, 5.0, cathodic=True)
        w_mono = make_monophasic_pulse(100.0, 0.5, 0.025, 5.0, cathodic=True)
        assert float(jnp.max(jnp.abs(w_train - w_mono))) == 0.0

    def test_freq_zero_gives_single_biphasic(self) -> None:
        """freq=0 biphasic produces exactly one pulse, matching make_biphasic_pulse."""
        w_train = make_pulse_train(
            100.0, 0.5, 0.025, 5.0, biphasic=True, cathodic=True, interphase_ms=0.1
        )
        w_bi = make_biphasic_pulse(100.0, 0.5, 0.025, 5.0, cathodic_first=True, interphase_ms=0.1)
        assert float(jnp.max(jnp.abs(w_train - w_bi))) == 0.0

    def test_pulse_count(self) -> None:
        """200 Hz over 50 ms = 10 pulses."""
        w = make_pulse_train(100.0, 0.5, 0.025, 50.0, frequency_hz=200.0, cathodic=True)
        # Each pulse is 20 samples (0.5ms / 0.025ms)
        # Count transitions from zero to non-zero
        nonzero = w != 0
        onsets = nonzero[1:] & ~nonzero[:-1]
        # First pulse starts at t=0 so count it separately
        n_pulses = int(jnp.sum(onsets)) + (1 if bool(nonzero[0]) else 0)
        assert n_pulses == 10

    def test_frequency_spacing(self) -> None:
        """Pulses spaced at 1/frequency intervals."""
        freq = 100.0  # 10ms period
        dt = 0.025
        w = make_pulse_train(100.0, 0.5, dt, 50.0, frequency_hz=freq)
        period_samples = int(1000.0 / (freq * dt))  # 400
        # First sample of each period should be active
        for i in range(5):
            assert float(w[i * period_samples]) != 0.0

    def test_polarity_cathodic(self) -> None:
        w = make_pulse_train(100.0, 0.5, 0.025, 10.0, frequency_hz=200.0, cathodic=True)
        assert float(w.min()) == pytest.approx(-100.0)
        assert float(w.max()) == 0.0

    def test_polarity_anodic(self) -> None:
        w = make_pulse_train(100.0, 0.5, 0.025, 10.0, frequency_hz=200.0, cathodic=False)
        assert float(w.max()) == pytest.approx(100.0)
        assert float(w.min()) == 0.0

    def test_biphasic_charge_balance(self) -> None:
        w = make_pulse_train(100.0, 0.5, 0.025, 50.0, frequency_hz=100.0, biphasic=True)
        assert float(jnp.abs(w.sum())) == pytest.approx(0.0, abs=1e-4)

    def test_delay(self) -> None:
        w = make_pulse_train(100.0, 0.5, 0.025, 10.0, frequency_hz=200.0, delay_ms=1.0)
        onset = int(1.0 / 0.025)
        assert float(jnp.abs(w[:onset]).sum()) == 0.0
        assert float(w[onset]) != 0.0


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
        assert int(feats["spike_count"]) == 1
        assert float(feats["mean_isi_ms"]) == pytest.approx(-1.0)  # only 1 spike
        assert float(feats["firing_rate_hz"]) > 0.0

    def test_subthreshold_trace(self) -> None:
        v = jnp.array([-65.0, -60.0, -55.0, -60.0, -65.0])
        feats = extract_response_features(v, dt_ms=0.025)
        assert not bool(feats["spiked"])
        assert float(feats["latency_ms"]) == pytest.approx(5 * 0.025)
        assert int(feats["spike_count"]) == 0
        assert float(feats["mean_isi_ms"]) == pytest.approx(-1.0)
        assert float(feats["firing_rate_hz"]) == pytest.approx(0.0)

    def test_multi_spike_trace(self) -> None:
        v = jnp.array([-10.0, 10.0, -10.0, 10.0, -10.0, 10.0, -10.0])
        feats = extract_response_features(v, dt_ms=1.0)
        assert int(feats["spike_count"]) == 3
        assert float(feats["mean_isi_ms"]) == pytest.approx(2.0)
        assert float(feats["firing_rate_hz"]) == pytest.approx(3.0 / 0.007, rel=1e-4)


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
        for key in feats:
            assert feats[key].shape == (3,)

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


# ===================================================================
# Spike count tests
# ===================================================================


class TestSpikeCount:
    def test_no_spikes(self) -> None:
        v = jnp.array([-65.0, -60.0, -55.0, -60.0])
        assert int(spike_count(v, threshold_mV=0.0)) == 0

    def test_one_spike(self) -> None:
        v = jnp.array([-65.0, -60.0, 10.0, -30.0, -65.0])
        assert int(spike_count(v, threshold_mV=0.0)) == 1

    def test_three_spikes(self) -> None:
        # Three upward crossings through 0 mV
        v = jnp.array([-10.0, 10.0, -10.0, 10.0, -10.0, 10.0, -10.0])
        assert int(spike_count(v, threshold_mV=0.0)) == 3

    def test_vmap_compatible(self) -> None:
        v_batch = jnp.array(
            [
                [-10.0, 10.0, -10.0, 10.0, -10.0],  # 2 spikes
                [-65.0, -60.0, -55.0, -60.0, -65.0],  # 0 spikes
            ]
        )
        counts = jax.vmap(partial(spike_count, threshold_mV=0.0))(v_batch)
        assert int(counts[0]) == 2
        assert int(counts[1]) == 0


# ===================================================================
# Mean ISI tests
# ===================================================================


class TestMeanISI:
    def test_regular_train(self) -> None:
        # Spikes at indices 1, 3, 5 -> crossings at 0, 2, 4 (from below to above)
        # ISI = 2 steps each, dt=1.0ms -> mean ISI = 2.0 ms
        v = jnp.array([-10.0, 10.0, -10.0, 10.0, -10.0, 10.0, -10.0])
        isi = mean_isi_ms(v, dt_ms=1.0, threshold_mV=0.0)
        assert float(isi) == pytest.approx(2.0)

    def test_sentinel_for_zero_spikes(self) -> None:
        v = jnp.array([-65.0, -60.0, -55.0])
        isi = mean_isi_ms(v, dt_ms=0.025, threshold_mV=0.0)
        assert float(isi) == pytest.approx(-1.0)

    def test_sentinel_for_one_spike(self) -> None:
        v = jnp.array([-65.0, 10.0, -30.0])
        isi = mean_isi_ms(v, dt_ms=0.025, threshold_mV=0.0)
        assert float(isi) == pytest.approx(-1.0)

    def test_vmap_compatible(self) -> None:
        v_batch = jnp.array(
            [
                [-10.0, 10.0, -10.0, 10.0, -10.0],  # 2 spikes, ISI=2 steps
                [-65.0, -60.0, -55.0, -60.0, -65.0],  # 0 spikes -> sentinel
            ]
        )
        isi_fn = partial(mean_isi_ms, dt_ms=1.0, threshold_mV=0.0)
        isis = jax.vmap(isi_fn)(v_batch)
        assert float(isis[0]) == pytest.approx(2.0)
        assert float(isis[1]) == pytest.approx(-1.0)


# ===================================================================
# Firing rate tests
# ===================================================================


class TestFiringRate:
    def test_known_rate(self) -> None:
        # 3 spikes in 7 samples at dt=1.0ms = 7ms = 0.007s
        # rate = 3 / 0.007 ~= 428.57 Hz
        v = jnp.array([-10.0, 10.0, -10.0, 10.0, -10.0, 10.0, -10.0])
        rate = firing_rate_hz(v, dt_ms=1.0, threshold_mV=0.0)
        assert float(rate) == pytest.approx(3.0 / 0.007, rel=1e-4)

    def test_zero_for_no_spikes(self) -> None:
        v = jnp.array([-65.0, -60.0, -55.0, -60.0])
        rate = firing_rate_hz(v, dt_ms=0.025, threshold_mV=0.0)
        assert float(rate) == pytest.approx(0.0)

    def test_vmap_compatible(self) -> None:
        v_batch = jnp.array(
            [
                [-10.0, 10.0, -10.0, 10.0, -10.0],  # 2 spikes
                [-65.0, -60.0, -55.0, -60.0, -65.0],  # 0 spikes
            ]
        )
        rate_fn = partial(firing_rate_hz, dt_ms=1.0, threshold_mV=0.0)
        rates = jax.vmap(rate_fn)(v_batch)
        assert float(rates[0]) == pytest.approx(2.0 / 0.005, rel=1e-4)
        assert float(rates[1]) == pytest.approx(0.0)
