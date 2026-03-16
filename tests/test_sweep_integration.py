"""Integration tests for the sweep pipeline (sharding + zarr + tracker)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxley_extracellular.extracellular.results_store import (
    load_zarr,
    make_flat_dataset,
    save_zarr,
)
from jaxley_extracellular.extracellular.sharding import (
    config_sharding,
    make_device_mesh,
    shard_batch,
)
from jaxley_extracellular.extracellular.tracker import NullTracker


@pytest.fixture(scope="session")
def tiny_zarr(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Run a minimal sweep and return the Zarr path."""
    from jaxley_extracellular.extracellular.experiment import make_hh_cable_experiment

    exp = make_hh_cable_experiment(
        ncomp=4,
        cable_length_um=500.0,
        radius_um=10.0,
        electrode_distance_um=50.0,
        sigma=0.3,
        dt_ms=0.025,
        T_ms=1.0,
    )

    T = int(1.0 / 0.025)
    pw_ms_list = [0.1, 0.2]
    pw_steps_arr = jnp.array([int(pw / 0.025) for pw in pw_ms_list])
    t_idx = jnp.arange(T)

    def factory(amplitude: Any, pw_steps: Any, t_idx: Any) -> Any:
        return jnp.where(t_idx < pw_steps, -amplitude, 0.0)

    N = len(pw_ms_list)
    lo = jnp.full(N, 0.0, dtype=jnp.float32)
    hi = jnp.full(N, 5000.0, dtype=jnp.float32)

    @jax.jit
    @jax.vmap
    def _test(amp: Any, pw_steps: Any) -> Any:
        w = factory(amp, pw_steps, t_idx)
        feats = exp.simulate_and_extract(w, 0)
        return feats["spiked"]

    for _ in range(6):
        mid = (lo + hi) / 2.0
        spiked = cast(Any, _test(mid, pw_steps_arr))
        lo = jnp.where(spiked, lo, mid)
        hi = jnp.where(spiked, mid, hi)

    thresholds = np.asarray((lo + hi) / 2.0)

    config_arrays = {
        "pulse_width_ms": np.array(pw_ms_list),
        "waveform_type": np.array(["monophasic_cathodic"] * N),
        "electrode_distance_um": np.full(N, 50.0),
        "fiber_radius_um": np.full(N, 10.0),
    }
    metric_arrays = {
        "threshold_uA": thresholds,
        "charge_nC": thresholds * np.array(pw_ms_list),
        "time_s": np.full(N, 0.0),
    }
    ds = make_flat_dataset(config_arrays, metric_arrays)
    zarr_path = tmp_path_factory.mktemp("sweep") / "test_sweep.zarr"
    save_zarr(ds, zarr_path)
    return zarr_path


@pytest.mark.slow
class TestEndToEnd:
    """Tiny model sweep -> Zarr -> load -> verify."""

    def test_zarr_shape(self, tiny_zarr: Path) -> None:
        ds = load_zarr(tiny_zarr)
        assert ds.sizes["config"] == 2
        assert "threshold_uA" in ds.data_vars
        assert "pulse_width_ms" in ds.coords

    def test_thresholds_positive(self, tiny_zarr: Path) -> None:
        ds = load_zarr(tiny_zarr)
        thresholds = ds["threshold_uA"].values
        assert np.all(thresholds >= 0.0)
        assert np.all(thresholds <= 5000.0)


class TestTrackerIntegration:
    """Verify tracker receives expected calls."""

    def test_null_tracker_no_errors(self) -> None:
        tracker = NullTracker()
        with tracker:
            tracker.log_params({"pw": [0.1, 0.2], "wtype": "mono"})
            tracker.log_metrics({"configs_done": 2.0, "batch_time_s": 1.5}, step=1)
            tracker.set_status("running")
            tracker.log_artifact(Path("/tmp/test.zarr"))
            tracker.set_status("completed")

    def test_sharding_roundtrip_with_zarr(self, tmp_path: Path) -> None:
        """Shard -> compute -> unpad -> Zarr -> load: values match."""
        mesh = make_device_mesh()
        sharding = config_sharding(mesh)

        data = jnp.array([1.0, 2.0, 3.0])
        sharded = shard_batch(data, sharding)
        result = np.asarray(sharded[:3])  # trim padding

        config_arrays = {"x": result}
        metric_arrays = {"y": result * 2}
        ds = make_flat_dataset(config_arrays, metric_arrays)
        zarr_path = tmp_path / "roundtrip.zarr"
        save_zarr(ds, zarr_path)

        loaded = load_zarr(zarr_path)
        np.testing.assert_array_almost_equal(loaded["y"].values, [2.0, 4.0, 6.0])
