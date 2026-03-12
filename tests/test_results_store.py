"""Tests for xarray + Zarr results storage."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from jaxley_extracellular.extracellular.results_store import (
    append_zarr,
    load_zarr,
    make_flat_dataset,
    save_zarr,
    sweep_metadata,
)

# ------------------------------------------------------------------
# make_flat_dataset
# ------------------------------------------------------------------


class TestMakeFlatDataset:
    def test_dims_and_coords(self) -> None:
        configs = {"pulse_width_ms": np.array([0.1, 0.2, 0.5])}
        metrics = {"threshold_uA": np.array([100.0, 80.0, 60.0])}
        ds = make_flat_dataset(configs, metrics)

        assert "config" in ds.dims
        assert ds.sizes["config"] == 3
        assert "pulse_width_ms" in ds.coords
        assert "threshold_uA" in ds.data_vars

    def test_preserves_attrs(self) -> None:
        configs = {"pw": np.array([1.0])}
        metrics = {"thr": np.array([42.0])}
        ds = make_flat_dataset(configs, metrics, attrs={"version": "1.0"})
        assert ds.attrs["version"] == "1.0"

    def test_string_config_arrays(self) -> None:
        configs = {"waveform_type": np.array(["mono", "biphasic"])}
        metrics = {"threshold_uA": np.array([100.0, 200.0])}
        ds = make_flat_dataset(configs, metrics)
        assert ds.coords["waveform_type"].values[0] == "mono"

    def test_multiple_configs_and_metrics(self) -> None:
        configs = {
            "pw_ms": np.array([0.1, 0.2]),
            "distance_um": np.array([50.0, 100.0]),
        }
        metrics = {
            "threshold_uA": np.array([100.0, 200.0]),
            "charge_nC": np.array([10.0, 40.0]),
        }
        ds = make_flat_dataset(configs, metrics)
        assert len(ds.coords) == 2
        assert len(ds.data_vars) == 2

    def test_empty_attrs_default(self) -> None:
        ds = make_flat_dataset({"x": np.array([1])}, {"y": np.array([2])})
        assert ds.attrs == {}


# ------------------------------------------------------------------
# save_zarr + load_zarr round-trip
# ------------------------------------------------------------------


class TestZarrRoundTrip:
    def test_save_load(self, tmp_path: Path) -> None:
        configs = {"pw": np.array([0.1, 0.2, 0.5])}
        metrics = {"thr": np.array([100.0, 80.0, 60.0])}
        ds = make_flat_dataset(configs, metrics, attrs={"note": "test"})

        zarr_path = tmp_path / "test.zarr"
        save_zarr(ds, zarr_path)
        loaded = load_zarr(zarr_path)

        assert loaded.sizes["config"] == 3
        np.testing.assert_array_almost_equal(loaded["thr"].values, [100.0, 80.0, 60.0])
        assert loaded.attrs["note"] == "test"

    def test_save_returns_path(self, tmp_path: Path) -> None:
        ds = make_flat_dataset({"x": np.array([1])}, {"y": np.array([2])})
        result = save_zarr(ds, tmp_path / "out.zarr")
        assert result == tmp_path / "out.zarr"


# ------------------------------------------------------------------
# append_zarr
# ------------------------------------------------------------------


class TestAppendZarr:
    def test_extends_store(self, tmp_path: Path) -> None:
        zarr_path = tmp_path / "append.zarr"

        ds1 = make_flat_dataset({"pw": np.array([0.1, 0.2])}, {"thr": np.array([100.0, 80.0])})
        save_zarr(ds1, zarr_path)

        ds2 = make_flat_dataset({"pw": np.array([0.5])}, {"thr": np.array([60.0])})
        append_zarr(ds2, zarr_path)

        loaded = load_zarr(zarr_path)
        assert loaded.sizes["config"] == 3

    def test_preserves_existing_data(self, tmp_path: Path) -> None:
        zarr_path = tmp_path / "append2.zarr"

        ds1 = make_flat_dataset({"pw": np.array([0.1])}, {"thr": np.array([100.0])})
        save_zarr(ds1, zarr_path)

        ds2 = make_flat_dataset({"pw": np.array([0.2])}, {"thr": np.array([80.0])})
        append_zarr(ds2, zarr_path)

        loaded = load_zarr(zarr_path)
        np.testing.assert_array_almost_equal(loaded["thr"].values, [100.0, 80.0])


# ------------------------------------------------------------------
# sweep_metadata
# ------------------------------------------------------------------


class TestSweepMetadata:
    def test_has_required_fields(self) -> None:
        meta = sweep_metadata({"pulse_widths": [0.1, 0.2]})
        assert "git_hash" in meta
        assert "timestamp" in meta
        assert "config_json" in meta

    def test_config_json_is_valid(self) -> None:
        import json

        meta = sweep_metadata({"a": 1, "b": [2, 3]})
        parsed = json.loads(meta["config_json"])
        assert parsed["a"] == 1

    def test_timestamp_is_iso(self) -> None:
        import datetime

        meta = sweep_metadata({})
        # Should parse without error
        datetime.datetime.fromisoformat(meta["timestamp"])
