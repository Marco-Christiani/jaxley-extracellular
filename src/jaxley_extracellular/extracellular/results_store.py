"""xarray Dataset construction and Zarr I/O for sweep results."""

from __future__ import annotations

import datetime
import json
import subprocess
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import xarray as xr

ZarrMode = Literal["w", "w-", "a", "a-", "r+", "r"]


def make_flat_dataset(
    config_arrays: dict[str, np.ndarray[Any, Any]],
    metric_arrays: dict[str, np.ndarray[Any, Any]],
    attrs: dict[str, Any] | None = None,
) -> xr.Dataset:
    """Build a 1-D ``config``-indexed Dataset from parallel arrays.

    *config_arrays* become coordinates, *metric_arrays* become data variables.
    All arrays must share the same length (the ``config`` dimension).
    """
    coords = {k: ("config", v) for k, v in config_arrays.items()}
    data_vars = {k: ("config", v) for k, v in metric_arrays.items()}
    return xr.Dataset(
        data_vars=data_vars,
        coords=coords,
        attrs=attrs or {},
    )


def save_zarr(ds: xr.Dataset, path: Path, mode: ZarrMode = "w") -> Path:
    """Write *ds* to a Zarr store (blosc/lz4 default compression)."""
    # xarray accepts str paths at runtime; cast bridges narrower stubs.
    ds.to_zarr(cast(Any, str(path)), mode=mode)
    return path


def append_zarr(ds: xr.Dataset, path: Path, dim: str = "config") -> None:
    """Append *ds* to an existing Zarr store along *dim*."""
    ds.to_zarr(cast(Any, str(path)), mode="a", append_dim=dim)


def load_zarr(path: Path) -> xr.Dataset:
    """Load a Zarr store into an xarray Dataset."""
    ds: xr.Dataset = xr.open_zarr(str(path))
    return ds


def _get_git_hash() -> str:
    """Return the current git HEAD hash, or ``'unknown'``."""
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def sweep_metadata(config: dict[str, Any]) -> dict[str, Any]:
    """Capture git hash, ISO timestamp, and serialised config."""
    return {
        "git_hash": _get_git_hash(),
        "timestamp": datetime.datetime.now(tz=datetime.UTC).isoformat(),
        "config_json": json.dumps(config, default=str),
    }
