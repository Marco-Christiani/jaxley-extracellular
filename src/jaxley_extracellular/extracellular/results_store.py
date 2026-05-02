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

    Parameters
    ----------
    config_arrays : dict[str, numpy.ndarray]
        Per-config parameter values, keyed by parameter name. Become
        coordinates on the ``config`` dimension.
    metric_arrays : dict[str, numpy.ndarray]
        Per-config measured values, keyed by metric name. Become data
        variables on the ``config`` dimension.
    attrs : dict, optional
        Attributes to attach to the resulting Dataset.

    Returns
    -------
    xarray.Dataset
        Dataset with a single ``config`` dimension. All input arrays
        must share the same length along that dimension.
    """
    coords = {k: ("config", v) for k, v in config_arrays.items()}
    data_vars = {k: ("config", v) for k, v in metric_arrays.items()}
    return xr.Dataset(
        data_vars=data_vars,
        coords=coords,
        attrs=attrs or {},
    )


def save_zarr(ds: xr.Dataset, path: Path, mode: ZarrMode = "w") -> Path:
    """Write a Dataset to a Zarr store with default compression.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to write.
    path : pathlib.Path
        Output Zarr store directory.
    mode : {'w', 'w-', 'a', 'a-', 'r+', 'r'}, optional
        Zarr write mode (default ``'w'``, overwrite).

    Returns
    -------
    pathlib.Path
        The store path that was written to.
    """
    # xarray accepts str paths at runtime. Cast bridges narrower stubs.
    ds.to_zarr(cast(Any, str(path)), mode=mode)
    return path


def append_zarr(ds: xr.Dataset, path: Path, dim: str = "config") -> None:
    """Append a Dataset to an existing Zarr store along ``dim``.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to append.
    path : pathlib.Path
        Existing Zarr store directory.
    dim : str, optional
        Dimension to append along (default ``"config"``).
    """
    ds.to_zarr(cast(Any, str(path)), mode="a", append_dim=dim)


def load_zarr(path: Path) -> xr.Dataset:
    """Load a Zarr store into an xarray Dataset.

    Parameters
    ----------
    path : pathlib.Path
        Path to a Zarr store directory.

    Returns
    -------
    xarray.Dataset
        Lazy Dataset backed by the Zarr store.
    """
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
    """Build a metadata dict with git hash, timestamp, and config.

    Parameters
    ----------
    config : dict
        Sweep configuration. Serialised to JSON via ``str`` fallback.

    Returns
    -------
    dict
        Mapping with keys ``git_hash`` (current HEAD or
        ``"unknown"``), ``timestamp`` (UTC ISO 8601), and
        ``config_json``. Suitable for an xarray Dataset's ``attrs``.
    """
    return {
        "git_hash": _get_git_hash(),
        "timestamp": datetime.datetime.now(tz=datetime.UTC).isoformat(),
        "config_json": json.dumps(config, default=str),
    }
