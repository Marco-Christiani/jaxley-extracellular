"""Shared helpers for Jaxley vs NEURON parity scripts.

This module intentionally stays lightweight so both NEURON-side and Jaxley-side
parity scripts can import it without pulling in either simulator.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np


def provenance_fields(
    *,
    platform: str,
    hardware_label: str,
    script_path: str | os.PathLike[str],
) -> dict[str, str]:
    """Identity metadata to attach to ``np.savez`` outputs.

    Pass-through ``**provenance_fields(...)`` lets a figure pipeline read
    ``platform`` (``"tpu"`` / ``"gpu"`` / ``"cpu"``), ``hardware_label``
    (e.g.\\ ``"v5e single-chip"``), the script that produced the file, and
    the git hash, without the figure code having to hand-type any of it.
    """
    rel = Path(script_path).resolve()
    try:
        rel = rel.relative_to(Path.cwd())
    except ValueError:
        pass
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        git_hash = "unknown"
    return {
        "platform": platform,
        "hardware_label": hardware_label,
        "script_relpath": str(rel),
        "git_hash": git_hash,
        "python_version": sys.version.split()[0],
    }


def interp_xyz_on_polyline(poly_xyz: np.ndarray, frac: float) -> np.ndarray:
    """Interpolate xyz at fractional arc length along a polyline."""
    seg_len = np.sqrt(((poly_xyz[1:] - poly_xyz[:-1]) ** 2).sum(axis=1))
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = float(cum[-1])
    if total <= 0:
        return poly_xyz[0].copy()
    target = float(np.clip(frac, 0.0, 1.0)) * total
    return np.array([np.interp(target, cum, poly_xyz[:, i]) for i in range(3)], dtype=float)


def interpolate_branch_voltage(
    full_v: np.ndarray,
    branch,
    frac: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate branch voltage at fractional location from center traces."""
    nodes = branch.nodes.sort_values("local_comp_index")
    gci = nodes["global_comp_index"].to_numpy(dtype=int)
    ncomp = len(gci)
    centers = (np.arange(ncomp, dtype=float) + 0.5) / ncomp
    frac = float(np.clip(frac, 0.0, 1.0))

    if frac <= centers[0] or ncomp == 1:
        return full_v[gci[0]], np.array([gci[0], gci[0]], dtype=int), np.array([1.0, 0.0], dtype=float)
    if frac >= centers[-1]:
        return full_v[gci[-1]], np.array([gci[-1], gci[-1]], dtype=int), np.array([1.0, 0.0], dtype=float)

    hi = int(np.searchsorted(centers, frac, side="right"))
    lo = hi - 1
    c0, c1 = centers[lo], centers[hi]
    w1 = (frac - c0) / (c1 - c0)
    w0 = 1.0 - w1
    trace = w0 * full_v[gci[lo]] + w1 * full_v[gci[hi]]
    return trace, np.array([gci[lo], gci[hi]], dtype=int), np.array([w0, w1], dtype=float)


def segment_center_xyz(seg) -> np.ndarray:
    """Cartesian coordinates at normalized position seg.x within seg.sec."""
    sec = seg.sec
    n = sec.n3d()
    if n == 0:
        return np.array([0.0, 0.0, 0.0], dtype=float)
    arc = np.array([sec.arc3d(i) for i in range(n)], dtype=float)
    xs = np.array([sec.x3d(i) for i in range(n)], dtype=float)
    ys = np.array([sec.y3d(i) for i in range(n)], dtype=float)
    zs = np.array([sec.z3d(i) for i in range(n)], dtype=float)
    total = float(arc[-1]) if arc[-1] > 0 else float(sec.L)
    target = float(seg.x) * total
    return np.array(
        [
            np.interp(target, arc, xs),
            np.interp(target, arc, ys),
            np.interp(target, arc, zs),
        ],
        dtype=float,
    )


def waveform_metrics(
    t_ref: np.ndarray,
    v_ref: np.ndarray,
    t_other: np.ndarray,
    v_other: np.ndarray,
) -> dict[str, float]:
    """Resample `v_other` onto `t_ref`, then compute waveform metrics."""
    v_other_on_ref = np.interp(t_ref, t_other, v_other)
    rmse = float(np.sqrt(np.mean((v_ref - v_other_on_ref) ** 2)))
    mae = float(np.mean(np.abs(v_ref - v_other_on_ref)))
    max_abs = float(np.max(np.abs(v_ref - v_other_on_ref)))
    corr = float(np.corrcoef(v_ref, v_other_on_ref)[0, 1])
    return {"rmse_mV": rmse, "mae_mV": mae, "max_abs_mV": max_abs, "pearson_r": corr}
