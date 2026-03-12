#!/usr/bin/env python
"""Expanded ECS sweep: strength-duration curves across geometries.

Uses three pillars:
  1. JAX sharding for multi-device distribution
  2. xarray + Zarr for labeled, appendable storage
  3. Tracker protocol for experiment observability

Usage:
    python scripts/sweep.py [--outdir results/sweeps] [--tracker mlflow|null]
                            [--batch-size 64] [--resume path/to.zarr]
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol, cast

import jax
import jax.numpy as jnp
import numpy as np

from jaxley_extracellular.extracellular.experiment import make_hh_cable_experiment
from jaxley_extracellular.extracellular.results_store import (
    append_zarr,
    load_zarr,
    make_flat_dataset,
    save_zarr,
    sweep_metadata,
)
from jaxley_extracellular.extracellular.sharding import (
    config_sharding,
    make_device_mesh,
    shard_batch,
)
from jaxley_extracellular.extracellular.tracker import (
    MLflowTracker,
    NullTracker,
    TrackerProtocol,
)

# -----------------------------------------------------------------------
# Sweep parameters
# -----------------------------------------------------------------------

PULSE_WIDTHS_MS = np.array([0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0])
ELECTRODE_DISTANCES_UM = np.array([25.0, 50.0, 100.0, 200.0])
FIBER_RADII_UM = np.array([5.0, 10.0, 20.0])
WAVEFORM_TYPES = ["monophasic_cathodic", "monophasic_anodic", "biphasic_cathodic_first"]

# Binary search
AMP_LO = 0.0
AMP_HI = 5000.0  # uA
N_ITER = 14  # precision: 5000/2^14 ~= 0.3 uA

# Model
NCOMP = 50
CABLE_LENGTH_UM = 1250.0
DT_MS = 0.025
T_MS = 5.0
SIGMA = 0.3
RECORD_COMP = 0

# -----------------------------------------------------------------------
# Waveform factories (vmap-compatible, mask-based)
# -----------------------------------------------------------------------


class ECSExperimentProto(Protocol):
    def simulate_and_extract(self, waveform: jax.Array, record_comp: int) -> dict[str, Any]: ...


WaveformFactory = Callable[[jax.Array, jax.Array, jax.Array], jax.Array]


def _make_mono_cathodic(amplitude: jax.Array, pw_steps: jax.Array, t_idx: jax.Array) -> jax.Array:
    return jnp.where(t_idx < pw_steps, -amplitude, 0.0)


def _make_mono_anodic(amplitude: jax.Array, pw_steps: jax.Array, t_idx: jax.Array) -> jax.Array:
    return jnp.where(t_idx < pw_steps, amplitude, 0.0)


def _make_biphasic_cathodic_first(
    amplitude: jax.Array, pw_steps: jax.Array, t_idx: jax.Array
) -> jax.Array:
    cathodic = jnp.where(t_idx < pw_steps, -amplitude, 0.0)
    anodic = jnp.where((t_idx >= pw_steps) & (t_idx < 2 * pw_steps), amplitude, 0.0)
    return cathodic + anodic


WAVEFORM_FACTORIES: dict[str, WaveformFactory] = {
    "monophasic_cathodic": _make_mono_cathodic,
    "monophasic_anodic": _make_mono_anodic,
    "biphasic_cathodic_first": _make_biphasic_cathodic_first,
}


# -----------------------------------------------------------------------
# Batched binary search (accepts pre-sharded lo/hi)
# -----------------------------------------------------------------------


def _find_thresholds_batched(
    exp: ECSExperimentProto,
    factory: WaveformFactory,
    pw_steps_arr: jax.Array,
    lo: jax.Array,
    hi: jax.Array,
    T: int,
    n_iter: int,
) -> jax.Array:
    """Binary search over amplitude, vmapped across pulse widths."""
    t_idx = jnp.arange(T)

    @jax.jit
    @jax.vmap
    def _test(amp: jax.Array, pw_steps: jax.Array) -> jax.Array:
        w = factory(amp, pw_steps, t_idx)
        feats = exp.simulate_and_extract(w, RECORD_COMP)
        return cast(jax.Array, feats["spiked"])

    for _ in range(n_iter):
        mid = (lo + hi) / 2.0
        spiked: jax.Array = cast(jax.Array, _test(mid, pw_steps_arr))
        lo = jnp.where(spiked, lo, mid)
        hi = jnp.where(spiked, mid, hi)

    return (lo + hi) / 2.0


# -----------------------------------------------------------------------
# Resume support
# -----------------------------------------------------------------------


def _already_completed(zarr_path: Path) -> set[tuple[str, float, float]]:
    """Load existing Zarr coordinates and return completed (wtype, dist, radius) keys."""
    if not zarr_path.exists():
        return set()
    ds = load_zarr(zarr_path)
    completed: set[tuple[str, float, float]] = set()
    wt = ds.coords["waveform_type"].values
    dist = ds.coords["electrode_distance_um"].values
    rad = ds.coords["fiber_radius_um"].values
    for i in range(ds.sizes["config"]):
        completed.add((str(wt[i]), float(dist[i]), float(rad[i])))
    return completed


def _write_batch(
    ds: Any,
    zarr_path: Path,
) -> None:
    """Save or append a batch to the Zarr store."""
    if zarr_path.exists():
        append_zarr(ds, zarr_path)
    else:
        zarr_path.parent.mkdir(parents=True, exist_ok=True)
        save_zarr(ds, zarr_path)


# -----------------------------------------------------------------------
# Main sweep
# -----------------------------------------------------------------------


def run_sweep(
    outdir: Path,
    tracker: TrackerProtocol,
    resume_path: Path | None = None,
    batch_size: int = 64,
) -> Path:
    """Run the full sweep and return the Zarr output path."""
    zarr_path = outdir / "sweep.zarr"
    if resume_path is not None:
        zarr_path = resume_path

    # Sweep config for metadata / logging
    sweep_config: dict[str, Any] = {
        "pulse_widths_ms": PULSE_WIDTHS_MS.tolist(),
        "electrode_distances_um": ELECTRODE_DISTANCES_UM.tolist(),
        "fiber_radii_um": FIBER_RADII_UM.tolist(),
        "waveform_types": WAVEFORM_TYPES,
        "amp_lo": AMP_LO,
        "amp_hi": AMP_HI,
        "n_iter": N_ITER,
        "ncomp": NCOMP,
        "cable_length_um": CABLE_LENGTH_UM,
        "dt_ms": DT_MS,
        "t_ms": T_MS,
        "sigma": SIGMA,
        "batch_size": batch_size,
    }

    total_configs = (
        len(PULSE_WIDTHS_MS)
        * len(WAVEFORM_TYPES)
        * len(ELECTRODE_DISTANCES_UM)
        * len(FIBER_RADII_UM)
    )
    print(f"Sweep: {total_configs} configs")
    print(f"  pulse widths: {PULSE_WIDTHS_MS} ms")
    print(f"  waveform types: {WAVEFORM_TYPES}")
    print(f"  electrode distances: {ELECTRODE_DISTANCES_UM} um")
    print(f"  fiber radii: {FIBER_RADII_UM} um")
    print(f"  binary search: {N_ITER} iterations, bracket [{AMP_LO}, {AMP_HI}] uA")
    print()

    # Sharding setup
    mesh = make_device_mesh()
    sharding = config_sharding(mesh)
    print(f"Device mesh: {mesh.devices.shape} devices")

    # Resume
    completed = _already_completed(zarr_path)
    if completed:
        print(f"Resuming: {len(completed)} (wtype, dist, radius) groups already done")

    T = int(T_MS / DT_MS)
    pw_steps_arr = jnp.array([int(pw / DT_MS) for pw in PULSE_WIDTHS_MS])
    pw_steps_sharded = shard_batch(pw_steps_arr, sharding)

    meta = sweep_metadata(sweep_config)
    configs_done = 0
    batch_idx = 0

    with tracker:
        tracker.log_params(sweep_config)
        tracker.set_status("running")

        for dist_um in ELECTRODE_DISTANCES_UM:
            for radius_um in FIBER_RADII_UM:
                exp = make_hh_cable_experiment(
                    ncomp=NCOMP,
                    cable_length_um=CABLE_LENGTH_UM,
                    radius_um=float(radius_um),
                    electrode_distance_um=float(dist_um),
                    sigma=SIGMA,
                    dt_ms=DT_MS,
                    T_ms=T_MS,
                )

                for wtype in WAVEFORM_TYPES:
                    key = (wtype, float(dist_um), float(radius_um))
                    if key in completed:
                        print(f"  SKIP {wtype} dist={dist_um} r={radius_um}")
                        continue

                    factory = WAVEFORM_FACTORIES[wtype]
                    N = len(PULSE_WIDTHS_MS)
                    lo = shard_batch(jnp.full(N, AMP_LO, dtype=jnp.float32), sharding)
                    hi = shard_batch(jnp.full(N, AMP_HI, dtype=jnp.float32), sharding)

                    t0 = time.time()
                    thresholds = _find_thresholds_batched(
                        exp, factory, pw_steps_sharded, lo, hi, T, N_ITER
                    )
                    elapsed = time.time() - t0
                    thresholds_np = np.asarray(thresholds)[:N]  # trim padding

                    # Build batch dataset
                    config_arrays = {
                        "pulse_width_ms": np.array(PULSE_WIDTHS_MS),
                        "waveform_type": np.array(
                            [wtype] * N, dtype=f"<U{max(len(w) for w in WAVEFORM_TYPES)}"
                        ),
                        "electrode_distance_um": np.full(N, float(dist_um)),
                        "fiber_radius_um": np.full(N, float(radius_um)),
                    }
                    metric_arrays = {
                        "threshold_uA": thresholds_np,
                        "charge_nC": thresholds_np * np.array(PULSE_WIDTHS_MS),
                        "time_s": np.full(N, elapsed / N),
                    }
                    ds = make_flat_dataset(config_arrays, metric_arrays, attrs=meta)
                    _write_batch(ds, zarr_path)

                    configs_done += N
                    batch_idx += 1
                    tracker.log_metrics(
                        {"configs_done": float(configs_done), "batch_time_s": elapsed},
                        step=batch_idx,
                    )

                    for i, pw_ms in enumerate(PULSE_WIDTHS_MS):
                        thr = float(thresholds_np[i])
                        print(
                            f"  {wtype:30s}  dist={dist_um:5.0f}  r={radius_um:4.0f}  "
                            f"pw={pw_ms:.2f}ms  thr={thr:8.1f}uA"
                        )
                    print(f"  [{wtype}] batch: {elapsed:.1f}s for {N} configs\n")

        tracker.log_artifact_path(zarr_path)
        tracker.log_artifact(zarr_path)
        tracker.set_status("completed")

    print(f"\nSaved {configs_done} results to {zarr_path}")
    return zarr_path


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Expanded ECS waveform sweep")
    parser.add_argument("--outdir", type=str, default="results/sweeps")
    parser.add_argument("--tracker", choices=["null", "mlflow"], default="null")
    parser.add_argument(
        "--tracking-uri",
        type=str,
        default="http://127.0.0.1:5000",
        help="Tracking server URI (default: http://127.0.0.1:5000)",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--resume", type=str, default=None, help="Path to existing .zarr to resume")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    tracker: TrackerProtocol
    if args.tracker == "mlflow":
        tracker = MLflowTracker(tracking_uri=args.tracking_uri)
    else:
        tracker = NullTracker()

    resume_path = Path(args.resume) if args.resume else None
    run_sweep(outdir, tracker, resume_path=resume_path, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
