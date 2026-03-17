#!/usr/bin/env python
"""ECS parameter sweep: strength-duration curves across geometries and frequencies.

Key moving parts:
  1. JAX sharding for multi-device distribution
  2. xarray + Zarr for labeled, appendable storage
  3. Tracker protocol for experiment observability


Notes:
When running the remote tracking infrastructure, make sure `--tracking-uri` points to the correct (internal) IP
 for the tracking server in the VPC (which you can get from `tofu -chdir=infra/tofu output`)

In the case of MLFlow we have `--serve-artifacts` on the server (rather than specifying a specific artifact uri for the client
 in addition to the tracking uri) so the flow is:
  -> MLflowTracker.log_artifact(zarr_path)
    -> HTTP multipart upload to tracking server :5000
      -> tracking server streams to gs://bucket/mlflow/<run-id>/artifacts/
The compute instance never touches GCS directly. From the sweep's perspective it's just an HTTP POST to the tracking server URL,
 same as logging metrics.
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
from jaxley_extracellular.extracellular.system_monitor import Platform
from jaxley_extracellular.extracellular.tracker import (
    MLflowTracker,
    NullTracker,
    TrackerProtocol,
    collect_environment_params,
)

# -----------------------------------------------------------------------
# Sweep parameters
# -----------------------------------------------------------------------

PULSE_WIDTHS_MS = np.array([0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0])
ELECTRODE_DISTANCES_UM = np.array([25.0, 50.0, 100.0, 200.0])
FIBER_RADII_UM = np.array([5.0, 10.0, 20.0])
WAVEFORM_TYPES = ["monophasic_cathodic", "monophasic_anodic", "biphasic_cathodic_first"]
FREQUENCIES_HZ = np.array([0.0, 100.0, 200.0, 500.0])

# Binary search
AMP_LO = 0.0
AMP_HI = 5000.0  # uA
N_ITER = 14  # precision: 5000/2^14 ~= 0.3 uA

# Model
NCOMP = 50
CABLE_LENGTH_UM = 1250.0
DT_MS = 0.025
T_MS_SINGLE = 5.0  # for single-pulse (freq=0)
T_MS_TRAIN = 50.0  # for pulse trains (freq>0)
SIGMA = 0.3
RECORD_COMP = 0

# -----------------------------------------------------------------------
# Waveform factories (vmap-compatible, mask-based)
# -----------------------------------------------------------------------


class ECSExperimentProto(Protocol):
    def simulate_and_extract(self, waveform: jax.Array, record_comp: int) -> dict[str, Any]: ...


WaveformFactory = Callable[[jax.Array, jax.Array, jax.Array, jax.Array], jax.Array]


def _make_mono_cathodic(
    amplitude: jax.Array, pw_steps: jax.Array, period_steps: jax.Array, t_idx: jax.Array
) -> jax.Array:
    phase = t_idx % period_steps
    return jnp.where(phase < pw_steps, -amplitude, 0.0)


def _make_mono_anodic(
    amplitude: jax.Array, pw_steps: jax.Array, period_steps: jax.Array, t_idx: jax.Array
) -> jax.Array:
    phase = t_idx % period_steps
    return jnp.where(phase < pw_steps, amplitude, 0.0)


def _make_biphasic_cathodic_first(
    amplitude: jax.Array,
    pw_steps: jax.Array,
    period_steps: jax.Array,
    t_idx: jax.Array,
) -> jax.Array:
    phase = t_idx % period_steps
    cathodic = jnp.where(phase < pw_steps, -amplitude, 0.0)
    anodic = jnp.where((phase >= pw_steps) & (phase < 2 * pw_steps), amplitude, 0.0)
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
    period_steps_arr: jax.Array,
    lo: jax.Array,
    hi: jax.Array,
    T: int,
    n_iter: int,
) -> jax.Array:
    """Binary search over amplitude, vmapped across pulse widths."""
    t_idx = jnp.arange(T)

    @jax.jit
    @jax.vmap
    def _test(amp: jax.Array, pw_steps: jax.Array, per_steps: jax.Array) -> jax.Array:
        w = factory(amp, pw_steps, per_steps, t_idx)
        feats = exp.simulate_and_extract(w, RECORD_COMP)
        return cast(jax.Array, feats["spiked"])

    for _ in range(n_iter):
        mid = (lo + hi) / 2.0
        spiked: jax.Array = cast(jax.Array, _test(mid, pw_steps_arr, period_steps_arr))
        lo = jnp.where(spiked, lo, mid)
        hi = jnp.where(spiked, mid, hi)

    return (lo + hi) / 2.0


def _extract_features_at_threshold(
    exp: ECSExperimentProto,
    factory: WaveformFactory,
    thresholds: jax.Array,
    pw_steps_arr: jax.Array,
    period_steps_arr: jax.Array,
    T: int,
) -> dict[str, jax.Array]:
    """Re-simulate at found thresholds and return full feature dicts."""
    t_idx = jnp.arange(T)

    @jax.jit
    @jax.vmap
    def _run(amp: jax.Array, pw_steps: jax.Array, per_steps: jax.Array) -> dict[str, jax.Array]:
        w = factory(amp, pw_steps, per_steps, t_idx)
        return exp.simulate_and_extract(w, RECORD_COMP)

    return cast(dict[str, jax.Array], _run(thresholds, pw_steps_arr, period_steps_arr))


# -----------------------------------------------------------------------
# Resume support
# -----------------------------------------------------------------------


def _already_completed(zarr_path: Path) -> set[tuple[str, float, float, float]]:
    """Load existing Zarr coordinates and return completed (wtype, dist, radius, freq) keys."""
    if not zarr_path.exists():
        return set()
    ds = load_zarr(zarr_path)
    completed: set[tuple[str, float, float, float]] = set()
    wt = ds.coords["waveform_type"].values
    dist = ds.coords["electrode_distance_um"].values
    rad = ds.coords["fiber_radius_um"].values
    freq = ds.coords["frequency_hz"].values
    for i in range(ds.sizes["config"]):
        completed.add((str(wt[i]), float(dist[i]), float(rad[i]), float(freq[i])))
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
    batch_size: int = 64,
) -> Path:
    """Run the full sweep and return the Zarr output path."""
    zarr_path = outdir / "sweep.zarr"

    # Sweep config for metadata / logging
    sweep_config: dict[str, Any] = {
        "pulse_widths_ms": PULSE_WIDTHS_MS.tolist(),
        "electrode_distances_um": ELECTRODE_DISTANCES_UM.tolist(),
        "fiber_radii_um": FIBER_RADII_UM.tolist(),
        "waveform_types": WAVEFORM_TYPES,
        "frequencies_hz": FREQUENCIES_HZ.tolist(),
        "amp_lo": AMP_LO,
        "amp_hi": AMP_HI,
        "n_iter": N_ITER,
        "ncomp": NCOMP,
        "cable_length_um": CABLE_LENGTH_UM,
        "dt_ms": DT_MS,
        "t_ms_single": T_MS_SINGLE,
        "t_ms_train": T_MS_TRAIN,
        "sigma": SIGMA,
        "batch_size": batch_size,
    }

    total_configs = (
        len(PULSE_WIDTHS_MS)
        * len(WAVEFORM_TYPES)
        * len(FREQUENCIES_HZ)
        * len(ELECTRODE_DISTANCES_UM)
        * len(FIBER_RADII_UM)
    )
    print(f"Sweep: {total_configs} configs")
    print(f"  pulse widths: {PULSE_WIDTHS_MS} ms")
    print(f"  waveform types: {WAVEFORM_TYPES}")
    print(f"  frequencies: {FREQUENCIES_HZ} Hz")
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
        print(f"Resuming: {len(completed)} groups already done")

    pw_steps_arr = jnp.array([int(pw / DT_MS) for pw in PULSE_WIDTHS_MS])
    pw_steps_sharded = shard_batch(pw_steps_arr, sharding)

    meta = sweep_metadata(sweep_config)
    configs_done = 0
    configs_skipped = 0
    batch_idx = 0
    wtype_max_len = max(len(w) for w in WAVEFORM_TYPES)

    sweep_t0 = time.time()

    with tracker:
        tracker.log_params(sweep_config)
        tracker.log_params(collect_environment_params())
        tracker.set_status("running")

        for freq_hz in FREQUENCIES_HZ:
            T_ms = T_MS_SINGLE if freq_hz == 0.0 else T_MS_TRAIN
            T = int(T_ms / DT_MS)

            period_steps_val = int(1000.0 / (freq_hz * DT_MS)) if freq_hz > 0.0 else T

            N = len(PULSE_WIDTHS_MS)
            period_steps_arr = jnp.full(N, period_steps_val, dtype=jnp.int32)
            period_steps_sharded = shard_batch(period_steps_arr, sharding)

            for dist_um in ELECTRODE_DISTANCES_UM:
                for radius_um in FIBER_RADII_UM:
                    exp = make_hh_cable_experiment(
                        ncomp=NCOMP,
                        cable_length_um=CABLE_LENGTH_UM,
                        radius_um=float(radius_um),
                        electrode_distance_um=float(dist_um),
                        sigma=SIGMA,
                        dt_ms=DT_MS,
                        T_ms=T_ms,
                    )

                    for wtype in WAVEFORM_TYPES:
                        key = (wtype, float(dist_um), float(radius_um), float(freq_hz))
                        if key in completed:
                            configs_skipped += N
                            print(f"  SKIP {wtype} dist={dist_um} r={radius_um} f={freq_hz}")
                            continue

                        factory = WAVEFORM_FACTORIES[wtype]
                        lo = shard_batch(jnp.full(N, AMP_LO, dtype=jnp.float32), sharding)
                        hi = shard_batch(jnp.full(N, AMP_HI, dtype=jnp.float32), sharding)

                        t0 = time.time()
                        thresholds = _find_thresholds_batched(
                            exp,
                            factory,
                            pw_steps_sharded,
                            period_steps_sharded,
                            lo,
                            hi,
                            T,
                            N_ITER,
                        )
                        # Extract firing pattern features at threshold
                        feats = _extract_features_at_threshold(
                            exp,
                            factory,
                            thresholds,
                            pw_steps_sharded,
                            period_steps_sharded,
                            T,
                        )
                        elapsed = time.time() - t0
                        thresholds_np = np.asarray(thresholds)[:N]

                        config_arrays = {
                            "pulse_width_ms": np.array(PULSE_WIDTHS_MS),
                            "waveform_type": np.array([wtype] * N, dtype=f"<U{wtype_max_len}"),
                            "electrode_distance_um": np.full(N, float(dist_um)),
                            "fiber_radius_um": np.full(N, float(radius_um)),
                            "frequency_hz": np.full(N, float(freq_hz)),
                        }
                        metric_arrays = {
                            "threshold_uA": thresholds_np,
                            "charge_nC": thresholds_np * np.array(PULSE_WIDTHS_MS),
                            "spike_count": np.asarray(feats["spike_count"])[:N],
                            "mean_isi_ms": np.asarray(feats["mean_isi_ms"])[:N],
                            "firing_rate_hz": np.asarray(feats["firing_rate_hz"])[:N],
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

                        # Per-batch threshold metrics with structured keys
                        for i, pw_ms in enumerate(PULSE_WIDTHS_MS):
                            thr = float(thresholds_np[i])
                            metric_key = (
                                f"threshold/{wtype}/d{dist_um:.0f}/r{radius_um:.0f}/pw{pw_ms:.2f}"
                            )
                            tracker.log_metrics({metric_key: thr}, step=batch_idx)
                            print(
                                f"  {wtype:30s}  f={freq_hz:5.0f}Hz  "
                                f"dist={dist_um:5.0f}  r={radius_um:4.0f}  "
                                f"pw={pw_ms:.2f}ms  thr={thr:8.1f}uA"
                            )
                        print(f"  [{wtype}] batch: {elapsed:.1f}s for {N} configs\n")

        # Summary metrics
        total_time = time.time() - sweep_t0
        tracker.log_metrics(
            {
                "summary/total_time_s": total_time,
                "summary/configs_computed": float(configs_done),
                "summary/configs_skipped": float(configs_skipped),
                "summary/total_configs": float(total_configs),
            }
        )

        tracker.log_artifact(zarr_path)
        tracker.set_status("completed")

    print(f"\nSaved {configs_done} results to {zarr_path}")
    return zarr_path


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="ECS parameter sweep")
    parser.add_argument("--outdir", type=str, default="results/sweeps")
    parser.add_argument("--tracker", choices=["null", "mlflow"], default="null")
    parser.add_argument(
        "--tracking-uri",
        type=str,
        default="http://127.0.0.1:5000",
        help="Tracking server URI (default: http://127.0.0.1:5000)",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--platform",
        choices=["auto", "gpu", "tpu", "cpu"],
        default="auto",
        help="Platform for system metrics (default: auto-detect)",
    )
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    platform: Platform | None = None
    if args.platform != "auto":
        platform = Platform[args.platform.upper()]

    tracker: TrackerProtocol
    if args.tracker == "mlflow":
        tracker = MLflowTracker(tracking_uri=args.tracking_uri, platform=platform)
    else:
        tracker = NullTracker()

    run_sweep(outdir, tracker, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
